/**
 * NPC Voice Pipeline Server
 *
 * WebSocket endpoint: ws://localhost:3001
 * - Accepts mic PCM audio from browser
 * - Streams audio → Deepgram ASR → OpenAI GPT-4o → ElevenLabs TTS
 * - Simultaneously processes TTS audio through NVIDIA Audio2Face-3D for blendshapes
 * - Returns audio chunks + timestamped ARKit blendshape frames to browser
 *
 * Message protocol:
 *   Client → Server:
 *     First message:  JSON { type:'init', personalityPrompt, openAiKey, deepgramKey,
 *                                        elevenLabsKey, voiceId, nvidiaApiKey? }
 *     Subsequent:     Binary (Int16 PCM, 16kHz, mono)
 *
 *   Server → Client:
 *     JSON:   { type:'transcript', text }
 *     JSON:   { type:'response_start' }
 *     Binary: audio bytes (MP3 chunks from ElevenLabs)
 *     JSON:   { type:'blendshapes', timestamp:number, values:Record<string,number> }
 *     JSON:   { type:'response_end' }
 *     JSON:   { type:'error', message }
 */

import 'dotenv/config';
import { WebSocketServer, WebSocket } from 'ws';
import { createServer } from 'http';
import express from 'express';
import cors from 'cors';
import { DeepgramClient } from '@deepgram/sdk';
import type { V1Socket } from '@deepgram/sdk/dist/cjs/api/resources/listen/resources/v1/client/Socket.js';
import OpenAI from 'openai';

const PORT = Number(process.env.PORT ?? 3001);
const ELEVENLABS_BASE = 'https://api.elevenlabs.io';

/** Build the NVIDIA NIM REST endpoint URL for Audio2Face-3D given a function/model ID. */
function nvidiaA2FUrl(functionId: string): string {
  return `https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/${functionId}`;
}

// ─── Express HTTP (health-check only — main API lives in root server) ────────
const app = express();
app.use(cors());
app.use(express.json());
app.get('/health', (_req, res) => res.json({ ok: true, service: 'npc-voice' }));

const httpServer = createServer(app);
const wss = new WebSocketServer({ server: httpServer });

// ─── Helpers ─────────────────────────────────────────────────────────────────

function send(ws: WebSocket, data: object) {
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(data));
}

function sendBinary(ws: WebSocket, buf: Buffer | Uint8Array) {
  if (ws.readyState === WebSocket.OPEN) ws.send(buf);
}

/**
 * Stream text to ElevenLabs websocket TTS.
 * Returns async generator of raw MP3 audio chunks.
 */
async function* streamElevenLabsTTS(
  text: string,
  voiceId: string,
  apiKey: string,
): AsyncGenerator<Buffer> {
  const url = `${ELEVENLABS_BASE}/v1/text-to-speech/${voiceId}/stream`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'xi-api-key': apiKey,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text,
      model_id: 'eleven_flash_v2_5',
      output_format: 'mp3_44100_128',
    }),
  });

  if (!res.ok || !res.body) {
    const errText = await res.text().catch(() => res.statusText);
    throw new Error(`ElevenLabs TTS error ${res.status}: ${errText}`);
  }

  const reader = res.body.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) yield Buffer.from(value);
  }
}

/**
 * Call NVIDIA Audio2Face-3D REST API with a complete audio buffer.
 * Returns ARKit blendshape frames: Array of { timestamp, values }
 *
 * Note: Requires NVIDIA API key and the NIM endpoint to be accessible.
 * Falls back gracefully (returns empty array) if unavailable.
 */
async function fetchA2FBlendshapes(
  audioPcmBuffer: Buffer,
  sampleRate: number,
  nvidiaApiKey: string,
  nvidiaFunctionId: string,
): Promise<Array<{ timestamp: number; values: Record<string, number> }>> {
  if (!nvidiaApiKey || !nvidiaFunctionId) return [];

  try {
    // Build multipart form with WAV header + PCM data
    const wavBuffer = pcmToWav(audioPcmBuffer, sampleRate, 1, 16);

    const formData = new FormData();
    formData.append(
      'audio',
      new Blob([wavBuffer], { type: 'audio/wav' }),
      'speech.wav',
    );

    const res = await fetch(nvidiaA2FUrl(nvidiaFunctionId), {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${nvidiaApiKey}`,
      },
      body: formData,
    });

    if (!res.ok) {
      console.warn(`[A2F] Non-OK response ${res.status} — no blendshapes`);
      return [];
    }

    const json = (await res.json()) as {
      output?: Array<{ time_stamp: number; blend_shapes: Record<string, number> }>;
    };

    const frames = json.output ?? [];
    return frames.map((f) => ({ timestamp: f.time_stamp, values: f.blend_shapes }));
  } catch (err) {
    console.warn('[A2F] Error fetching blendshapes:', err);
    return [];
  }
}

/** Build a minimal WAV header around raw Int16 PCM. */
function pcmToWav(
  pcm: Buffer,
  sampleRate: number,
  channels: number,
  bitDepth: number,
): Buffer {
  const headerSize = 44;
  const byteRate = (sampleRate * channels * bitDepth) / 8;
  const blockAlign = (channels * bitDepth) / 8;
  const buf = Buffer.alloc(headerSize + pcm.length);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + pcm.length, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16); // PCM subchunk size
  buf.writeUInt16LE(1, 20); // AudioFormat=PCM
  buf.writeUInt16LE(channels, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(byteRate, 28);
  buf.writeUInt16LE(blockAlign, 32);
  buf.writeUInt16LE(bitDepth, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(pcm.length, 40);
  pcm.copy(buf, 44);
  return buf;
}

// ─── WebSocket session handler ────────────────────────────────────────────────

interface SessionConfig {
  personalityPrompt: string;
  openAiKey: string;
  deepgramKey: string;
  elevenLabsKey: string;
  voiceId: string;
  nvidiaApiKey: string;
  nvidiaFunctionId: string;
}

wss.on('connection', (ws: WebSocket) => {
  console.log('[WS] Client connected');
  let config: SessionConfig | null = null;
  let deepgramSocket: V1Socket | null = null;
  let deepgramReady = false;
  const pendingAudioBuffer: Buffer[] = [];
  let currentTranscript = '';
  const incomingAudioChunks: Buffer[] = [];

  // Track whether we're currently generating a response (prevent overlapping)
  let responding = false;

  async function handleTranscript(transcript: string) {
    if (!config || responding || !transcript.trim()) return;
    responding = true;
    currentTranscript = '';

    try {
      send(ws, { type: 'transcript', text: transcript });
      send(ws, { type: 'response_start' });

      const openai = new OpenAI({ apiKey: config.openAiKey });

      // Stream LLM response
      let fullResponse = '';
      console.log(`[OpenAI] Sending to LLM: "${transcript}"`);
      try {
        const stream = await openai.chat.completions.create({
          model: 'gpt-4o',
          stream: true,
          messages: [
            { role: 'system', content: config.personalityPrompt },
            { role: 'user', content: transcript },
          ],
        });

        for await (const chunk of stream) {
          const token = chunk.choices[0]?.delta?.content ?? '';
          fullResponse += token;
        }
        console.log(`[OpenAI] Response received: "${fullResponse}"`);
      } catch (err: unknown) {
        console.error('[OpenAI] Error calling LLM:', err instanceof Error ? err.stack ?? err.message : err);
        throw err;
      }

      // TTS: stream audio back to client
      console.log(`[ElevenLabs] Sending TTS for: "${fullResponse}"`);
      try {
        let totalBytes = 0;
        for await (const chunk of streamElevenLabsTTS(
          fullResponse,
          config.voiceId,
          config.elevenLabsKey,
        )) {
          totalBytes += chunk.length;
          console.log(`[ElevenLabs] Audio received, bytes: ${chunk.length} (total so far: ${totalBytes})`);
          console.log('[WS] Sending audio chunk to client');
          sendBinary(ws, chunk);
        }
        console.log(`[ElevenLabs] TTS complete, total bytes: ${totalBytes}`);
      } catch (err: unknown) {
        console.error('[ElevenLabs] Error calling TTS:', err instanceof Error ? err.stack ?? err.message : err);
        throw err;
      }

      // A2F blendshapes from accumulated user mic audio
      if (config.nvidiaApiKey && config.nvidiaFunctionId && incomingAudioChunks.length > 0) {
        const micAudio = Buffer.concat(incomingAudioChunks);
        const frames = await fetchA2FBlendshapes(micAudio, 16000, config.nvidiaApiKey, config.nvidiaFunctionId);
        for (const frame of frames) {
          send(ws, { type: 'blendshapes', timestamp: frame.timestamp, values: frame.values });
        }
        incomingAudioChunks.length = 0;
      }

      send(ws, { type: 'response_end' });
    } catch (err: unknown) {
      console.error('[Session] Pipeline error:', err instanceof Error ? err.stack ?? err.message : err);
      send(ws, { type: 'error', message: err instanceof Error ? err.message : 'Response error' });
    } finally {
      responding = false;
    }
  }

  async function startConversation() {
    if (!config || responding) return;
    responding = true;

    const openai = new OpenAI({ apiKey: config.openAiKey });
    const system = config.personalityPrompt ?? 'You are a helpful NPC.';
    const greetingPrompt = 'Start the conversation by greeting the player and asking them one short question. Speak in character.';

    try {
      let fullResponse = '';
      const stream = await openai.chat.completions.create({
        model: 'gpt-4o',
        stream: true,
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: greetingPrompt },
        ],
      });

      for await (const chunk of stream) {
        const token = chunk.choices[0]?.delta?.content ?? '';
        fullResponse += token;
      }

      console.log(`[startConversation] Greeting: "${fullResponse}"`);
      send(ws, { type: 'response_start' });

      for await (const chunk of streamElevenLabsTTS(
        fullResponse,
        config.voiceId,
        config.elevenLabsKey,
      )) {
        sendBinary(ws, chunk);
      }

      send(ws, { type: 'response_end' });
    } catch (err: unknown) {
      console.error('[startConversation] Error:', err instanceof Error ? err.message : err);
      send(ws, { type: 'error', message: err instanceof Error ? err.message : 'Greeting error' });
    } finally {
      responding = false;
    }
  }

  async function initDeepgram() {
    if (!config) return;
    const dg = new DeepgramClient({ apiKey: config.deepgramKey });
    const socket = await dg.listen.v1.connect({
      model: 'nova-3',
      language: 'en',
      smart_format: true,
      punctuate: true,
      interim_results: true,
      endpointing: 400,
      encoding: 'linear16',
      sample_rate: 16000,
      Authorization: `Token ${config.deepgramKey}`,
    });

    socket.connect();
    deepgramSocket = socket;

    socket.on('open', () => {
      console.log('[Deepgram] Connection open');
      deepgramReady = true;
      for (const chunk of pendingAudioBuffer) {
        socket.sendMedia(chunk);
      }
      pendingAudioBuffer.length = 0;
    });
    socket.on('close', () => {
      console.log('[Deepgram] Connection closed');
      deepgramReady = false;
    });
    socket.on('error', (err: Error) => {
      console.error('[Deepgram] Error:', err);
      send(ws, { type: 'error', message: `ASR error: ${err.message}` });
    });
    socket.on('message', (msg) => {
      if (msg.type !== 'Results') return;
      const alt = msg.channel?.alternatives?.[0];
      if (!alt) {
        console.log('[Deepgram] Message with no alternatives, skipping');
        return;
      }
      const t = alt.transcript ?? '';
      const confidence = alt.confidence ?? 0;
      console.log(`[Deepgram] Result — is_final:${msg.is_final} speech_final:${msg.speech_final} confidence:${confidence.toFixed(3)} transcript:"${t}"`);

      if (confidence < 0.5 && t.trim()) {
        console.log(`[Deepgram] Low confidence (${confidence.toFixed(3)}) — skipping transcript: "${t}"`);
        return;
      }

      if (msg.is_final) {
        currentTranscript += (currentTranscript ? ' ' : '') + t;
      }
      if (msg.speech_final && currentTranscript.trim() && !responding) {
        const toSend = currentTranscript.trim();
        console.log(`[Deepgram] Transcript received: "${toSend}"`);
        currentTranscript = '';
        handleTranscript(toSend);
      }
    });
  }

  ws.on('message', (data: Buffer, isBinary: boolean) => {
    if (!config) {
      // First message must be the init JSON
      try {
        const msg = JSON.parse(data.toString()) as { type?: string } & Partial<SessionConfig>;
        if (msg.type !== 'init') {
          send(ws, { type: 'error', message: 'First message must be type:init' });
          return;
        }
        if (!msg.openAiKey || !msg.deepgramKey || !msg.elevenLabsKey || !msg.voiceId) {
          send(ws, { type: 'error', message: 'Missing required keys in init message' });
          return;
        }
        config = {
          personalityPrompt: msg.personalityPrompt ?? 'You are a helpful NPC.',
          openAiKey: msg.openAiKey,
          deepgramKey: msg.deepgramKey,
          elevenLabsKey: msg.elevenLabsKey,
          voiceId: msg.voiceId,
          nvidiaApiKey: msg.nvidiaApiKey ?? '',
          nvidiaFunctionId: msg.nvidiaFunctionId ?? '',
        };
        initDeepgram()
          .then(() => {
            send(ws, { type: 'ready' });
            console.log('[WS] Session initialized');
            startConversation();
          })
          .catch((err: Error) => {
            send(ws, { type: 'error', message: `ASR init failed: ${err.message}` });
          });
      } catch {
        send(ws, { type: 'error', message: 'Invalid init message' });
      }
      return;
    }

    // Subsequent binary messages are raw Int16 PCM audio
    if (isBinary) {
      incomingAudioChunks.push(Buffer.from(data));
      if (deepgramSocket && deepgramReady) {
        deepgramSocket.sendMedia(data);
      } else if (deepgramSocket) {
        pendingAudioBuffer.push(Buffer.from(data));
      }
    }
  });

  ws.on('close', () => {
    console.log('[WS] Client disconnected');
    deepgramSocket?.close();
    deepgramSocket = null;
  });

  ws.on('error', (err) => console.error('[WS] Socket error:', err));
});

// ─── Start ────────────────────────────────────────────────────────────────────

httpServer.listen(PORT, () => {
  console.log(`[NPC Voice Server] Listening on ws://localhost:${PORT}`);
  console.log(`[NPC Voice Server] Health: http://localhost:${PORT}/health`);
});
