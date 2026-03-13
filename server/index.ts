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
 *     JSON:   { type:'npc_response', audio:string (base64 WAV), blendshapes:Array<{timestamp,values}>, fps:30 }
 *     JSON:   { type:'response_end' }
 *     JSON:   { type:'error', message }
 */

import 'dotenv/config';
import { WebSocketServer, WebSocket } from 'ws';
import { createServer } from 'http';
import express from 'express';
import cors from 'cors';
import { DeepgramClient } from '@deepgram/sdk';
import type { ListenLiveClient } from '@deepgram/sdk';
import OpenAI from 'openai';
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { execFile } from 'child_process';
import * as fs from 'fs';
import * as crypto from 'crypto';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PORT = Number(process.env.PORT ?? 3001);
const ELEVENLABS_BASE = 'https://api.elevenlabs.io';

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

/** Buffer all TTS MP3 chunks and return the raw MP3 buffer and its base64 encoding. */
async function bufferTTSToMp3(
  source: AsyncIterable<Buffer>,
): Promise<{ mp3Buffer: Buffer; base64: string }> {
  const chunks: Buffer[] = [];
  for await (const chunk of source) {
    chunks.push(chunk);
  }
  const mp3Buffer = Buffer.concat(chunks);
  const base64 = mp3Buffer.toString('base64');
  return { mp3Buffer, base64 };
}

/** Convert an MP3 buffer to a 16kHz mono WAV buffer (base64) via ffmpeg. */
async function mp3ToWavBase64(mp3Buffer: Buffer): Promise<string> {
  const tmpId = crypto.randomBytes(8).toString('hex');
  const inputPath = path.join('/tmp', `a2f_in_${tmpId}.mp3`);
  const outputPath = path.join('/tmp', `a2f_out_${tmpId}.wav`);
  try {
    await fs.promises.writeFile(inputPath, mp3Buffer);
    await new Promise<void>((resolve, reject) => {
      execFile('ffmpeg', ['-y', '-i', inputPath, '-ar', '16000', '-ac', '1', '-f', 'wav', outputPath], (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
    const wavBuffer = await fs.promises.readFile(outputPath);
    return wavBuffer.toString('base64');
  } finally {
    await fs.promises.unlink(inputPath).catch(() => {});
    await fs.promises.unlink(outputPath).catch(() => {});
  }
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
  if (!voiceId) throw new Error('ElevenLabs voiceId is required but was not provided in session config');
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
 * Call NVIDIA Audio2Face-3D via gRPC on NVCF.
 *
 * Loads proto/a2f_nvcf.proto (NVCF cloud variant where PushAudioStream returns a
 * stream of AnimationDataStream rather than a single Status). Connects to
 * grpc.nvcf.nvidia.com:443 with TLS, sets authorization + function-id metadata,
 * then streams the WAV PCM in 4096-byte AudioWithEmotion chunks. Collects
 * blendshape names from the AnimationDataStreamHeader and float values from each
 * AnimationData frame, returning them as { timestamp, values } objects.
 */
async function fetchA2FBlendshapesOnce(
  audioBase64: string,
  nvidiaApiKey: string,
  nvidiaFunctionId: string,
): Promise<Array<{ timestamp: number; values: Record<string, number> }>> {
  if (!nvidiaApiKey || !nvidiaFunctionId) return [];

  console.log(`[A2F] WAV buffer byte length: ${Buffer.byteLength(audioBase64, 'base64')}`);
  console.log(`[A2F] base64 preview (first 100 chars): ${audioBase64.slice(0, 100)}`);
  console.log(`[A2F] functionId: ${nvidiaFunctionId}`);

  try {
    const protoDir = path.resolve(__dirname, 'proto');
    const packageDef = protoLoader.loadSync(
      path.join(protoDir, 'nvidia_ace.services.a2f_controller.v1.proto'),
      {
        keepCase: true,
        longs: Number,
        enums: Number,
        defaults: true,
        oneofs: true,
        includeDirs: [
          protoDir,
          path.resolve(__dirname, '..', 'proto'),
          path.resolve(__dirname, 'server', 'proto'),
        ],
      },
    );

    const grpcObj = grpc.loadPackageDefinition(packageDef) as Record<string, unknown>;
    console.log('[A2F] grpcObj top-level keys:', JSON.stringify(Object.keys(grpcObj)));

    const svc = (grpcObj as any)?.nvidia_ace?.services?.a2f_controller?.v1;
    console.log('[A2F] Available service keys:', JSON.stringify(Object.keys(svc ?? {})));
    const A2FServiceClient = svc?.A2FControllerService;
    if (!A2FServiceClient) throw new Error('[A2F] A2FControllerService not found in loaded proto');

    const meta = new grpc.Metadata();
    meta.set('authorization', `Bearer ${nvidiaApiKey}`);
    // NVCF gRPC routing header — 'nvcf-function-id' is the current standard;
    // 'function-id' is kept as a fallback for older NVCF deployments.
    meta.set('nvcf-function-id', nvidiaFunctionId);
    meta.set('function-id', nvidiaFunctionId);

    const client = new A2FServiceClient(
      'grpc.nvcf.nvidia.com:443',
      grpc.credentials.createSsl(),
      {
        'grpc.max_receive_message_length': 64 * 1024 * 1024, // 64 MB
        'grpc.max_send_message_length': 64 * 1024 * 1024,
      },
    );

    console.log(`[A2F] Calling /nvidia_ace.services.a2f_controller.v1.A2FControllerService/ProcessAudioStream on grpc.nvcf.nvidia.com:443`);

    return new Promise((resolve) => {
      const blendshapeNames: string[] = [];
      const frames: Array<{ timestamp: number; values: Record<string, number> }> = [];

      // Bidirectional stream: we send AudioStream messages, server sends AnimationDataStream.
      // Set a deadline to accommodate NVCF cold-start worker provisioning.
      const deadline = new Date(Date.now() + 30_000);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const call = (client as any).processAudioStream(meta, { deadline });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      call.on('data', (msg: any) => {
        // First response message: header containing blendshape names for the session.
        if (msg.animation_data_stream_header) {
          const names: string[] =
            msg.animation_data_stream_header?.skel_animation_header?.blend_shapes ?? [];
          blendshapeNames.push(...names);
          console.log(`[A2F] Header received — ${blendshapeNames.length} blendshape names`);
        }

        // Subsequent messages: animation data with one or more blend_shape_weights frames.
        if (msg.animation_data?.skel_animation) {
          for (const bsw of msg.animation_data.skel_animation.blend_shape_weights ?? []) {
            // bsw: FloatArrayWithTimeCode { time_code: number, values: number[] }
            const frameValues: Record<string, number> = {};
            (bsw.values as number[]).forEach((weight, i) => {
              if (blendshapeNames[i] !== undefined) frameValues[blendshapeNames[i]] = weight;
            });
            frames.push({ timestamp: bsw.time_code as number, values: frameValues });
          }
        }

        // Status message (typically the last one in the stream).
        if (msg.status) {
          console.log(`[A2F] Status — code: ${msg.status.code} message: ${msg.status.message}`);
        }
      });

      call.on('end', () => {
        console.log(`[A2F] Stream ended — ${frames.length} blendshape frames collected`);
        resolve(frames);
      });

      call.on('error', (err: Error & { code?: number; details?: string }) => {
        console.warn(`[A2F] gRPC stream error: code=${err.code} message="${err.message}" details="${err.details ?? ''}"`);
        resolve([]);
      });

      // ── Send audio ───────────────────────────────────────────────────────────

      call.write({
        audio_stream_header: {
          audio_header: {
            audio_format: 0,
            channel_count: 1,
            samples_per_second: 16000,
            bits_per_sample: 16,
          },
        },
      });

      const wavBuf = Buffer.from(audioBase64, 'base64');
      call.write({
        audio_with_emotion: {
          audio_buffer: wavBuf,
        },
      });

      call.end();
    });
  } catch (err) {
    console.warn('[A2F] Error fetching blendshapes:', err);
    return [];
  }
}

/**
 * Retrying wrapper around fetchA2FBlendshapesOnce.
 * NVCF workers cold-start on the first request and can return DEADLINE_EXCEEDED
 * ("failed to establish link to worker") until the worker is ready. We retry
 * up to MAX_RETRIES times with a short delay to handle this gracefully.
 */
async function fetchA2FBlendshapes(
  audioBase64: string,
  nvidiaApiKey: string,
  nvidiaFunctionId: string,
): Promise<Array<{ timestamp: number; values: Record<string, number> }>> {
  const MAX_RETRIES = 2;
  const RETRY_DELAY_MS = 3000;
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    const result = await fetchA2FBlendshapesOnce(audioBase64, nvidiaApiKey, nvidiaFunctionId);
    if (result.length > 0) return result;
    if (attempt < MAX_RETRIES) {
      console.log(`[A2F] No frames on attempt ${attempt}, retrying in ${RETRY_DELAY_MS}ms…`);
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY_MS));
    }
  }
  console.warn('[A2F] All retry attempts exhausted, returning empty frames.');
  return [];
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
  let deepgramSocket: ListenLiveClient | null = null;
  let deepgramReady = false;
  const pendingAudioBuffer: Buffer[] = [];
  let currentTranscript = '';
  const incomingAudioChunks: Buffer[] = [];

  // Track whether we're currently speaking (LLM + TTS pipeline active)
  let isSpeaking = false;
  // Set synchronously before handleTranscript is called; prevents a second
  // speech_final event from slipping through before isSpeaking is set inside handleTranscript.
  let processingUtterance = false;

  async function handleTranscript(transcript: string) {
    if (isSpeaking) return;
    if (!config || !transcript.trim()) return;
    isSpeaking = true;
    currentTranscript = '';
    if (deepgramSocket && deepgramReady) {
      try { deepgramSocket.requestClose(); } catch { /* ignore */ }
    }

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

      // TTS: buffer all MP3 audio
      console.log(`[ElevenLabs] Sending TTS for: "${fullResponse}"`);
      let mp3Buffer: Buffer;
      let audioBase64: string;
      try {
        ({ mp3Buffer, base64: audioBase64 } = await bufferTTSToMp3(
          streamElevenLabsTTS(fullResponse, config.voiceId, config.elevenLabsKey),
        ));
        console.log(`[ElevenLabs] TTS complete, total bytes: ${mp3Buffer.length}`);
      } catch (err: unknown) {
        console.error('[ElevenLabs] Error calling TTS:', err instanceof Error ? err.stack ?? err.message : err);
        throw err;
      }

      incomingAudioChunks.length = 0;

      // Send MP3 audio immediately — do NOT block on A2F
      send(ws, { type: 'npc_response', audio: audioBase64, blendshapes: [], fps: 30 });
      send(ws, { type: 'response_end' });

      // Run A2F in background; convert MP3→WAV then fetch blendshapes
      if (config.nvidiaApiKey && config.nvidiaFunctionId) {
        const capturedConfig = config;
        const capturedMp3 = mp3Buffer;
        const audioSentAt = Date.now();
        mp3ToWavBase64(capturedMp3)
          .then((wavBase64) => fetchA2FBlendshapes(wavBase64, capturedConfig.nvidiaApiKey, capturedConfig.nvidiaFunctionId))
          .then((frames) => {
            if (frames.length > 0) {
              send(ws, { type: 'npc_blendshapes', frames, fps: 30, audioOffsetMs: Date.now() - audioSentAt });
            }
          })
          .catch(() => { /* logged inside fetchA2FBlendshapes */ });
      }
    } catch (err: unknown) {
      console.error('[Session] Pipeline error:', err instanceof Error ? err.stack ?? err.message : err);
      send(ws, { type: 'error', message: err instanceof Error ? err.message : 'Response error' });
    } finally {
      try { reconnectDeepgram(); } catch (err) { console.error('[Deepgram] Reconnect failed:', err); }
      isSpeaking = false;
    }
  }

  async function startConversation() {
    if (!config || isSpeaking) return;
    isSpeaking = true;
    if (deepgramSocket && deepgramReady) {
      try { deepgramSocket.requestClose(); } catch { /* ignore */ }
    }

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

      const { mp3Buffer, base64: audioBase64 } = await bufferTTSToMp3(
        streamElevenLabsTTS(fullResponse, config.voiceId, config.elevenLabsKey),
      );

      // Send MP3 audio immediately — do NOT block on A2F
      send(ws, { type: 'npc_response', audio: audioBase64, blendshapes: [], fps: 30 });
      send(ws, { type: 'response_end' });

      // Run A2F in background; convert MP3→WAV then fetch blendshapes
      if (config.nvidiaApiKey && config.nvidiaFunctionId) {
        const capturedConfig = config;
        const capturedMp3 = mp3Buffer;
        const audioSentAt = Date.now();
        mp3ToWavBase64(capturedMp3)
          .then((wavBase64) => fetchA2FBlendshapes(wavBase64, capturedConfig.nvidiaApiKey, capturedConfig.nvidiaFunctionId))
          .then((frames) => {
            if (frames.length > 0) {
              send(ws, { type: 'npc_blendshapes', frames, fps: 30, audioOffsetMs: Date.now() - audioSentAt });
            }
          })
          .catch(() => { /* logged inside fetchA2FBlendshapes */ });
      }
    } catch (err: unknown) {
      console.error('[startConversation] Error:', err instanceof Error ? err.message : err);
      send(ws, { type: 'error', message: err instanceof Error ? err.message : 'Greeting error' });
    } finally {
      try { reconnectDeepgram(); } catch (err) { console.error('[Deepgram] Reconnect failed:', err); }
      isSpeaking = false;
    }
  }

  function reconnectDeepgram() {
    if (deepgramSocket) {
      try { deepgramSocket.requestClose(); } catch { /* ignore */ }
      deepgramSocket = null;
      deepgramReady = false;
    }
    initDeepgram();
  }

  function initDeepgram() {
    if (!config) return;
    const dg = new DeepgramClient({ key: config.deepgramKey });
    const socket = dg.listen.live({
      model: 'nova-3',
      language: 'en',
      smart_format: true,
      punctuate: true,
      interim_results: true,
      endpointing: 400,
      encoding: 'linear16',
      sample_rate: 16000,
    });

    deepgramSocket = socket;

    socket.on('open', () => {
      console.log('[Deepgram] Connection open');
      deepgramReady = true;
      for (const chunk of pendingAudioBuffer) {
        socket.send(chunk);
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
    socket.on('Results', (msg) => {
      if (isSpeaking || processingUtterance) return;

      const alt = msg.channel?.alternatives?.[0];
      if (!alt) return;

      const t = alt.transcript ?? '';
      const confidence = alt.confidence ?? 0;
      console.log(`[Deepgram] Result — is_final:${msg.is_final} speech_final:${msg.speech_final} confidence:${confidence.toFixed(3)} transcript:"${t}"`);

      // Accumulate intermediate finals; only trigger on a fully-closed utterance.
      if (!msg.is_final || !msg.speech_final) {
        if (msg.is_final) {
          currentTranscript += (currentTranscript ? ' ' : '') + t;
        }
        return;
      }

      // Guard 1: both is_final and speech_final must be true (already enforced above).
      // Guard 2: skip low-confidence results.
      if (confidence < 0.5) {
        console.log(`[Deepgram] Low confidence (${confidence.toFixed(3)}) — skipping`);
        currentTranscript = '';
        return;
      }

      // Guard 3: skip empty transcript.
      const toSend = (currentTranscript + (currentTranscript ? ' ' : '') + t).trim();
      currentTranscript = '';
      if (!toSend) return;

      // Guard 4: set processingUtterance synchronously before the async call so any
      // duplicate speech_final events arriving before isSpeaking is set are dropped.
      console.log(`[Deepgram] Transcript received: "${toSend}"`);
      processingUtterance = true;
      handleTranscript(toSend).finally(() => { processingUtterance = false; });
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
        try {
          initDeepgram();
          send(ws, { type: 'ready' });
          console.log('[WS] Session initialized');
          startConversation();
        } catch (err: unknown) {
          send(ws, { type: 'error', message: `ASR init failed: ${err instanceof Error ? err.message : String(err)}` });
        }
      } catch {
        send(ws, { type: 'error', message: 'Invalid init message' });
      }
      return;
    }

    // Subsequent binary messages are raw Int16 PCM audio
    if (isBinary) {
      incomingAudioChunks.push(Buffer.from(data));
      if (deepgramSocket && deepgramReady) {
        deepgramSocket.send(data);
      } else if (deepgramSocket) {
        pendingAudioBuffer.push(Buffer.from(data));
      }
    }
  });

  ws.on('close', () => {
    console.log('[WS] Client disconnected');
    deepgramSocket?.disconnect();
    deepgramSocket = null;
  });

  ws.on('error', (err) => console.error('[WS] Socket error:', err));
});

// ─── Start ────────────────────────────────────────────────────────────────────

httpServer.listen(PORT, () => {
  console.log(`[NPC Voice Server] Listening on ws://localhost:${PORT}`);
  console.log(`[NPC Voice Server] Health: http://localhost:${PORT}/health`);
});
