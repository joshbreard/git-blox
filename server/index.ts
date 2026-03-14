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
 *     JSON:   { type:'npc_response', audio:string (base64 MP3), blendshapes:Array<{timestamp,values}>, fps:30 }
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

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PORT = Number(process.env.PORT ?? 3001);

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

/**
 * Stream text to ElevenLabs TTS via WebSocket streaming-input endpoint.
 * Returns raw PCM_16000 chunks as an async generator.
 */
async function* streamElevenLabsTTS(
  text: string,
  voiceId: string,
  apiKey: string,
): AsyncGenerator<Buffer> {
  if (!voiceId) throw new Error('ElevenLabs voiceId is required');

  const url = `wss://api.elevenlabs.io/v1/text-to-speech/${voiceId}/stream-input?model_id=eleven_flash_v2_5&output_format=pcm_16000`;

  const ws = new WebSocket(url, {
    headers: { 'xi-api-key': apiKey },
  });

  const chunks: Buffer[] = [];
  let resolve: (() => void) | null = null;
  let done = false;
  let error: Error | null = null;

  ws.on('open', () => {
    // Send BOS with config
    ws.send(JSON.stringify({
      text: ' ',
      voice_settings: { stability: 0.5, similarity_boost: 0.75 },
      generation_config: { chunk_length_schedule: [120, 160, 250, 290] },
    }));
    // Send the actual text
    ws.send(JSON.stringify({ text }));
    // Send EOS
    ws.send(JSON.stringify({ text: '' }));
  });

  ws.on('message', (data: Buffer) => {
    try {
      const msg = JSON.parse(data.toString()) as { audio?: string; isFinal?: boolean; error?: string };
      if (msg.error) {
        error = new Error(`ElevenLabs WS error: ${msg.error}`);
        done = true;
      } else if (msg.audio) {
        chunks.push(Buffer.from(msg.audio, 'base64'));
      }
      if (msg.isFinal) {
        done = true;
      }
    } catch {
      // ignore parse errors
    }
    resolve?.();
    resolve = null;
  });

  ws.on('error', (err) => {
    error = err;
    done = true;
    resolve?.();
    resolve = null;
  });

  ws.on('close', () => {
    done = true;
    resolve?.();
    resolve = null;
  });

  // Yield chunks as they arrive
  while (true) {
    if (chunks.length > 0) {
      yield chunks.shift()!;
      continue;
    }
    if (done) break;
    if (error) throw error;
    await new Promise<void>((r) => { resolve = r; });
  }
  // Flush any remaining chunks
  while (chunks.length > 0) yield chunks.shift()!;
  if (error) throw error;
}

/**
 * Wrap raw PCM data in a WAV container for browser playback.
 * Trims any odd trailing byte (alignment guard) before writing the header.
 */
function pcmToWav(pcm: Buffer, sampleRate = 16000, channels = 1, bitDepth = 16): Buffer {
  const data = pcm.length % 2 === 0 ? pcm : pcm.subarray(0, pcm.length - 1);
  const byteRate = (sampleRate * channels * bitDepth) / 8;
  const blockAlign = (channels * bitDepth) / 8;
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + data.length, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);           // PCM format
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitDepth, 34);
  header.write('data', 36);
  header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]);
}

/**
 * Open a bidirectional A2F gRPC stream. Returns a live `call` object for writing
 * PCM chunks and a `framesPromise` that resolves with all collected frames when
 * the stream ends. Audio should be written as chunks arrive and `call.end()` called
 * when done — A2F then processes in parallel with the rest of the pipeline.
 */
function openA2FCall(
  nvidiaApiKey: string,
  nvidiaFunctionId: string,
): { call: { write: (msg: unknown) => void; end: () => void }; framesPromise: Promise<Array<{ timestamp: number; values: Record<string, number> }>> } {
  const noop = { write: () => {}, end: () => {} };
  if (!nvidiaApiKey || !nvidiaFunctionId) {
    return { call: noop, framesPromise: Promise.resolve([]) };
  }

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
    const svc = (grpcObj as any)?.nvidia_ace?.services?.a2f_controller?.v1;
    const A2FServiceClient = svc?.A2FControllerService;
    if (!A2FServiceClient) throw new Error('[A2F] A2FControllerService not found in loaded proto');

    const meta = new grpc.Metadata();
    meta.set('authorization', `Bearer ${nvidiaApiKey}`);
    meta.set('nvcf-function-id', nvidiaFunctionId);
    meta.set('function-id', nvidiaFunctionId);

    const client = new A2FServiceClient(
      'grpc.nvcf.nvidia.com:443',
      grpc.credentials.createSsl(),
      {
        'grpc.max_receive_message_length': 64 * 1024 * 1024,
        'grpc.max_send_message_length': 64 * 1024 * 1024,
      },
    );

    const deadline = new Date(Date.now() + 30_000);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const call = (client as any).processAudioStream(meta, { deadline });

    const framesPromise = new Promise<Array<{ timestamp: number; values: Record<string, number> }>>((resolve) => {
      const blendshapeNames: string[] = [];
      const frames: Array<{ timestamp: number; values: Record<string, number> }> = [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      call.on('data', (msg: any) => {
        if (msg.animation_data_stream_header) {
          const names: string[] =
            msg.animation_data_stream_header?.skel_animation_header?.blend_shapes ?? [];
          blendshapeNames.push(...names);
          console.log(`[A2F] Header received — ${blendshapeNames.length} blendshape names`);
        }

        if (msg.animation_data?.skel_animation) {
          for (const bsw of msg.animation_data.skel_animation.blend_shape_weights ?? []) {
            const frameValues: Record<string, number> = {};
            (bsw.values as number[]).forEach((weight: number, i: number) => {
              if (blendshapeNames[i] !== undefined) frameValues[blendshapeNames[i]] = weight;
            });
            frames.push({ timestamp: bsw.time_code as number, values: frameValues });
          }
        }

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
    });

    return { call, framesPromise };
  } catch (err) {
    console.warn('[A2F] Error opening A2F call:', err);
    return { call: noop, framesPromise: Promise.resolve([]) };
  }
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

      console.log(`[ElevenLabs] Sending TTS for: "${fullResponse}"`);

      // Open A2F gRPC call before streaming starts so it's ready to receive
      const { call, framesPromise } = openA2FCall(config.nvidiaApiKey, config.nvidiaFunctionId);
      call.write({
        audio_stream_header: {
          audio_header: { audio_format: 0, channel_count: 1, samples_per_second: 16000, bits_per_sample: 16 },
        },
      });

      // Single WebSocket TTS call — yields raw PCM_16000
      // Stream PCM to A2F as chunks arrive, accumulate for WAV conversion
      const pcmChunks: Buffer[] = [];
      let idx = 0;
      try {
        for await (const chunk of streamElevenLabsTTS(fullResponse, config!.voiceId, config!.elevenLabsKey)) {
          pcmChunks.push(chunk);
          call.write({ audio_with_emotion: { audio_buffer: chunk } });
          if (idx % 10 === 0) console.log(`[A2F] Streaming PCM chunk: ${chunk.length} bytes`);
          idx++;
        }
      } catch (err: unknown) {
        console.error('[ElevenLabs] PCM stream error:', err instanceof Error ? err.message : err);
      }
      call.end();
      console.log(`[ElevenLabs] PCM TTS complete, chunks: ${idx}`);

      const wavBuffer = pcmToWav(Buffer.concat(pcmChunks));
      console.log(`[ElevenLabs] WAV TTS complete, total bytes: ${wavBuffer.length}`);

      // Send WAV audio to browser immediately — do NOT wait for A2F
      const audioBase64 = wavBuffer.toString('base64');
      incomingAudioChunks.length = 0;
      send(ws, { type: 'npc_response', audio: audioBase64, blendshapes: [], fps: 30 });
      send(ws, { type: 'response_end' });

      // A2F continues processing in background — send blendshapes when done
      const audioSentAt = Date.now();
      framesPromise.then((frames) => {
        if (frames.length > 0) {
          send(ws, { type: 'npc_blendshapes', frames, fps: 30, audioOffsetMs: Date.now() - audioSentAt });
        }
      }).catch(() => { /* logged inside openA2FCall */ });
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

      // Open A2F gRPC call before streaming starts so it's ready to receive
      const { call: scCall, framesPromise: scFramesPromise } = openA2FCall(config.nvidiaApiKey, config.nvidiaFunctionId);
      scCall.write({
        audio_stream_header: {
          audio_header: { audio_format: 0, channel_count: 1, samples_per_second: 16000, bits_per_sample: 16 },
        },
      });

      // Single WebSocket TTS call — yields raw PCM_16000
      // Stream PCM to A2F as chunks arrive, accumulate for WAV conversion
      const scPcmChunks: Buffer[] = [];
      let scIdx = 0;
      try {
        for await (const chunk of streamElevenLabsTTS(fullResponse, config!.voiceId, config!.elevenLabsKey)) {
          scPcmChunks.push(chunk);
          scCall.write({ audio_with_emotion: { audio_buffer: chunk } });
          if (scIdx % 10 === 0) console.log(`[A2F] Streaming PCM chunk: ${chunk.length} bytes`);
          scIdx++;
        }
      } catch (err: unknown) {
        console.error('[ElevenLabs] PCM stream error:', err instanceof Error ? err.message : err);
      }
      scCall.end();
      console.log(`[ElevenLabs] PCM TTS complete, chunks: ${scIdx}`);

      const scWavBuffer = pcmToWav(Buffer.concat(scPcmChunks));
      console.log(`[ElevenLabs] WAV TTS complete, total bytes: ${scWavBuffer.length}`);

      // Send WAV audio to browser immediately — do NOT wait for A2F
      const scAudioBase64 = scWavBuffer.toString('base64');
      send(ws, { type: 'npc_response', audio: scAudioBase64, blendshapes: [], fps: 30 });
      send(ws, { type: 'response_end' });

      // A2F continues processing in background — send blendshapes when done
      const scAudioSentAt = Date.now();
      scFramesPromise.then((frames) => {
        if (frames.length > 0) {
          send(ws, { type: 'npc_blendshapes', frames, fps: 30, audioOffsetMs: Date.now() - scAudioSentAt });
        }
      }).catch(() => { /* logged inside openA2FCall */ });
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
      endpointing: 300,
      utterance_end_ms: 1000,
      vad_events: true,
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

      const t = (alt.transcript ?? '').trim();
      const confidence = alt.confidence ?? 0;

      // Accumulate is_final segments (not speech_final yet)
      if (msg.is_final && t) {
        currentTranscript += (currentTranscript ? ' ' : '') + t;
        console.log(`[Deepgram] Segment final — confidence:${confidence.toFixed(3)} transcript:"${t}" accumulated:"${currentTranscript}"`);
      }

      // Only fire when speech_final=true AND we have accumulated text with good confidence
      if (msg.speech_final) {
        const toSend = currentTranscript.trim();
        currentTranscript = '';
        if (!toSend) return;
        if (confidence < 0.5) {
          console.log(`[Deepgram] speech_final but low confidence (${confidence.toFixed(3)}) — skipping`);
          return;
        }
        console.log(`[Deepgram] Utterance complete — sending: "${toSend}"`);
        processingUtterance = true;
        handleTranscript(toSend).finally(() => { processingUtterance = false; });
      }
    });
    socket.on('UtteranceEnd', () => {
      if (isSpeaking || processingUtterance) return;
      const toSend = currentTranscript.trim();
      if (!toSend) return;
      currentTranscript = '';
      console.log(`[Deepgram] UtteranceEnd fallback — sending: "${toSend}"`);
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
