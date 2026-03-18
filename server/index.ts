/**
 * NPC Voice Pipeline Server
 *
 * WebSocket endpoint: ws://localhost:3001
 * - Accepts mic PCM audio from browser
 * - Streams audio → Deepgram ASR → OpenAI LLM → ElevenLabs TTS
 * - Simultaneously processes TTS audio through NVIDIA Audio2Face-3D for blendshapes
 * - Returns audio chunks + per-sentence blendshape frames to browser
 *
 * Message protocol:
 *   Client → Server:
 *     First message:  JSON { type:'init', personalityPrompt, openAiKey, deepgramKey,
 *                                        elevenLabsKey, voiceId, nvidiaApiKey?, llmModel? }
 *     Subsequent:     Binary (Int16 PCM, 16kHz, mono)
 *
 *   Server → Client:
 *     JSON:   { type:'transcript', text }
 *     JSON:   { type:'response_start' }
 *     JSON:   { type:'npc_response', audio:string (base64 WAV), sentenceIndex, blendshapes:[], fps:30 }
 *     JSON:   { type:'npc_filler_motion', sentenceIndex, durationMs }
 *     JSON:   { type:'npc_sentence_blendshapes', sentenceIndex, frames, fps:30 }
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

function logTime(label: string, startMs: number) {
  console.log(`[PERF] ${label}: ${Date.now() - startMs}ms`);
}

function extractCompleteSentences(buffer: string, firstSentenceSent: boolean, maxFirstSentenceChars = 60): { sentences: string[]; remainder: string } {
  const regex = /[^.!?]*[.!?]+/g;
  const sentences: string[] = [];
  let lastIndex = 0;
  let match;
  while ((match = regex.exec(buffer)) !== null) {
    const s = match[0].trim();
    if (s) sentences.push(s);
    lastIndex = regex.lastIndex;
  }
  let remainder = buffer.slice(lastIndex).trim();

  // If no sentence-ending punctuation found and first sentence hasn't been sent,
  // treat comma with at least 4 words before it as a flush point to reduce TTFA
  if (sentences.length === 0 && !firstSentenceSent && remainder) {
    const commaIdx = remainder.indexOf(',');
    if (commaIdx >= 0) {
      const beforeComma = remainder.slice(0, commaIdx).trim();
      const wordCount = beforeComma.split(/\s+/).filter(Boolean).length;
      if (wordCount >= 4) {
        const afterComma = remainder.slice(commaIdx + 1).trim();
        sentences.push(remainder.slice(0, commaIdx + 1).trim());
        remainder = afterComma;
      }
    }
  }

  // Hard-truncate the first sentence if it exceeds maxFirstSentenceChars
  if (!firstSentenceSent && sentences.length > 0 && sentences[0].length > maxFirstSentenceChars) {
    const first = sentences[0];
    // Find the last word boundary at or before the cap
    const truncateAt = first.lastIndexOf(' ', maxFirstSentenceChars);
    if (truncateAt > 0) {
      const truncated = first.slice(0, truncateAt).replace(/[.,!?]+$/, '') + '.';
      const leftover = first.slice(truncateAt + 1);
      sentences[0] = truncated;
      // Prepend the leftover to the remainder so it becomes the next sentence
      remainder = leftover + (remainder ? ' ' + remainder : '');
    }
  }

  return { sentences, remainder };
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
      generation_config: { chunk_length_schedule: [50, 100, 150, 200] },
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
  llmModel: string;
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
  let lastAudioReceivedMs = 0;
  let firstSegmentMs = 0;
  let lastSegmentConfidence = 0;

  async function handleTranscript(transcript: string) {
    if (isSpeaking) return;
    if (!config || !transcript.trim()) return;
    isSpeaking = true;
    currentTranscript = '';
    const t0 = Date.now();
    console.log(`[PERF] ── Pipeline start ──`);
    if (deepgramSocket && deepgramReady) {
      try { deepgramSocket.requestClose(); } catch { /* ignore */ }
      logTime('Deepgram close', t0);
    }

    try {
      send(ws, { type: 'transcript', text: transcript });

      const openai = new OpenAI({ apiKey: config.openAiKey });

      send(ws, { type: 'response_start' });

      let sentenceBuffer = '';
      let sentenceIndex = 0;
      let firstSentenceSent = false;
      let fullResponse = '';

      console.log(`[OpenAI] Sending to LLM: "${transcript}"`);
      let llmStream;
      try {
        llmStream = await openai.chat.completions.create({
          model: config.llmModel,
          stream: true,
          messages: [
            { role: 'system', content: `${config.personalityPrompt}\n\nIMPORTANT: Your FIRST sentence must be 4 words or fewer and end with punctuation. No exceptions. After that first short sentence, continue naturally with as much detail as you want.` },
            { role: 'user', content: transcript },
          ],
        });
      } catch (err: unknown) {
        console.error('[OpenAI] Error calling LLM:', err instanceof Error ? err.stack ?? err.message : err);
        throw err;
      }

      async function processSentence(sentence: string) {
        console.log(`[JIT] processSentence ${sentenceIndex} called at: ${Date.now() - t0}ms`);
        const idx = sentenceIndex++;
        console.log(`[JIT] Sentence ${idx}: "${sentence}"`);

        // Per-sentence A2F call
        const { call: a2fCall, framesPromise } = openA2FCall(config!.nvidiaApiKey, config!.nvidiaFunctionId);
        a2fCall.write({
          audio_stream_header: {
            audio_header: { audio_format: 0, channel_count: 1, samples_per_second: 16000, bits_per_sample: 16 },
          },
        });

        const pcmChunks: Buffer[] = [];
        for await (const chunk of streamElevenLabsTTS(sentence, config!.voiceId, config!.elevenLabsKey)) {
          pcmChunks.push(chunk);
        }

        // Send full WAV to A2F (A2F requires WAV format, not raw PCM chunks)
        const fullPcm = Buffer.concat(pcmChunks);
        const wavForA2F = pcmToWav(fullPcm);
        a2fCall.write({ audio_with_emotion: { audio_buffer: wavForA2F } });
        await new Promise(r => setTimeout(r, 50));
        a2fCall.end();

        // Build WAV for browser playback separately
        const wavBuffer = pcmToWav(fullPcm);
        console.log(`[JIT] Sentence ${idx} WAV ready: ${wavBuffer.length} bytes`);

        // Send audio to client
        send(ws, {
          type: 'npc_response',
          audio: wavBuffer.toString('base64'),
          sentenceIndex: idx,
          blendshapes: [],
          fps: 30,
        });

        if (idx === 0) logTime('First audio sent', t0);

        // Send filler motion so client can play idle animation while A2F processes
        send(ws, {
          type: 'npc_filler_motion',
          sentenceIndex: idx,
          durationMs: Math.round((wavBuffer.length / (16000 * 2)) * 1000),
        });

        // When A2F resolves for this sentence, send blendshapes immediately
        framesPromise.then((frames) => {
          if (idx === 0) logTime('First A2F blendshapes ready', t0);
          if (frames.length > 0) {
            send(ws, { type: 'npc_sentence_blendshapes', sentenceIndex: idx, frames, fps: 30 });
          }
        }).catch(() => { /* logged inside openA2FCall */ });
      }

      const sentencePromises: Promise<void>[] = [];

      for await (const chunk of llmStream) {
        const token = chunk.choices[0]?.delta?.content ?? '';
        fullResponse += token;
        sentenceBuffer += token;
        const { sentences, remainder } = extractCompleteSentences(sentenceBuffer, firstSentenceSent);
        if (sentences.length > 0) {
          firstSentenceSent = true;
          sentenceBuffer = remainder;
          for (const sentence of sentences) {
            const prev = sentencePromises[sentencePromises.length - 1] ?? Promise.resolve();
            sentencePromises.push(prev.then(() => processSentence(sentence)));
          }
        }
      }

      console.log(`[OpenAI] Response received: "${fullResponse}"`);
      logTime('LLM complete', t0);
      console.log(`[PERF] LLM response length: ${fullResponse.length} chars`);

      // Handle any remaining text without terminal punctuation
      if (sentenceBuffer.trim()) {
        const prev = sentencePromises[sentencePromises.length - 1] ?? Promise.resolve();
        sentencePromises.push(prev.then(() => processSentence(sentenceBuffer.trim())));
      }

      await Promise.all(sentencePromises);

      console.log(`[JIT] All sentences sent`);
      incomingAudioChunks.length = 0;
      send(ws, { type: 'response_end' });
    } catch (err: unknown) {
      console.error('[Session] Pipeline error:', err instanceof Error ? err.stack ?? err.message : err);
      send(ws, { type: 'error', message: err instanceof Error ? err.message : 'Response error' });
    } finally {
      isSpeaking = false;
      // Do NOT reconnect Deepgram here — client will signal when playback ends
    }
  }

  async function startConversation() {
    if (!config || isSpeaking) return;
    isSpeaking = true;
    const t0 = Date.now();
    console.log(`[PERF] ── Pipeline start ──`);
    if (deepgramSocket && deepgramReady) {
      try { deepgramSocket.requestClose(); } catch { /* ignore */ }
      logTime('Deepgram close', t0);
    }

    const openai = new OpenAI({ apiKey: config.openAiKey });
    const system = `${config.personalityPrompt ?? 'You are a helpful NPC.'}\n\nIMPORTANT: Your FIRST sentence must be 4 words or fewer and end with punctuation. No exceptions. After that first short sentence, continue naturally with as much detail as you want.`;
    const greetingPrompt = 'Say hello in 3 words. Then ask one question.';

    try {
      send(ws, { type: 'response_start' });

      let sentenceBuffer = '';
      let sentenceIndex = 0;
      let firstSentenceSent = false;
      let fullResponse = '';

      const llmStream = await openai.chat.completions.create({
        model: config.llmModel,
        stream: true,
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: greetingPrompt },
        ],
      });

      async function processSentence(sentence: string) {
        console.log(`[JIT] processSentence ${sentenceIndex} called at: ${Date.now() - t0}ms`);
        const idx = sentenceIndex++;
        console.log(`[JIT] Sentence ${idx}: "${sentence}"`);

        // Per-sentence A2F call
        const { call: a2fCall, framesPromise } = openA2FCall(config!.nvidiaApiKey, config!.nvidiaFunctionId);
        a2fCall.write({
          audio_stream_header: {
            audio_header: { audio_format: 0, channel_count: 1, samples_per_second: 16000, bits_per_sample: 16 },
          },
        });

        const pcmChunks: Buffer[] = [];
        for await (const chunk of streamElevenLabsTTS(sentence, config!.voiceId, config!.elevenLabsKey)) {
          pcmChunks.push(chunk);
        }

        // Send full WAV to A2F (A2F requires WAV format, not raw PCM chunks)
        const fullPcm = Buffer.concat(pcmChunks);
        const wavForA2F = pcmToWav(fullPcm);
        a2fCall.write({ audio_with_emotion: { audio_buffer: wavForA2F } });
        await new Promise(r => setTimeout(r, 50));
        a2fCall.end();

        // Build WAV for browser playback separately
        const wavBuffer = pcmToWav(fullPcm);
        console.log(`[JIT] Sentence ${idx} WAV ready: ${wavBuffer.length} bytes`);

        // Send audio to client
        send(ws, {
          type: 'npc_response',
          audio: wavBuffer.toString('base64'),
          sentenceIndex: idx,
          blendshapes: [],
          fps: 30,
        });

        if (idx === 0) logTime('First audio sent', t0);

        // Send filler motion so client can play idle animation while A2F processes
        send(ws, {
          type: 'npc_filler_motion',
          sentenceIndex: idx,
          durationMs: Math.round((wavBuffer.length / (16000 * 2)) * 1000),
        });

        // When A2F resolves for this sentence, send blendshapes immediately
        framesPromise.then((frames) => {
          if (idx === 0) logTime('First A2F blendshapes ready', t0);
          if (frames.length > 0) {
            send(ws, { type: 'npc_sentence_blendshapes', sentenceIndex: idx, frames, fps: 30 });
          }
        }).catch(() => { /* logged inside openA2FCall */ });
      }

      const sentencePromises: Promise<void>[] = [];

      for await (const chunk of llmStream) {
        const token = chunk.choices[0]?.delta?.content ?? '';
        fullResponse += token;
        sentenceBuffer += token;
        const { sentences, remainder } = extractCompleteSentences(sentenceBuffer, firstSentenceSent);
        if (sentences.length > 0) {
          firstSentenceSent = true;
          sentenceBuffer = remainder;
          for (const sentence of sentences) {
            const prev = sentencePromises[sentencePromises.length - 1] ?? Promise.resolve();
            sentencePromises.push(prev.then(() => processSentence(sentence)));
          }
        }
      }

      console.log(`[startConversation] Greeting: "${fullResponse}"`);
      logTime('LLM complete', t0);
      console.log(`[PERF] LLM response length: ${fullResponse.length} chars`);

      // Handle any remaining text without terminal punctuation
      if (sentenceBuffer.trim()) {
        const prev = sentencePromises[sentencePromises.length - 1] ?? Promise.resolve();
        sentencePromises.push(prev.then(() => processSentence(sentenceBuffer.trim())));
      }

      await Promise.all(sentencePromises);

      console.log(`[JIT] All sentences sent`);
      send(ws, { type: 'response_end' });
    } catch (err: unknown) {
      console.error('[startConversation] Error:', err instanceof Error ? err.message : err);
      send(ws, { type: 'error', message: err instanceof Error ? err.message : 'Greeting error' });
    } finally {
      isSpeaking = false;
      // Do NOT reconnect Deepgram here — client will signal when playback ends
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
      endpointing: 100,
      utterance_end_ms: 1000,
      no_delay: true,
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
        if (!firstSegmentMs) {
          firstSegmentMs = Date.now();
          console.log(`[PERF-DG] First is_final segment — ${Date.now() - lastAudioReceivedMs}ms after last audio chunk`);
        }
        currentTranscript += (currentTranscript ? ' ' : '') + t;
        lastSegmentConfidence = confidence;
        console.log(`[Deepgram] Segment final — confidence:${confidence.toFixed(3)} transcript:"${t}" accumulated:"${currentTranscript}"`);
      }

      // Only fire when speech_final=true AND we have accumulated text with good confidence
      if (msg.speech_final) {
        const toSend = currentTranscript.trim();
        currentTranscript = '';
        if (!toSend) return;
        if (lastSegmentConfidence < 0.5) {
          console.log(`[Deepgram] speech_final but low confidence (${lastSegmentConfidence.toFixed(3)}) — skipping`);
          return;
        }
        console.log(`[PERF-DG] speech_final fired — ${Date.now() - firstSegmentMs}ms after first segment, ${Date.now() - lastAudioReceivedMs}ms after last audio`);
        firstSegmentMs = 0;
        lastSegmentConfidence = 0;
        console.log(`[Deepgram] Utterance complete — sending: "${toSend}"`);
        processingUtterance = true;
        handleTranscript(toSend).finally(() => { processingUtterance = false; });
      }
    });
  }

  ws.on('message', (data: Buffer, isBinary: boolean) => {
    if (!config) {
      // First message must be the init JSON
      try {
        const msg = JSON.parse(data.toString()) as { type?: string; llmModel?: string } & Partial<SessionConfig>;
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
          llmModel: msg.llmModel ?? 'gpt-4o-mini',
        };
        try {
          initDeepgram();
          send(ws, { type: 'ready' });
          console.log(`[WS] Session initialized (llmModel: ${config.llmModel})`);
          startConversation();
        } catch (err: unknown) {
          send(ws, { type: 'error', message: `ASR init failed: ${err instanceof Error ? err.message : String(err)}` });
        }
      } catch {
        send(ws, { type: 'error', message: 'Invalid init message' });
      }
      return;
    }

    // JSON control messages from client
    if (!isBinary) {
      try {
        const parsed = JSON.parse(data.toString()) as { type?: string };
        if (parsed.type === 'playback_complete') {
          console.log('[WS] Playback complete — reconnecting Deepgram');
          try { reconnectDeepgram(); } catch (err) { console.error('[Deepgram] Reconnect failed:', err); }
          return;
        }
      } catch { /* ignore parse errors */ }
      return;
    }

    // Subsequent binary messages are raw Int16 PCM audio
    if (isBinary) {
      lastAudioReceivedMs = Date.now();
      incomingAudioChunks.push(Buffer.from(data));
      if (!isSpeaking) {
        if (deepgramSocket && deepgramReady) {
          deepgramSocket.send(data);
        } else if (deepgramSocket) {
          pendingAudioBuffer.push(Buffer.from(data));
        }
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
