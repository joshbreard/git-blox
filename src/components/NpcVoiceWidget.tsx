/**
 * NpcVoiceWidget
 *
 * "Talk to NPC" button that:
 * 1. Captures mic via Web Audio API (ScriptProcessorNode → Int16 PCM at 16 kHz)
 * 2. Opens a WebSocket to the NPC voice backend
 * 3. Streams PCM + personality config on init
 * 4. Receives npc_response JSON with base64 WAV audio + blendshape frames
 * 5. Decodes and plays audio; simultaneously steps through blendshape frames at 30fps via setInterval
 */

import { useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { useEditorStore } from '../store/editorStore';
import { engineRef } from '../engine/engineRef';
import arkitMap from '../engine/arkit-blendshape-map.json';

const WS_URL = 'ws://localhost:3001';
const TARGET_SAMPLE_RATE = 16000;
const SCRIPT_PROC_BUFFER = 2048;

type Status = 'idle' | 'connecting' | 'listening' | 'responding' | 'error';

interface BlendshapeFrame {
  timestamp: number;
  values: Record<string, number>;
}

/** Resolve ARKit blendshape key → morph target index in the given mesh (or -1). */
function resolveBlendshapeIndex(
  mesh: THREE.Mesh,
  arkitKey: string,
): number {
  const dict = mesh.morphTargetDictionary;
  if (!dict) return -1;
  // A2F returns PascalCase names (e.g. "JawOpen"); ARKit map uses camelCase ("jawOpen") — normalize
  const normalizedKey = arkitKey.charAt(0).toLowerCase() + arkitKey.slice(1);
  const candidates = ((arkitMap as unknown) as Record<string, string[]>)[normalizedKey] ?? [];
  for (const name of candidates) {
    if (name in dict) return dict[name];
  }
  return -1;
}

/** Parse a WAV buffer and return the byte offset where the PCM data chunk begins.
 *  Scans RIFF sub-chunks for the "data" fourCC. Falls back to 44 if not found. */
function findWavDataOffset(bytes: Uint8Array): number {
  // Minimum valid RIFF header is 12 bytes ("RIFF" + size + "WAVE")
  if (bytes.length < 12) return 44;
  const dec = new TextDecoder('ascii');
  const riff = dec.decode(bytes.slice(0, 4));
  const wave = dec.decode(bytes.slice(8, 12));
  if (riff !== 'RIFF' || wave !== 'WAVE') return 44;
  let offset = 12;
  while (offset + 8 <= bytes.length) {
    const id = dec.decode(bytes.slice(offset, offset + 4));
    const chunkSize = new DataView(bytes.buffer, bytes.byteOffset + offset + 4, 4).getUint32(0, true);
    offset += 8;
    if (id === 'data') return offset;
    // Chunks are word-aligned — advance by chunkSize rounded up to even
    offset += chunkSize + (chunkSize & 1);
  }
  return 44;
}

/** Find the first SkinnedMesh descendant of an Object3D that has morph targets. */
function findMorphMesh(root: THREE.Object3D): THREE.Mesh | null {
  let found: THREE.Mesh | null = null;
  root.traverse((child) => {
    if (found) return;
    const mesh = child as THREE.Mesh;
    if (mesh.isMesh && mesh.morphTargetDictionary && Object.keys(mesh.morphTargetDictionary).length > 0) {
      found = mesh;
    }
  });
  return found;
}

export default function NpcVoiceWidget({ objectId }: { objectId: string }) {
  const config = useEditorStore((s) => s.npcConfig);
  const objects = useEditorStore((s) => s.objects);
  const obj = objects[objectId];
  const personality = obj?.npcPersonality;
  const status = useEditorStore((s) => s.npcVoiceStatus);
  const setStatus = useEditorStore((s) => s.setNpcVoiceStatus);

  const [errorMsg, setErrorMsg] = useState('');

  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const listeningRef = useRef<boolean>(false);

  const blendshapeIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const morphMeshRef = useRef<THREE.Mesh | null>(null);
  const audioStartMsRef = useRef<number>(0);

  const isActive = status !== 'idle' && status !== 'error';

  // ── Stop / cleanup ─────────────────────────────────────────────────────────
  const stop = useCallback(() => {
    if (blendshapeIntervalRef.current) {
      clearInterval(blendshapeIntervalRef.current);
      blendshapeIntervalRef.current = null;
    }
    processorRef.current?.disconnect();
    processorRef.current = null;
    micSourceRef.current?.disconnect();
    micSourceRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    playbackCtxRef.current?.close().catch(() => {});
    playbackCtxRef.current = null;
    wsRef.current?.close();
    wsRef.current = null;
    listeningRef.current = false;
    setStatus('idle');

    // Reset morph targets on the mesh
    const mesh = morphMeshRef.current;
    if (mesh?.morphTargetInfluences) {
      for (let i = 0; i < mesh.morphTargetInfluences.length; i++) {
        mesh.morphTargetInfluences[i] = 0;
      }
    }
    morphMeshRef.current = null;
  }, []);

  // ── Start ──────────────────────────────────────────────────────────────────
  const start = useCallback(async () => {
    if (!config.openAiKey || !config.deepgramKey || !config.elevenLabsKey || !config.elevenLabsVoiceId) {
      setErrorMsg('Fill in OpenAI, Deepgram, and ElevenLabs keys in the NPC Voice Config panel.');
      setStatus('error');
      return;
    }
    if (!personality) {
      setErrorMsg('No NPC personality configured. Select a character and set up its personality in the NPC Voice Config panel.');
      setStatus('error');
      return;
    }

    setStatus('connecting');
    setErrorMsg('');

    try {
      // ── Locate morph-target mesh for this object ──────────────────────────
      const sceneObj = engineRef.current?.sceneManager?.getMeshById(objectId);
      if (sceneObj) {
        morphMeshRef.current = findMorphMesh(sceneObj);
        if (morphMeshRef.current) {
          const names = Object.keys(morphMeshRef.current.morphTargetDictionary ?? {});
          console.log('[NpcVoice] Morph mesh found:', morphMeshRef.current.name,
            '—', names.length, 'targets:', names.slice(0, 10).join(', '));
        } else {
          console.warn('[NpcVoice] No morph mesh found on object', objectId,
            '— re-import the GLB and check [SceneManager] logs for morph target counts');
        }
      } else {
        console.warn('[NpcVoice] getMeshById returned null for', objectId);
      }

      // ── Mic capture ───────────────────────────────────────────────────────
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      streamRef.current = stream;

      const audioCtx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
      // Resume immediately while still inside the user-gesture call stack.
      // Calling resume() later (e.g. inside ws.onmessage) may be blocked by
      // browsers that require AudioContext activation from a user gesture.
      await audioCtx.resume();
      audioCtxRef.current = audioCtx;

      // Separate AudioContext for NPC audio playback — created and resumed here
      // inside the user gesture so it is guaranteed to be in "running" state
      // when npc_response arrives. Reusing a single context avoids the
      // create/close churn per response that causes CoreAudio pops and static.
      const playbackCtx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
      await playbackCtx.resume();
      playbackCtxRef.current = playbackCtx;

      const micSource = audioCtx.createMediaStreamSource(stream);
      micSourceRef.current = micSource;

      const processor = audioCtx.createScriptProcessor(SCRIPT_PROC_BUFFER, 1, 1);
      processorRef.current = processor;
      micSource.connect(processor);
      processor.connect(audioCtx.createGain());

      // ── WebSocket ─────────────────────────────────────────────────────────
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        // Send init
        ws.send(JSON.stringify({
          type: 'init',
          personalityPrompt: personality.systemPrompt,
          openAiKey: config.openAiKey,
          deepgramKey: config.deepgramKey,
          elevenLabsKey: config.elevenLabsKey,
          voiceId: config.elevenLabsVoiceId,
          nvidiaApiKey: config.nvidiaApiKey,
          nvidiaFunctionId: config.nvidiaFunctionId,
        }));
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data as string) as { type: string; [k: string]: unknown };
          switch (msg.type) {
            case 'ready':
              // Server will speak first — stay in responding state until response_end
              listeningRef.current = false;
              setStatus('responding');
              break;
            case 'response_start':
              listeningRef.current = false;
              setStatus('responding');
              break;
            case 'npc_response': {
              const audioB64 = msg.audio as string;
              const frames = msg.blendshapes as BlendshapeFrame[];
              const fps = (msg.fps as number) ?? 30;

              // Record when audio starts so blendshapes arriving later can sync
              audioStartMsRef.current = performance.now();

              // Decode base64 WAV → AudioBuffer and play.
              // We bypass decodeAudioData because browsers reject PCM WAVs whose
              // data chunk has an odd byte length (ElevenLabs pcm_16000 sometimes
              // returns an odd number of bytes, making the RIFF headers invalid).
              // Instead, manually parse the known format: 44-byte header, 16kHz,
              // mono, Int16 little-endian PCM.
              const bytes = Uint8Array.from(atob(audioB64), (c) => c.charCodeAt(0));
              console.log('[Audio] Received WAV, byte length:', bytes.length);
              const playCtx = playbackCtxRef.current;
              if (playCtx) {
                try {
                  const WAV_HEADER = findWavDataOffset(bytes);
                  const pcmBytes = bytes.length - WAV_HEADER;
                  const numSamples = Math.floor(pcmBytes / 2);
                  const audioBuffer = playCtx.createBuffer(1, numSamples, TARGET_SAMPLE_RATE);
                  const channelData = audioBuffer.getChannelData(0);
                  const view = new DataView(bytes.buffer, bytes.byteOffset + WAV_HEADER);
                  for (let i = 0; i < numSamples; i++) {
                    channelData[i] = view.getInt16(i * 2, true) / 32768;
                  }
                  console.log('[Audio] PCM decoded — samples:', numSamples,
                    'duration:', audioBuffer.duration.toFixed(2), 's');
                  const src = playCtx.createBufferSource();
                  src.buffer = audioBuffer;
                  src.connect(playCtx.destination);
                  src.start(playCtx.currentTime);
                } catch (err) {
                  console.error('[Audio] PCM decode failed:', err);
                }
              } else {
                console.error('[Audio] Playback AudioContext is null — cannot play');
              }

              // Drive blendshapes inline only if server sent them with the audio (warm A2F path)
              if (blendshapeIntervalRef.current) clearInterval(blendshapeIntervalRef.current);
              const mesh = morphMeshRef.current;
              if (mesh && frames.length > 0) {
                let frameIdx = 0;
                blendshapeIntervalRef.current = setInterval(() => {
                  if (frameIdx >= frames.length) {
                    clearInterval(blendshapeIntervalRef.current!);
                    blendshapeIntervalRef.current = null;
                    return;
                  }
                  const frame = frames[frameIdx++];
                  if (!mesh.morphTargetInfluences || !mesh.morphTargetDictionary) return;
                  for (const [arkitKey, weight] of Object.entries(frame.values)) {
                    const idx = resolveBlendshapeIndex(mesh, arkitKey);
                    if (idx >= 0) mesh.morphTargetInfluences[idx] = weight;
                  }
                }, 1000 / fps);
              }
              break;
            }

            case 'npc_blendshapes': {
              // A2F completed after audio was already sent — sync animation to elapsed playback time
              const frames = msg.frames as BlendshapeFrame[];
              const fps = (msg.fps as number) ?? 30;
              const mesh = morphMeshRef.current;
              if (!mesh || frames.length === 0) {
                console.warn('[NpcVoice] npc_blendshapes received but',
                  !mesh ? 'morph mesh is null' : 'frames array is empty');
                break;
              }

              const elapsedMs = performance.now() - audioStartMsRef.current;
              const totalDurationMs = (frames.length / fps) * 1000;
              // If A2F arrived after the audio already finished, replay from frame 0.
              // Without this guard startFrameIdx is clamped to frames.length-1 and
              // the interval fires exactly once — no visible animation.
              const rawStartIdx = Math.floor(elapsedMs / (1000 / fps));
              const startFrameIdx = rawStartIdx >= frames.length ? 0
                : Math.min(rawStartIdx, frames.length - 1);
              console.log('[NpcVoice] npc_blendshapes — frames:', frames.length,
                'elapsedMs:', elapsedMs.toFixed(0), 'totalDurationMs:', totalDurationMs.toFixed(0),
                'startFrame:', startFrameIdx, elapsedMs >= totalDurationMs ? '(replaying from 0)' : '(synced)');

              if (blendshapeIntervalRef.current) clearInterval(blendshapeIntervalRef.current);
              let frameIdx = startFrameIdx;
              blendshapeIntervalRef.current = setInterval(() => {
                if (frameIdx >= frames.length) {
                  clearInterval(blendshapeIntervalRef.current!);
                  blendshapeIntervalRef.current = null;
                  return;
                }
                const frame = frames[frameIdx++];
                if (!mesh.morphTargetInfluences || !mesh.morphTargetDictionary) return;
                for (const [arkitKey, weight] of Object.entries(frame.values)) {
                  const idx = resolveBlendshapeIndex(mesh, arkitKey);
                  if (idx >= 0) mesh.morphTargetInfluences[idx] = weight;
                }
              }, 1000 / fps);
              break;
            }
            case 'response_end':
              listeningRef.current = true;
              setStatus('listening');
              break;
            case 'error':
              setErrorMsg(String(msg.message ?? 'Server error'));
              setStatus('error');
              break;
          }
        } catch { /* ignore parse errors */ }
      };

      ws.onerror = () => {
        listeningRef.current = false;
        setErrorMsg('Cannot connect to NPC voice server (ws://localhost:3001). Is it running?');
        setStatus('error');
      };

      ws.onclose = () => {
        listeningRef.current = false;
        if (status !== 'idle') setStatus('idle');
      };

      // ── PCM streaming ─────────────────────────────────────────────────────
      processor.onaudioprocess = (e) => {
        if (!listeningRef.current || ws.readyState !== WebSocket.OPEN) return;
        const float32 = e.inputBuffer.getChannelData(0);
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
        }
        ws.send(int16.buffer);
      };
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : 'Microphone access denied');
      setStatus('error');
      stop();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config, personality, objectId, stop]);

  const toggle = useCallback(() => {
    if (isActive) {
      stop();
    } else {
      start();
    }
  }, [isActive, start, stop]);

  const label =
    status === 'idle' ? 'Talk to NPC'
    : status === 'connecting' ? 'Connecting...'
    : status === 'listening' ? 'Listening...'
    : status === 'responding' ? 'Speaking...'
    : 'Error';

  return (
    <div className="npc-voice-widget">
      <button
        className={`npc-talk-btn ${status === 'listening' ? 'listening' : ''} ${status === 'responding' ? 'responding' : ''} ${status === 'error' ? 'error' : ''}`}
        onClick={toggle}
        title={errorMsg || label}
      >
        <span className="npc-talk-icon">{isActive ? '■' : '◉'}</span>
        {label}
      </button>
      {status === 'error' && (
        <div className="npc-talk-error" onClick={() => setStatus('idle')}>
          {errorMsg} <span style={{ opacity: 0.5 }}>(click to dismiss)</span>
        </div>
      )}
    </div>
  );
}
