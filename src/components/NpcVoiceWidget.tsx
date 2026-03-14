/**
 * NpcVoiceWidget
 *
 * "Talk to NPC" button that:
 * 1. Captures mic via Web Audio API (AudioWorkletNode → Int16 PCM at 16 kHz)
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
  const processorRef = useRef<AudioWorkletNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const listeningRef = useRef<boolean>(false);

  const rafIdRef = useRef<number | null>(null);
  const morphMeshRef = useRef<THREE.Mesh | null>(null);
  const audioScheduledAtRef = useRef<number>(0);
  const nextStartTimeRef = useRef<number>(0);

  const isActive = status !== 'idle' && status !== 'error';

  // ── Stop / cleanup ─────────────────────────────────────────────────────────
  const stop = useCallback(() => {
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
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
      await audioCtx.audioWorklet.addModule('/mic-processor.js');
      audioCtxRef.current = audioCtx;

      // Separate AudioContext for NPC audio playback — created and resumed here
      // inside the user gesture so it is guaranteed to be in "running" state
      // when npc_response arrives. Reusing a single context avoids the
      // create/close churn per response that causes CoreAudio pops and static.
      const playbackCtx = new AudioContext();
      await playbackCtx.resume();
      playbackCtxRef.current = playbackCtx;

      const micSource = audioCtx.createMediaStreamSource(stream);
      micSourceRef.current = micSource;

      const processor = new AudioWorkletNode(audioCtx, 'mic-processor');
      processorRef.current = processor;
      micSource.connect(processor);

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
              const { audio } = msg as { audio: string; sentenceIndex: number };

              const ctx = playbackCtxRef.current;
              if (!ctx) {
                console.error('[Audio] Playback AudioContext is null — cannot play');
                break;
              }

              const u8 = Uint8Array.from(atob(audio), c => c.charCodeAt(0));
              const arrayBuf = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);

              const reservedStart = Math.max(ctx.currentTime + 0.05, nextStartTimeRef.current);
              nextStartTimeRef.current = reservedStart + 5; // reserve 5s slot, will shrink after decode

              ctx.decodeAudioData(arrayBuf as ArrayBuffer).then(audioBuf => {
                // Update nextStartTimeRef with actual duration now that we know it
                const actualEnd = reservedStart + audioBuf.duration;
                if (nextStartTimeRef.current === reservedStart + 5) {
                  // Only update if no newer sentence has already claimed this slot
                  nextStartTimeRef.current = actualEnd;
                } else {
                  // A later sentence already reserved; don't shrink their slot
                  nextStartTimeRef.current = Math.max(nextStartTimeRef.current, actualEnd);
                }

                const src = ctx.createBufferSource();
                src.buffer = audioBuf;
                src.connect(ctx.destination);
                src.start(reservedStart);
                audioScheduledAtRef.current = reservedStart;

                console.log(`[PERF-CLIENT] Sentence scheduled at ${reservedStart.toFixed(2)}s, duration: ${audioBuf.duration.toFixed(2)}s`);
              }).catch(err => console.error('[Audio] decodeAudioData failed:', err));
              break;
            }

            case 'npc_blendshapes': {
              const frames = msg.frames as BlendshapeFrame[];
              const fps = (msg.fps as number) ?? 30;
              const mesh = morphMeshRef.current;
              if (!mesh || frames.length === 0) {
                console.warn('[NpcVoice] npc_blendshapes received but',
                  !mesh ? 'morph mesh is null' : 'frames array is empty');
                break;
              }
              const ctx = playbackCtxRef.current;
              if (!ctx) break;

              if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);

              // Build index map once — skip keys not found in this mesh
              const indexMap = new Map<string, number>();
              for (const key of Object.keys(arkitMap as Record<string, unknown>)) {
                const idx = resolveBlendshapeIndex(mesh, key);
                if (idx >= 0) indexMap.set(key, idx);
              }

              const tick = () => {
                const t = Math.max(0, ctx.currentTime - audioScheduledAtRef.current);

                // Binary search: last frame where frame.timestamp <= t
                let lo = 0, hi = frames.length - 1, i = 0;
                while (lo <= hi) {
                  const mid = (lo + hi) >> 1;
                  if (frames[mid].timestamp <= t) { i = mid; lo = mid + 1; }
                  else hi = mid - 1;
                }

                // Past the last frame — zero out and stop
                if (t > frames[frames.length - 1].timestamp) {
                  if (mesh.morphTargetInfluences) {
                    for (const idx of indexMap.values()) {
                      mesh.morphTargetInfluences[idx] = 0;
                    }
                  }
                  rafIdRef.current = null;
                  return;
                }

                if (i < frames.length - 1) {
                  const alpha = (t - frames[i].timestamp) / (frames[i + 1].timestamp - frames[i].timestamp);
                  if (mesh.morphTargetInfluences) {
                    for (const [key, idx] of indexMap) {
                      mesh.morphTargetInfluences[idx] =
                        (frames[i].values[key] ?? 0) * (1 - alpha) +
                        (frames[i + 1].values[key] ?? 0) * alpha;
                    }
                  }
                } else {
                  if (mesh.morphTargetInfluences) {
                    for (const [key, idx] of indexMap) {
                      mesh.morphTargetInfluences[idx] = frames[i].values[key] ?? 0;
                    }
                  }
                }

                rafIdRef.current = requestAnimationFrame(tick);
              };

              rafIdRef.current = requestAnimationFrame(tick);
              break;
            }
            case 'response_end': {
              const ctx = playbackCtxRef.current;
              const msRemaining = ctx && nextStartTimeRef.current > ctx.currentTime
                ? (nextStartTimeRef.current - ctx.currentTime) * 1000
                : 0;
              setTimeout(() => {
                if (wsRef.current?.readyState === WebSocket.OPEN) {
                  wsRef.current.send(JSON.stringify({ type: 'playback_complete' }));
                  console.log('[CLIENT] playback_complete sent to server');
                }
                nextStartTimeRef.current = 0;
                listeningRef.current = true;
                setStatus('listening');
              }, msRemaining + 200);
              break;
            }
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
        setStatus('idle');
      };

      // ── PCM streaming ─────────────────────────────────────────────────────
      processor.port.onmessage = (e: MessageEvent<Int16Array>) => {
        if (!listeningRef.current || ws.readyState !== WebSocket.OPEN) return;
        ws.send(e.data.buffer);
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
