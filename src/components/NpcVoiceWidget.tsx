/**
 * NpcVoiceWidget
 *
 * "Talk to NPC" button that:
 * 1. Captures mic via Web Audio API (AudioWorkletNode → Int16 PCM at 16 kHz)
 * 2. Opens a WebSocket to the NPC voice backend
 * 3. Streams PCM + personality config on init
 * 4. Receives per-sentence npc_response with base64 WAV audio
 * 5. Receives per-sentence npc_sentence_blendshapes with A2F frames
 * 6. Plays filler idle animation until real blendshapes arrive
 * 7. Syncs blendshape frames to audio playback position via requestAnimationFrame
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

/** Apply idle/filler blendshape animation: subtle jaw oscillation + periodic blink. */
function applyFillerAnimation(mesh: THREE.Mesh, timeMs: number) {
  if (!mesh.morphTargetInfluences || !mesh.morphTargetDictionary) return;

  // Reset all morph targets to neutral
  for (let i = 0; i < mesh.morphTargetInfluences.length; i++) {
    mesh.morphTargetInfluences[i] = 0;
  }

  // Subtle jawOpen oscillation at ~2Hz (0 → 0.04)
  const jawVal = Math.abs(Math.sin(timeMs / 1000 * Math.PI * 4)) * 0.04;
  const jawIdx = resolveBlendshapeIndex(mesh, 'jawOpen');
  if (jawIdx >= 0) mesh.morphTargetInfluences[jawIdx] = jawVal;

  // Eye blink every ~4 seconds (quick 150ms close-open triangle wave)
  const BLINK_PERIOD = 4.0;
  const BLINK_DURATION = 0.15;
  const t = (timeMs / 1000) % BLINK_PERIOD;
  let blinkVal = 0;
  if (t > BLINK_PERIOD - BLINK_DURATION) {
    const p = (t - (BLINK_PERIOD - BLINK_DURATION)) / BLINK_DURATION;
    blinkVal = p < 0.5 ? p * 2 : (1 - p) * 2;
  }
  const leftIdx = resolveBlendshapeIndex(mesh, 'eyeBlinkLeft');
  const rightIdx = resolveBlendshapeIndex(mesh, 'eyeBlinkRight');
  if (leftIdx >= 0) mesh.morphTargetInfluences[leftIdx] = blinkVal;
  if (rightIdx >= 0) mesh.morphTargetInfluences[rightIdx] = blinkVal;
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

  const morphMeshRef = useRef<THREE.Mesh | null>(null);
  const nextStartTimeRef = useRef<number>(0);

  // Audio queue for sequential playback of npc_response messages
  const audioQueueRef = useRef<Array<{ audio: string; sentenceIndex: number }>>([]);
  const isPlayingAudioRef = useRef(false);

  // Per-sentence blendshape state
  const sentenceBlendshapesRef = useRef<Map<number, BlendshapeFrame[]>>(new Map());
  const sentenceAudioRef = useRef<Map<number, { startTime: number; duration: number }>>(new Map());
  const fillerEnabledRef = useRef<Set<number>>(new Set());
  const blendshapeRafRef = useRef<number | null>(null);

  const isActive = status !== 'idle' && status !== 'error';

  // ── Animation loop ────────────────────────────────────────────────────────
  function startBlendshapeLoop() {
    if (blendshapeRafRef.current !== null) return; // already running

    function tick() {
      const ctx = playbackCtxRef.current;
      const mesh = morphMeshRef.current;
      if (!ctx || !mesh?.morphTargetInfluences) {
        blendshapeRafRef.current = requestAnimationFrame(tick);
        return;
      }

      const now = ctx.currentTime;

      // Find which sentence is currently playing
      let currentSentence = -1;
      let sentenceOffset = 0;
      for (const [idx, info] of sentenceAudioRef.current.entries()) {
        if (now >= info.startTime && now < info.startTime + info.duration) {
          currentSentence = idx;
          sentenceOffset = now - info.startTime;
          break;
        }
      }

      if (currentSentence >= 0) {
        const frames = sentenceBlendshapesRef.current.get(currentSentence);
        if (frames && frames.length > 0) {
          // Apply real A2F blendshapes synced to audio playback position
          const fps = 30;
          const frameIdx = Math.min(Math.floor(sentenceOffset * fps), frames.length - 1);
          const frame = frames[frameIdx];
          for (const [arkitKey, weight] of Object.entries(frame.values)) {
            const morphIdx = resolveBlendshapeIndex(mesh, arkitKey);
            if (morphIdx >= 0) mesh.morphTargetInfluences[morphIdx] = weight;
          }
        } else if (fillerEnabledRef.current.has(currentSentence)) {
          // Blendshapes haven't arrived yet — play filler idle animation
          applyFillerAnimation(mesh, performance.now());
        }
      }

      blendshapeRafRef.current = requestAnimationFrame(tick);
    }

    blendshapeRafRef.current = requestAnimationFrame(tick);
  }

  function stopBlendshapeLoop() {
    if (blendshapeRafRef.current !== null) {
      cancelAnimationFrame(blendshapeRafRef.current);
      blendshapeRafRef.current = null;
    }
  }

  /** Clear all per-sentence animation state and cancel the animation loop. */
  function clearAnimationState() {
    stopBlendshapeLoop();
    sentenceBlendshapesRef.current.clear();
    sentenceAudioRef.current.clear();
    fillerEnabledRef.current.clear();

    // Reset morph targets to neutral
    const mesh = morphMeshRef.current;
    if (mesh?.morphTargetInfluences) {
      for (let i = 0; i < mesh.morphTargetInfluences.length; i++) {
        mesh.morphTargetInfluences[i] = 0;
      }
    }
  }

  // ── Audio queue playback ───────────────────────────────────────────────────
  function playNextInQueue() {
    if (isPlayingAudioRef.current) return;
    const next = audioQueueRef.current.shift();
    if (!next) return;

    isPlayingAudioRef.current = true;
    const { audio, sentenceIndex } = next;

    const ctx = playbackCtxRef.current;
    if (!ctx) {
      console.error('[Audio] Playback AudioContext is null — cannot play');
      isPlayingAudioRef.current = false;
      return;
    }

    const u8 = Uint8Array.from(atob(audio), c => c.charCodeAt(0));
    const arrayBuf = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);

    ctx.decodeAudioData(arrayBuf as ArrayBuffer).then(audioBuf => {
      const startTime = ctx.currentTime + 0.05;

      // Track audio timing for blendshape sync
      sentenceAudioRef.current.set(sentenceIndex, {
        startTime,
        duration: audioBuf.duration,
      });

      const src = ctx.createBufferSource();
      src.buffer = audioBuf;
      src.connect(ctx.destination);
      src.start(startTime);

      console.log(`[PERF-CLIENT] Sentence ${sentenceIndex} playing at ${startTime.toFixed(2)}s, duration: ${audioBuf.duration.toFixed(2)}s`);

      // Start the blendshape animation loop (idempotent — only starts once)
      startBlendshapeLoop();

      src.onended = () => {
        isPlayingAudioRef.current = false;
        if (audioQueueRef.current.length > 0) {
          playNextInQueue();
        } else {
          // Full queue drained — notify server
          const ws = wsRef.current;
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'playback_complete' }));
          }
        }
      };
    }).catch(err => {
      console.error('[Audio] decodeAudioData failed:', err);
      isPlayingAudioRef.current = false;
      playNextInQueue();
    });
  }

  // ── Stop / cleanup ─────────────────────────────────────────────────────────
  const stop = useCallback(() => {
    clearAnimationState();
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
              // Clear queue for new response
              audioQueueRef.current = [];
              isPlayingAudioRef.current = false;
              break;

            case 'npc_response': {
              const { audio, sentenceIndex } = msg as { audio: string; sentenceIndex: number; type: string };
              audioQueueRef.current.push({ audio, sentenceIndex });
              playNextInQueue();
              break;
            }

            case 'npc_filler_motion': {
              const { sentenceIndex } = msg as { sentenceIndex: number; durationMs: number; type: string };
              // Mark this sentence for filler animation — the animation loop will
              // play idle motion for it until real blendshapes arrive
              fillerEnabledRef.current.add(sentenceIndex);
              break;
            }

            case 'npc_sentence_blendshapes': {
              const { sentenceIndex, frames } = msg as {
                sentenceIndex: number;
                frames: BlendshapeFrame[];
                fps: number;
                type: string;
              };
              // Store per-sentence blendshapes — the animation loop will pick them
              // up and swap from filler to real frames automatically
              sentenceBlendshapesRef.current.set(sentenceIndex, frames);
              // Filler is no longer needed for this sentence
              fillerEnabledRef.current.delete(sentenceIndex);
              console.log(`[NpcVoice] Blendshapes received for sentence ${sentenceIndex}: ${frames.length} frames`);
              break;
            }

            case 'response_end':
              listeningRef.current = true;
              setStatus('listening');
              clearAnimationState();
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
