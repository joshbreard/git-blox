/**
 * NpcVoiceWidget
 *
 * "Talk to NPC" button that:
 * 1. Captures mic via Web Audio API (ScriptProcessorNode → Int16 PCM at 16 kHz)
 * 2. Opens a WebSocket to the NPC voice backend
 * 3. Streams PCM + personality config on init
 * 4. Receives binary audio chunks → plays via AudioContext
 * 5. Receives blendshape JSON → applies to the active Three.js SkinnedMesh via morphTargetInfluences
 */

import { useRef, useState, useCallback, useEffect } from 'react';
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
  const candidates = ((arkitMap as unknown) as Record<string, string[]>)[arkitKey] ?? [];
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
  const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const listeningRef = useRef<boolean>(false);

  // Audio playback scheduling
  const nextStartTimeRef = useRef<number>(0); // AudioContext time when next chunk is scheduled
  const blendshapeQueueRef = useRef<BlendshapeFrame[]>([]);
  const blendshapeOffsetRef = useRef<number>(0); // AudioContext time when NPC started speaking
  const rafRef = useRef<number>(0);
  const morphMeshRef = useRef<THREE.Mesh | null>(null);

  const isActive = status !== 'idle' && status !== 'error';

  // ── Blendshape application loop ────────────────────────────────────────────
  const applyBlendshapes = useCallback(() => {
    rafRef.current = requestAnimationFrame(applyBlendshapes);
    const ctx = audioCtxRef.current;
    if (!ctx || !morphMeshRef.current) return;

    const relTime = ctx.currentTime - blendshapeOffsetRef.current;
    const queue = blendshapeQueueRef.current;
    if (queue.length === 0) return;

    // Find the frame closest to current playback time
    let best = queue[0];
    for (const frame of queue) {
      if (Math.abs(frame.timestamp - relTime) < Math.abs(best.timestamp - relTime)) {
        best = frame;
      }
    }

    const mesh = morphMeshRef.current;
    if (!mesh.morphTargetInfluences) return;
    for (const [arkitKey, weight] of Object.entries(best.values)) {
      const idx = resolveBlendshapeIndex(mesh, arkitKey);
      if (idx >= 0) mesh.morphTargetInfluences[idx] = weight;
    }
  }, []);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(applyBlendshapes);
    return () => cancelAnimationFrame(rafRef.current);
  }, [applyBlendshapes]);

  // ── Stop / cleanup ─────────────────────────────────────────────────────────
  const stop = useCallback(() => {
    processorRef.current?.disconnect();
    processorRef.current = null;
    micSourceRef.current?.disconnect();
    micSourceRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    wsRef.current?.close();
    wsRef.current = null;
    listeningRef.current = false;
    nextStartTimeRef.current = 0;
    setStatus('idle');

    // Reset morph targets on the mesh
    const mesh = morphMeshRef.current;
    if (mesh?.morphTargetInfluences) {
      for (let i = 0; i < mesh.morphTargetInfluences.length; i++) {
        mesh.morphTargetInfluences[i] = 0;
      }
    }
    blendshapeQueueRef.current = [];
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
      }

      // ── Mic capture ───────────────────────────────────────────────────────
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      streamRef.current = stream;

      const audioCtx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
      audioCtxRef.current = audioCtx;

      const micSource = audioCtx.createMediaStreamSource(stream);
      micSourceRef.current = micSource;

      const processor = audioCtx.createScriptProcessor(SCRIPT_PROC_BUFFER, 1, 1);
      processorRef.current = processor;
      micSource.connect(processor);
      processor.connect(audioCtx.destination);

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
        if (event.data instanceof ArrayBuffer) {
          // Audio chunk — schedule playback
          const ctx = audioCtxRef.current;
          if (!ctx) return;
          ctx.decodeAudioData(event.data.slice(0)).then((decoded) => {
            const src = ctx.createBufferSource();
            src.buffer = decoded;
            src.connect(ctx.destination);
            const startAt = Math.max(ctx.currentTime, nextStartTimeRef.current);
            src.start(startAt);
            nextStartTimeRef.current = startAt + decoded.duration;
          }).catch(() => { /* non-fatal decode error */ });
          return;
        }

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
              nextStartTimeRef.current = audioCtxRef.current?.currentTime ?? 0;
              blendshapeOffsetRef.current = audioCtxRef.current?.currentTime ?? 0;
              blendshapeQueueRef.current = [];
              break;
            case 'response_end':
              listeningRef.current = true;
              setStatus('listening');
              break;
            case 'blendshapes':
              blendshapeQueueRef.current.push({
                timestamp: msg.timestamp as number,
                values: msg.values as Record<string, number>,
              });
              break;
            case 'error':
              setErrorMsg(String(msg.message ?? 'Server error'));
              setStatus('error');
              break;
          }
        } catch { /* non-JSON binary frame handled above */ }
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
