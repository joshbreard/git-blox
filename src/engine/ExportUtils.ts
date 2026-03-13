import * as THREE from 'three';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { OBJExporter } from 'three/addons/exporters/OBJExporter.js';
import { getTask } from './MeshyAPI';

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function exportGlb(
  object: THREE.Object3D,
  name: string,
  clips: THREE.AnimationClip[] = [],
) {
  const exporter = new GLTFExporter();
  const buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
    exporter.parse(
      object,
      (result) => resolve(result as ArrayBuffer),
      reject,
      { binary: true, animations: clips },
    );
  });
  triggerDownload(new Blob([buffer], { type: 'model/gltf-binary' }), `${name}.glb`);
}

export function exportObj(object: THREE.Object3D, name: string) {
  const exporter = new OBJExporter();
  const text = exporter.parse(object);
  triggerDownload(new Blob([text], { type: 'text/plain' }), `${name}.obj`);
}

export async function exportFbxFromMeshy(meshyTaskId: string, name: string) {
  const task = await getTask(meshyTaskId);
  const fbxUrl = task.model_urls?.fbx;
  if (!fbxUrl) throw new Error('FBX URL not available for this model');
  const res = await fetch(`/api/proxy-download?url=${encodeURIComponent(fbxUrl)}`);
  if (!res.ok) throw new Error(`Failed to download FBX: ${res.status}`);
  const buffer = await res.arrayBuffer();
  triggerDownload(new Blob([buffer], { type: 'application/octet-stream' }), `${name}.fbx`);
}
