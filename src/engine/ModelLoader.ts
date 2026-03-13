import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

export interface LoadedModel {
  name: string;
  geometry: THREE.BufferGeometry;
  animations: THREE.AnimationClip[];
  /** Original scene root, only set when animations are present */
  scene: THREE.Object3D | null;
  /** True for GLB/GLTF imports where the scene hierarchy is used as-is */
  hasScene: boolean;
}

function extractGeometries(object: THREE.Object3D): THREE.BufferGeometry[] {
  const geos: THREE.BufferGeometry[] = [];
  object.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      const mesh = child as THREE.Mesh;
      const geo = mesh.geometry.clone();
      mesh.updateMatrixWorld();
      geo.applyMatrix4(mesh.matrixWorld);
      geos.push(geo);
    }
  });
  return geos;
}

export async function loadModelFile(file: File): Promise<LoadedModel> {
  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  const name = file.name.replace(/\.[^.]+$/, '');
  const arrayBuffer = await file.arrayBuffer();
  const url = URL.createObjectURL(new Blob([arrayBuffer]));

  try {
    if (ext === 'glb' || ext === 'gltf') {
      const loader = new GLTFLoader();
      const gltf = await loader.loadAsync(url);
      // Preserve morph targets and skinning by keeping the full scene hierarchy.
      // Normalize height the same way as other formats, but operate on the root
      // transform so submesh geometry (and its morph data) is never cloned.
      const root = gltf.scene;
      const box = new THREE.Box3().setFromObject(root);
      const height = box.max.y - box.min.y;
      if (height > 0.001) {
        const s = 2 / height;
        root.scale.multiplyScalar(s);
      }
      return {
        name,
        geometry: new THREE.BufferGeometry(),
        animations: gltf.animations ?? [],
        scene: root,
        hasScene: true,
      };
    }

    let root: THREE.Object3D;
    let animations: THREE.AnimationClip[] = [];

    if (ext === 'fbx') {
      const loader = new FBXLoader();
      root = await loader.loadAsync(url);
      animations = (root as THREE.Group & { animations?: THREE.AnimationClip[] }).animations ?? [];
    } else if (ext === 'obj') {
      const loader = new OBJLoader();
      const text = new TextDecoder().decode(arrayBuffer);
      root = loader.parse(text);
    } else {
      throw new Error(`Unsupported format: .${ext}`);
    }

    const geos = extractGeometries(root);
    if (geos.length === 0) throw new Error('No meshes found in file');

    let merged: THREE.BufferGeometry;
    if (geos.length === 1) {
      merged = geos[0];
    } else {
      const normalized = geos.map((g) => {
        if (!g.index) return g;
        return g.toNonIndexed();
      });
      merged = mergeGeometries(normalized, false) ?? normalized[0];
    }

    merged.computeVertexNormals();
    merged.computeBoundingBox();

    const TARGET_HEIGHT = 2;
    const box = merged.boundingBox!;
    const height = box.max.y - box.min.y;
    if (height > 0.001) {
      const s = TARGET_HEIGHT / height;
      merged.scale(s, s, s);
      root.scale.multiplyScalar(s);
    }
    merged.computeBoundingSphere();

    return {
      name,
      geometry: merged,
      animations,
      scene: root,
      hasScene: false,
    };
  } finally {
    URL.revokeObjectURL(url);
  }
}
