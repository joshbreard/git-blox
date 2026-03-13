import { create } from 'zustand';
import type {
  SceneObject,
  EditorMode,
  EditSubMode,
  ActiveTool,
  Vec3,
  GeometryType,
  AnimationData,
  NpcPersonality,
  NpcConfig,
} from './types';
import { NVIDIA_A2F_MODELS } from './types';

let nextId = 1;
function generateId(): string {
  return `obj_${nextId++}`;
}
function peekId(): string {
  return `obj_${nextId}`;
}

const geoNames: Record<GeometryType, string> = {
  box: 'Cube',
  sphere: 'Sphere',
  cylinder: 'Cylinder',
  cone: 'Cone',
  torus: 'Torus',
  plane: 'Plane',
  icosahedron: 'Icosahedron',
  imported: 'Import',
};

export interface EditorState {
  objects: Record<string, SceneObject>;
  selectedIds: string[];
  mode: EditorMode;
  editSubMode: EditSubMode;
  activeTool: ActiveTool;
  editObjectId: string | null;

  selectedVertices: Set<number>;
  selectedEdges: Set<number>;
  selectedFaces: Set<number>;

  enhanceScreenshot: string | null;
  enhanceResult: string | null;
  enhanceLoading: boolean;
  imageLibrary: string[];
  setEnhanceScreenshot: (s: string | null) => void;
  setEnhanceResult: (s: string | null) => void;
  setEnhanceLoading: (loading: boolean) => void;
  addToImageLibrary: (dataUri: string) => void;

  composerPrompt: string;
  composerLoading: boolean;
  composerError: string | null;
  composerRefImage: string | null;
  setComposerPrompt: (s: string) => void;
  setComposerLoading: (b: boolean) => void;
  setComposerError: (s: string | null) => void;
  setComposerRefImage: (s: string | null) => void;

  npcConfig: NpcConfig;
  setNpcConfig: (updates: Partial<NpcConfig>) => void;
  setNpcPersonality: (objectId: string, personality: NpcPersonality | null) => void;
  /** Creates a default NpcPersonality for objectId if one does not already exist. */
  ensureNpcPersonality: (objectId: string) => void;
  npcVoiceStatus: 'idle' | 'connecting' | 'listening' | 'responding' | 'error';
  setNpcVoiceStatus: (s: 'idle' | 'connecting' | 'listening' | 'responding' | 'error') => void;
  setGenerationPrompt: (objectId: string, prompt: string) => void;

  peekNextId: () => string;
  addObject: (type: GeometryType, position?: Vec3) => string;
  addImportedObject: (name: string, animations?: AnimationData[]) => string;
  setActiveAnimation: (objectId: string, animationName: string | null) => void;
  setRigTaskId: (objectId: string, taskId: string) => void;
  setMeshyTaskId: (objectId: string, taskId: string) => void;
  addAnimationsToObject: (objectId: string, anims: AnimationData[]) => void;
  removeObject: (id: string) => void;
  updateObject: (id: string, updates: Partial<SceneObject>) => void;
  setSelection: (ids: string[]) => void;
  addToSelection: (id: string) => void;
  removeFromSelection: (id: string) => void;
  setMode: (mode: EditorMode) => void;
  setEditSubMode: (subMode: EditSubMode) => void;
  setActiveTool: (tool: ActiveTool) => void;
  setEditObjectId: (id: string | null) => void;
  selectAllToggle: () => void;

  setSelectedVertices: (vertices: Set<number>) => void;
  setSelectedEdges: (edges: Set<number>) => void;
  setSelectedFaces: (faces: Set<number>) => void;
  toggleVertex: (index: number) => void;
  toggleEdge: (index: number) => void;
  toggleFace: (index: number) => void;
  clearEditSelection: () => void;
}

// Prevent Vite HMR from hot-reloading this module — a hot reload creates a new
// Zustand store instance while SceneManager keeps its subscription on the old
// one, so objects appear in React but never reach Three.js.  A full page reload
// ensures every module shares the same store singleton.
if (import.meta.hot) {
  import.meta.hot.decline();
}

// Remove any stale key written by a previous persist-middleware attempt.
try { localStorage.removeItem('git-blox-editor'); } catch {}

const NPC_CONFIG_KEY = 'git-blox-npc-config';

function loadNpcConfig(): NpcConfig {
  try {
    const raw = localStorage.getItem(NPC_CONFIG_KEY);
    if (raw) {
      const stored = { ...defaultNpcConfig, ...JSON.parse(raw) };
      // Always pin to the current canonical ID so stale localStorage values can't persist.
      stored.nvidiaFunctionId = NVIDIA_A2F_MODELS[0].id;
      return stored;
    }
  } catch {}
  return defaultNpcConfig;
}

const defaultNpcConfig: NpcConfig = {
  openAiKey: '',
  deepgramKey: '',
  elevenLabsKey: '',
  elevenLabsVoiceId: '21m00Tcm4TlvDq8ikWAM',
  nvidiaApiKey: '',
  nvidiaFunctionId: NVIDIA_A2F_MODELS[0].id,
};

export const useEditorStore = create<EditorState>((set, get) => ({
  objects: {},
  selectedIds: [],
  mode: 'object',
  editSubMode: 'vertex',
  activeTool: 'select',
  editObjectId: null,
  selectedVertices: new Set(),
  selectedEdges: new Set(),
  selectedFaces: new Set(),

  enhanceScreenshot: null,
  enhanceResult: null,
  enhanceLoading: false,
  imageLibrary: [],
  setEnhanceScreenshot: (s) => set({ enhanceScreenshot: s }),
  setEnhanceResult: (s) => set({ enhanceResult: s }),
  setEnhanceLoading: (loading) => set({ enhanceLoading: loading }),
  addToImageLibrary: (dataUri) =>
    set((s) => ({ imageLibrary: [dataUri, ...s.imageLibrary] })),

  composerPrompt: '',
  composerLoading: false,
  composerError: null,
  composerRefImage: null,
  setComposerPrompt: (s) => set({ composerPrompt: s }),
  setComposerLoading: (b) => set({ composerLoading: b }),
  setComposerError: (s) => set({ composerError: s }),
  setComposerRefImage: (s) => set({ composerRefImage: s }),

  npcConfig: loadNpcConfig(),
  setNpcConfig: (updates) =>
    set((s) => {
      const next = { ...s.npcConfig, ...updates };
      try { localStorage.setItem(NPC_CONFIG_KEY, JSON.stringify(next)); } catch {}
      return { npcConfig: next };
    }),
  npcVoiceStatus: 'idle',
  setNpcVoiceStatus: (s) => set({ npcVoiceStatus: s }),

  setNpcPersonality: (objectId, personality) =>
    set((s) => {
      if (!s.objects[objectId]) return s;
      return {
        objects: {
          ...s.objects,
          [objectId]: { ...s.objects[objectId], npcPersonality: personality },
        },
      };
    }),

  ensureNpcPersonality: (objectId) => {
    const s = get();
    if (!s.objects[objectId] || s.objects[objectId].npcPersonality) return;
    const obj = s.objects[objectId];
    const personality: NpcPersonality = {
      name: obj.name,
      backstory: '',
      speakingStyle: 'friendly and helpful',
      accentDescription: '',
      vocabularyQuirks: '',
      systemPrompt: `You are ${obj.name}, a character in a 3D world. Stay in character and be engaging. Speak naturally and keep responses concise.`,
    };
    set((prev) => ({
      objects: {
        ...prev.objects,
        [objectId]: { ...prev.objects[objectId], npcPersonality: personality },
      },
    }));
  },

  setGenerationPrompt: (objectId, prompt) =>
    set((s) => {
      if (!s.objects[objectId]) return s;
      return {
        objects: {
          ...s.objects,
          [objectId]: { ...s.objects[objectId], generationPrompt: prompt },
        },
      };
    }),

  peekNextId: () => peekId(),

  addObject: (type, position = [0, 0, 0]) => {
    const id = generateId();
    const baseName = geoNames[type] ?? 'Object';
    const existing = Object.values(get().objects).map((o) => o.name);
    let name = baseName;
    let c = 1;
    while (existing.includes(name)) {
      name = `${baseName}.${String(c).padStart(3, '0')}`;
      c++;
    }
    const obj: SceneObject = {
      id,
      name,
      type: 'mesh',
      geometryType: type,
      position,
      rotation: [0, 0, 0],
      scale: [1, 1, 1],
      color: '#808080',
      children: [],
      parent: null,
      visible: true,
    };
    set((s) => ({
      objects: { ...s.objects, [id]: obj },
      selectedIds: [id],
    }));
    return id;
  },

  addImportedObject: (importName, animations) => {
    const id = generateId();
    const existing = Object.values(get().objects).map((o) => o.name);
    let name = importName;
    let c = 1;
    while (existing.includes(name)) {
      name = `${importName}.${String(c).padStart(3, '0')}`;
      c++;
    }
    const obj: SceneObject = {
      id,
      name,
      type: 'mesh',
      geometryType: 'imported',
      position: [0, 0, 0],
      rotation: [0, 0, 0],
      scale: [1, 1, 1],
      color: '#808080',
      children: [],
      parent: null,
      visible: true,
      animations: animations ?? [],
      activeAnimation: null,
    };
    set((s) => ({
      objects: { ...s.objects, [id]: obj },
      selectedIds: [id],
    }));
    return id;
  },

  setActiveAnimation: (objectId, animationName) =>
    set((s) => {
      if (!s.objects[objectId]) return s;
      return {
        objects: {
          ...s.objects,
          [objectId]: { ...s.objects[objectId], activeAnimation: animationName },
        },
      };
    }),

  setRigTaskId: (objectId, taskId) =>
    set((s) => {
      if (!s.objects[objectId]) return s;
      return {
        objects: {
          ...s.objects,
          [objectId]: { ...s.objects[objectId], rigTaskId: taskId },
        },
      };
    }),

  setMeshyTaskId: (objectId, taskId) =>
    set((s) => {
      if (!s.objects[objectId]) return s;
      return {
        objects: {
          ...s.objects,
          [objectId]: { ...s.objects[objectId], meshyTaskId: taskId },
        },
      };
    }),

  addAnimationsToObject: (objectId, anims) =>
    set((s) => {
      if (!s.objects[objectId]) return s;
      const existing = s.objects[objectId].animations ?? [];
      return {
        objects: {
          ...s.objects,
          [objectId]: {
            ...s.objects[objectId],
            animations: [...existing, ...anims],
          },
        },
      };
    }),

  removeObject: (id) =>
    set((s) => {
      const { [id]: _, ...rest } = s.objects;
      return {
        objects: rest,
        selectedIds: s.selectedIds.filter((x) => x !== id),
      };
    }),

  updateObject: (id, updates) =>
    set((s) => {
      if (!s.objects[id]) return s;
      return {
        objects: { ...s.objects, [id]: { ...s.objects[id], ...updates } },
      };
    }),

  setSelection: (ids) => set({ selectedIds: ids }),
  addToSelection: (id) =>
    set((s) => ({
      selectedIds: s.selectedIds.includes(id)
        ? s.selectedIds
        : [...s.selectedIds, id],
    })),
  removeFromSelection: (id) =>
    set((s) => ({
      selectedIds: s.selectedIds.filter((x) => x !== id),
    })),

  setMode: (mode) => set({ mode }),
  setEditSubMode: (subMode) => set({ editSubMode: subMode }),
  setActiveTool: (tool) => set({ activeTool: tool }),
  setEditObjectId: (id) => set({ editObjectId: id }),

  selectAllToggle: () => {
    const s = get();
    if (s.mode === 'object') {
      const all = Object.keys(s.objects);
      set({ selectedIds: s.selectedIds.length === all.length ? [] : all });
    }
  },

  setSelectedVertices: (v) => set({ selectedVertices: v }),
  setSelectedEdges: (e) => set({ selectedEdges: e }),
  setSelectedFaces: (f) => set({ selectedFaces: f }),

  toggleVertex: (i) =>
    set((s) => {
      const n = new Set(s.selectedVertices);
      if (n.has(i)) n.delete(i);
      else n.add(i);
      return { selectedVertices: n };
    }),
  toggleEdge: (i) =>
    set((s) => {
      const n = new Set(s.selectedEdges);
      if (n.has(i)) n.delete(i);
      else n.add(i);
      return { selectedEdges: n };
    }),
  toggleFace: (i) =>
    set((s) => {
      const n = new Set(s.selectedFaces);
      if (n.has(i)) n.delete(i);
      else n.add(i);
      return { selectedFaces: n };
    }),

  clearEditSelection: () =>
    set({
      selectedVertices: new Set(),
      selectedEdges: new Set(),
      selectedFaces: new Set(),
    }),
}));
