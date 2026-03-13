export type Vec3 = [number, number, number];

export interface NpcPersonality {
  name: string;
  backstory: string;
  speakingStyle: string;
  accentDescription: string;
  vocabularyQuirks: string;
  /** First-person system prompt used during voice conversation */
  systemPrompt: string;
}

export interface NpcConfig {
  openAiKey: string;
  deepgramKey: string;
  elevenLabsKey: string;
  elevenLabsVoiceId: string;
  nvidiaApiKey: string;
  /** NVIDIA Audio2Face-3D NIM function ID (selects Mark / Claire / James voice model) */
  nvidiaFunctionId: string;
}

export const NVIDIA_A2F_MODELS = [
  { label: 'Mark (tongue enabled)', id: '8efc55f5-6f00-424e-afe9-26212cd2c630' },
  { label: 'audio2face-2b', id: '617f80a7-85e4-4bf0-9dd6-dcb61e886142' },
] as const;

export type GeometryType =
  | 'box'
  | 'sphere'
  | 'cylinder'
  | 'cone'
  | 'torus'
  | 'plane'
  | 'icosahedron'
  | 'imported';

export type EditorMode = 'object' | 'edit';
export type EditSubMode = 'vertex' | 'edge' | 'face';
export type ActiveTool = 'select' | 'move' | 'rotate' | 'scale' | 'extrude';

export interface AnimationData {
  name: string;
  duration: number;
}

export interface SceneObject {
  id: string;
  name: string;
  type: 'mesh' | 'group';
  geometryType: GeometryType;
  position: Vec3;
  rotation: Vec3;
  scale: Vec3;
  color: string;
  children: string[];
  parent: string | null;
  visible: boolean;
  animations?: AnimationData[];
  activeAnimation?: string | null;
  rigTaskId?: string | null;
  meshyTaskId?: string | null;
  /** Text prompt used when generating this character (e.g. "a medieval farmer") */
  generationPrompt?: string | null;
  /** NPC personality auto-generated from the generation prompt */
  npcPersonality?: NpcPersonality | null;
}
