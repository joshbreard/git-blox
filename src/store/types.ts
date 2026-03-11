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
  { label: 'Mark',  id: '8efc55f5-6f00-424e-afe9-26212cd2c630' },
  { label: 'Claire', id: '0961a6da-fb9e-4f2e-8491-247e5fd7bf8d' },
  { label: 'James',  id: '9327c39f-a361-4e02-bd72-e11b4c9b7b5e' },
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
