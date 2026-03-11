import { useRef } from 'react';
import { Layout, Model, type IJsonModel, type TabNode } from 'flexlayout-react';
import 'flexlayout-react/style/dark.css';
import SceneViewPanel from './panels/SceneViewPanel';
import HierarchyPanel from './panels/HierarchyPanel';
import InspectorPanel from './panels/InspectorPanel';
import AnimationsPanel from './panels/AnimationsPanel';
import MeshGenPanel from './panels/MeshGenPanel';
import EnhancePanel from './panels/EnhancePanel';
import SceneComposerPanel from './panels/SceneComposerPanel';
import NpcVoiceConfigPanel from './panels/NpcVoiceConfigPanel';
import NpcBioCard from './NpcBioCard';
import NpcVoiceWidget from './NpcVoiceWidget';
import { useEditorStore } from '../store/editorStore';

const layoutJson: IJsonModel = {
  global: {
    tabEnableClose: false,
    tabEnableRename: false,
    tabSetEnableMaximize: true,
    splitterSize: 4,
  },
  borders: [],
  layout: {
    type: 'row',
    weight: 100,
    children: [
      {
        type: 'row',
        weight: 18,
        children: [
          {
            type: 'tabset',
            weight: 50,
            children: [
              { type: 'tab', name: 'Hierarchy', component: 'hierarchy' },
            ],
          },
          {
            type: 'tabset',
            weight: 50,
            children: [
              { type: 'tab', name: 'Composer', component: 'composer' },
              { type: 'tab', name: 'Mesh Gen', component: 'meshGen' },
            ],
          },
        ],
      },
      {
        // Nested row → FlexLayout alternates direction, so this is vertical
        type: 'row',
        weight: 58,
        children: [
          {
            type: 'tabset',
            weight: 75,
            children: [
              { type: 'tab', name: 'Scene View', component: 'sceneView' },
            ],
          },
          {
            type: 'tabset',
            weight: 25,
            children: [
              { type: 'tab', name: 'Animation Gen', component: 'animations' },
              { type: 'tab', name: 'NPC Voice Config', component: 'npcVoiceConfig' },
            ],
          },
        ],
      },
      {
        type: 'tabset',
        weight: 24,
        children: [
          { type: 'tab', name: 'Inspector', component: 'inspector' },
          { type: 'tab', name: 'Image Gen', component: 'enhance' },
        ],
      },
    ],
  },
};

/** Single global "Talk to NPC" button — only rendered when an NPC with a personality is selected. */
function GlobalTalkButton() {
  const objects = useEditorStore((s) => s.objects);
  const selectedIds = useEditorStore((s) => s.selectedIds);
  const selectedObj = selectedIds.length > 0 ? objects[selectedIds[0]] : null;
  if (!selectedObj?.npcPersonality) return null;
  return (
    <div className="global-talk-button">
      <NpcVoiceWidget objectId={selectedObj.id} />
    </div>
  );
}

export default function EditorLayout() {
  const modelRef = useRef(Model.fromJson(layoutJson));

  function factory(node: TabNode) {
    switch (node.getComponent()) {
      case 'sceneView':
        return <SceneViewPanel />;
      case 'hierarchy':
        return <HierarchyPanel />;
      case 'inspector':
        return <InspectorPanel />;
      case 'animations':
        return <AnimationsPanel />;
      case 'meshGen':
        return <MeshGenPanel />;
      case 'enhance':
        return <EnhancePanel />;
      case 'composer':
        return <SceneComposerPanel />;
      case 'npcVoiceConfig':
        return <NpcVoiceConfigPanel />;
      default:
        return null;
    }
  }

  return (
    <div className="app-layout">
      <Layout model={modelRef.current} factory={factory} />
      <NpcBioCard />
      <GlobalTalkButton />
    </div>
  );
}
