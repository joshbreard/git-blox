import { useEditorStore } from '../store/editorStore';
import NpcVoiceWidget from './NpcVoiceWidget';

/**
 * Fixed top-right panel — visible whenever the selected object has an NPC personality.
 */
export default function NpcBioCard() {
  const objects = useEditorStore((s) => s.objects);
  const selectedIds = useEditorStore((s) => s.selectedIds);
  const selectedObj = selectedIds.length > 0 ? objects[selectedIds[0]] : null;
  const personality = selectedObj?.npcPersonality;

  if (!personality) return null;

  return (
    <div className="npc-bio-card">
      <div className="npc-bio-header">
        <span className="npc-bio-name">{personality.name}</span>
        <NpcVoiceWidget objectId={selectedObj!.id} />
      </div>
      <div className="npc-bio-body">
        <p className="npc-bio-backstory">{personality.backstory}</p>
        <div className="npc-bio-tags">
          <span className="npc-bio-tag">{personality.speakingStyle}</span>
        </div>
      </div>
    </div>
  );
}
