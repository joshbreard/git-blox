import { useEditorStore } from '../store/editorStore';

/**
 * Fixed top-left panel — visible whenever the selected object has an NPC personality.
 * Shows bio info only; the Talk button lives in the global GlobalTalkButton in Layout.
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
      </div>
      <div className="npc-bio-body">
        {personality.backstory && (
          <p className="npc-bio-backstory">{personality.backstory}</p>
        )}
        {personality.speakingStyle && (
          <div className="npc-bio-tags">
            <span className="npc-bio-tag">{personality.speakingStyle}</span>
          </div>
        )}
      </div>
    </div>
  );
}
