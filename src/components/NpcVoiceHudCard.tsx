import { useEditorStore } from '../store/editorStore';

const statusLabel: Record<string, string> = {
  idle: 'Idle',
  connecting: 'Connecting…',
  listening: 'Listening…',
  responding: 'Responding…',
  error: 'Error',
};

/**
 * Compact overlay anchored to the top-right of Scene View.
 * Shows the selected NPC name, persona summary, and current voice status.
 */
export default function NpcVoiceHudCard() {
  const objects = useEditorStore((s) => s.objects);
  const selectedIds = useEditorStore((s) => s.selectedIds);
  const npcVoiceStatus = useEditorStore((s) => s.npcVoiceStatus);
  const selectedObj = selectedIds.length > 0 ? objects[selectedIds[0]] : null;
  const personality = selectedObj?.npcPersonality;

  if (!personality) return null;

  return (
    <div className="npc-voice-hud-card">
      <div className="npc-hud-name">{personality.name}</div>
      <div className="npc-hud-persona">{personality.speakingStyle}</div>
      <div className={`npc-hud-status npc-hud-status--${npcVoiceStatus}`}>
        {statusLabel[npcVoiceStatus] ?? 'Idle'}
      </div>
    </div>
  );
}
