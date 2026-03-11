import { useEditorStore } from '../../store/editorStore';
import { NVIDIA_A2F_MODELS } from '../../store/types';
import NpcVoiceWidget from '../NpcVoiceWidget';

export default function NpcVoiceConfigPanel() {
  const config = useEditorStore((s) => s.npcConfig);
  const setNpcConfig = useEditorStore((s) => s.setNpcConfig);
  const objects = useEditorStore((s) => s.objects);
  const selectedIds = useEditorStore((s) => s.selectedIds);
  const selectedObj = selectedIds.length > 0 ? objects[selectedIds[0]] : null;
  const personality = selectedObj?.npcPersonality;

  return (
    <div className="npcconfig-panel">
      <div className="npcconfig-section">
        <div className="npcconfig-section-title">API Keys</div>
        <div className="npcconfig-field">
          <label className="npcconfig-label">OpenAI API Key</label>
          <input
            type="password"
            className="npcconfig-input"
            placeholder="sk-..."
            value={config.openAiKey}
            onChange={(e) => setNpcConfig({ openAiKey: e.target.value })}
            autoComplete="off"
          />
        </div>
        <div className="npcconfig-field">
          <label className="npcconfig-label">Deepgram API Key</label>
          <input
            type="password"
            className="npcconfig-input"
            placeholder="..."
            value={config.deepgramKey}
            onChange={(e) => setNpcConfig({ deepgramKey: e.target.value })}
            autoComplete="off"
          />
        </div>
        <div className="npcconfig-field">
          <label className="npcconfig-label">ElevenLabs API Key</label>
          <input
            type="password"
            className="npcconfig-input"
            placeholder="..."
            value={config.elevenLabsKey}
            onChange={(e) => setNpcConfig({ elevenLabsKey: e.target.value })}
            autoComplete="off"
          />
        </div>
        <div className="npcconfig-field">
          <label className="npcconfig-label">ElevenLabs Voice ID</label>
          <input
            type="text"
            className="npcconfig-input"
            placeholder="21m00Tcm4TlvDq8ikWAM"
            value={config.elevenLabsVoiceId}
            onChange={(e) => setNpcConfig({ elevenLabsVoiceId: e.target.value })}
            autoComplete="off"
          />
        </div>
        <div className="npcconfig-field">
          <label className="npcconfig-label">NVIDIA API Key <span className="npcconfig-optional">(A2F blendshapes)</span></label>
          <input
            type="password"
            className="npcconfig-input"
            placeholder="nvapi-..."
            value={config.nvidiaApiKey}
            onChange={(e) => setNpcConfig({ nvidiaApiKey: e.target.value })}
            autoComplete="off"
          />
        </div>
        <div className="npcconfig-field">
          <label className="npcconfig-label">A2F Voice Model <span className="npcconfig-optional">(blendshape character)</span></label>
          <select
            className="npcconfig-input npcconfig-select"
            value={config.nvidiaFunctionId}
            onChange={(e) => setNpcConfig({ nvidiaFunctionId: e.target.value })}
          >
            {NVIDIA_A2F_MODELS.map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="npcconfig-section">
        <div className="npcconfig-section-title">
          NPC System Prompt
          {selectedObj && !personality && (
            <span className="npcconfig-hint"> — select a character with NPC personality</span>
          )}
          {!selectedObj && (
            <span className="npcconfig-hint"> — no character selected</span>
          )}
        </div>
        {personality ? (
          <NpcSystemPromptEditor
            objectId={selectedObj!.id}
            personality={personality}
          />
        ) : (
          <div className="npcconfig-empty">
            Generate a mesh with an NPC concept to auto-populate this field.
          </div>
        )}
      </div>

      <div className="npcconfig-section npcconfig-talk-section">
        <div className="npcconfig-section-title">Voice Session</div>
        {selectedObj && personality ? (
          <NpcVoiceWidget objectId={selectedObj.id} />
        ) : (
          <div className="npcconfig-empty">
            Select an NPC to begin
          </div>
        )}
      </div>
    </div>
  );
}

function NpcSystemPromptEditor({
  objectId,
  personality,
}: {
  objectId: string;
  personality: NonNullable<ReturnType<typeof useEditorStore.getState>['objects'][string]['npcPersonality']>;
}) {
  const setNpcPersonality = useEditorStore((s) => s.setNpcPersonality);

  return (
    <div>
      <div className="npcconfig-field">
        <label className="npcconfig-label">Character Name</label>
        <div className="npcconfig-readonly">{personality.name}</div>
      </div>
      <div className="npcconfig-field">
        <label className="npcconfig-label">Speaking Style</label>
        <div className="npcconfig-readonly">{personality.speakingStyle}</div>
      </div>
      <div className="npcconfig-field">
        <label className="npcconfig-label" style={{ marginBottom: 4 }}>
          System Prompt
          <span className="npcconfig-optional"> (auto-generated — editable)</span>
        </label>
        <textarea
          className="npcconfig-prompt"
          value={personality.systemPrompt}
          onChange={(e) =>
            setNpcPersonality(objectId, { ...personality, systemPrompt: e.target.value })
          }
          rows={10}
        />
      </div>
    </div>
  );
}
