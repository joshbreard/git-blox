import { useEditorStore } from '../../store/editorStore';
import { NVIDIA_A2F_MODELS } from '../../store/types';
import type { NpcPersonality } from '../../store/types';

export default function NpcVoiceConfigPanel() {
  const config = useEditorStore((s) => s.npcConfig);
  const setNpcConfig = useEditorStore((s) => s.setNpcConfig);
  const objects = useEditorStore((s) => s.objects);
  const selectedIds = useEditorStore((s) => s.selectedIds);
  const ensureNpcPersonality = useEditorStore((s) => s.ensureNpcPersonality);
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
          NPC Personality
          {!selectedObj && (
            <span className="npcconfig-hint"> — no character selected</span>
          )}
        </div>
        {personality ? (
          <NpcPersonalityEditor
            objectId={selectedObj!.id}
            personality={personality}
          />
        ) : selectedObj ? (
          <div className="npcconfig-empty">
            <p style={{ marginBottom: 10 }}>
              This character has no NPC personality yet. Initialize one to enable voice conversations.
            </p>
            <button
              className="npcconfig-init-btn"
              onClick={() => ensureNpcPersonality(selectedObj.id)}
            >
              Initialize NPC Personality
            </button>
          </div>
        ) : (
          <div className="npcconfig-empty">
            Select any character in the scene to configure its NPC personality. Works with both generated and imported meshes.
          </div>
        )}
      </div>

    </div>
  );
}

function NpcPersonalityEditor({
  objectId,
  personality,
}: {
  objectId: string;
  personality: NpcPersonality;
}) {
  const setNpcPersonality = useEditorStore((s) => s.setNpcPersonality);
  const update = (patch: Partial<NpcPersonality>) =>
    setNpcPersonality(objectId, { ...personality, ...patch });

  return (
    <div>
      <div className="npcconfig-field">
        <label className="npcconfig-label">Character Name</label>
        <input
          type="text"
          className="npcconfig-input"
          value={personality.name}
          onChange={(e) => update({ name: e.target.value })}
        />
      </div>
      <div className="npcconfig-field">
        <label className="npcconfig-label">Backstory / Bio</label>
        <input
          type="text"
          className="npcconfig-input"
          placeholder="Short description shown in the bio card"
          value={personality.backstory}
          onChange={(e) => update({ backstory: e.target.value })}
        />
      </div>
      <div className="npcconfig-field">
        <label className="npcconfig-label">Speaking Style</label>
        <input
          type="text"
          className="npcconfig-input"
          placeholder="e.g. formal, gruff, cheerful"
          value={personality.speakingStyle}
          onChange={(e) => update({ speakingStyle: e.target.value })}
        />
      </div>
      <div className="npcconfig-field">
        <label className="npcconfig-label">
          System Prompt
          <span className="npcconfig-optional"> (sent to LLM — editable)</span>
        </label>
        <textarea
          className="npcconfig-prompt"
          value={personality.systemPrompt}
          onChange={(e) => update({ systemPrompt: e.target.value })}
          rows={10}
        />
      </div>
    </div>
  );
}
