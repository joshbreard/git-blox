/**
 * OpenAI API utilities — called directly from the browser using the user's key.
 */

import type { NpcPersonality } from '../store/types';

export async function generateNpcPersonality(
  characterPrompt: string,
  apiKey: string,
): Promise<NpcPersonality> {
  const systemMsg = `You are a game-design AI that generates vivid NPC personality profiles.
Return ONLY a valid JSON object (no markdown, no code fences) with these exact keys:
  name           — a fitting name for this character
  backstory      — 2-3 sentences of personal history
  speakingStyle  — one sentence describing cadence, tone, formality
  accentDescription — one sentence describing any accent or dialect
  vocabularyQuirks  — one sentence listing unusual words/phrases this character uses
  systemPrompt   — a first-person instruction paragraph (200-300 words) for an LLM to role-play as this character during live voice chat. Begin with "You are [name]."`;

  const userMsg = `Generate an NPC personality for: "${characterPrompt}"`;

  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: 'gpt-4o',
      temperature: 0.9,
      messages: [
        { role: 'system', content: systemMsg },
        { role: 'user', content: userMsg },
      ],
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`OpenAI API error ${res.status}: ${text}`);
  }

  const data = (await res.json()) as {
    choices: Array<{ message: { content: string } }>;
  };

  const raw = data.choices[0]?.message?.content ?? '{}';
  try {
    const parsed = JSON.parse(raw) as NpcPersonality;
    if (!parsed.name || !parsed.systemPrompt) throw new Error('Missing required fields');
    return parsed;
  } catch {
    throw new Error(`Failed to parse NPC personality JSON: ${raw.slice(0, 200)}`);
  }
}
