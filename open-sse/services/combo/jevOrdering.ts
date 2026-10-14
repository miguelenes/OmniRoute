/**
 * Jev ordering module (#15276).
 *
 * One interface: preference-ordered targets, the latest user text, whether a
 * live session pin already owns the conversation, and an `ask` port. The
 * result is the attempt order plus holds and admitted rows safe to record on
 * the decision trace. Probability floors, the admission cap, fail-open, and
 * the choice-option ceiling stay inside. Callers do not learn them.
 */

import type { ResolvedComboTarget } from "./types.ts";
import {
  buildJevCriteria,
  judgeJev,
  orderTargetsByJevVerdicts,
  type JevCandidate,
  type JevChoiceAnswer,
  type JevHoldReason,
  type JevPolicy,
  type JudgeJevResult,
} from "./jevStrategy.ts";
import { TYPESAFE_MAX_CHOICE_OPTIONS, TYPESAFE_MAX_STATE_CHARS } from "../typesafe/systemOne.ts";

export type JevAskResult =
  { ok: true; answer: JevChoiceAnswer } | { ok: false; reason: "missing_api_key" | "unavailable" };

export type JevAsk = (input: {
  state: string;
  criteria: Record<string, string | null>;
}) => Promise<JevAskResult>;

type JevAnswerResolution = { answer: JevChoiceAnswer | null; missingApiKey: boolean };

async function requestJevAnswer(
  candidates: readonly JevCandidate[],
  state: string,
  ask: JevAsk
): Promise<JevAnswerResolution> {
  const uniqueIds = new Set(candidates.map((candidate) => candidate.id));
  if (uniqueIds.size !== candidates.length || candidates.length > TYPESAFE_MAX_CHOICE_OPTIONS) {
    return { answer: null, missingApiKey: false };
  }

  const boundedState =
    state.length > TYPESAFE_MAX_STATE_CHARS ? state.slice(0, TYPESAFE_MAX_STATE_CHARS) : state;
  try {
    const result = await ask({ state: boundedState, criteria: buildJevCriteria(candidates) });
    if (result.ok) return { answer: result.answer, missingApiKey: false };
    const unavailable = result as Extract<JevAskResult, { ok: false }>;
    return { answer: null, missingApiKey: unavailable.reason === "missing_api_key" };
  } catch {
    return { answer: null, missingApiKey: false };
  }
}

/** Per-combo overrides for the judge. Absent fields fall back to module constants. */
export type JevOrderOptions = {
  sessionPinned: boolean;
  ask: JevAsk;
  policy?: JevPolicy;
};

export type JevHoldRecord = {
  executionKey: string;
  modelStr: string;
  userRank: number;
  reason: JevHoldReason;
  detail?: string;
};

export type JevAdmittedRecord = {
  executionKey: string;
  modelStr: string;
  userRank: number;
  probability: number | null;
  detail: string;
};

export type JevOrderOutcome = {
  targets: ResolvedComboTarget[];
  holds: JevHoldRecord[];
  admitted: JevAdmittedRecord[];
  /**
   * True when this request's head is a Jev verdict or a fail-open preference
   * head and later stages must not replace it. False when a session pin owns
   * the request — stickiness promotes that pin afterwards.
   */
  protectHead: boolean;
  fellOpen: boolean;
  fallbackReason: "low_confidence" | "jev_unavailable" | null;
  /** True when the ask port reported a missing TypeSafe key. */
  missingApiKey: boolean;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function toTextContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (!isRecord(part)) return "";
        if (typeof part.text === "string") return part.text;
        if (typeof part.content === "string") return part.content;
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

/**
 * Latest user text only — never the tool transcript or system prompts.
 * Chat Completions: last `messages` entry with role user.
 * Responses API: last `input` item with role user (not a join of the whole array).
 *
 * Privacy: this is the only state forwarded to the System One judge. The
 * narrow scope is what makes a local decision backend (Ollama Clef / Clef Flash,
 * Nimble, Tev) safe to run on a host that should not see the full transcript.
 */
export function extractPromptForJevState(body: Record<string, unknown> | null | undefined): string {
  if (!body || typeof body !== "object") return "";
  const fromMessages = latestUserMessageText(body.messages);
  if (fromMessages !== null) return fromMessages;
  if (typeof body.input === "string") return body.input;
  const fromInput = latestResponsesUserInput(body.input);
  if (fromInput !== null) return fromInput;
  if (typeof body.prompt === "string") return body.prompt;
  return "";
}

function latestUserMessageText(messages: unknown): string | null {
  if (!Array.isArray(messages)) return null;
  const message = [...messages].reverse().find((item) => isRecord(item) && item.role === "user");
  return isRecord(message) ? toTextContent(message.content) : null;
}

function latestResponsesUserInput(input: unknown): string | null {
  if (!Array.isArray(input)) return null;
  for (const item of [...input].reverse()) {
    if (!isRecord(item) || item.role !== "user") continue;
    if (typeof item.content === "string" || Array.isArray(item.content)) {
      return toTextContent(item.content);
    }
    if (typeof item.text === "string") return item.text;
  }
  return null;
}

export async function orderJevComboTargets(
  targets: readonly ResolvedComboTarget[],
  state: string,
  options: JevOrderOptions
): Promise<JevOrderOutcome> {
  const ports = options;
  if (ports.sessionPinned || targets.length <= 1) {
    return {
      targets: [...targets],
      holds: [],
      admitted: [],
      protectHead: false,
      fellOpen: false,
      fallbackReason: null,
      missingApiKey: false,
    };
  }

  const candidates = targets.map((target, index) => ({
    id: target.executionKey,
    userRank: index,
    label: target.label || target.modelStr,
  }));
  const { answer, missingApiKey } = await requestJevAnswer(candidates, state, ports.ask);

  return toOutcome(
    targets,
    judgeJev(candidates, answer, options?.policy ?? {}),
    missingApiKey,
    answer?.confidence ?? null,
    answer?.model ?? null
  );
}

function formatAdmittedDetail(
  userRank: number,
  probability: number | null,
  confidence: number | null,
  fellOpen: boolean,
  fallbackReason: JudgeJevResult["fallbackReason"],
  missingApiKey: boolean,
  model: string | null
): string {
  const parts = [`verdict=admitted`, `rank=${userRank}`];
  if (probability !== null && probability !== undefined) parts.push(`p=${probability}`);
  if (confidence !== null && confidence !== undefined && !fellOpen) {
    parts.push(`c=${confidence}`);
  }
  if (model) parts.push(`model=${model}`);
  if (fellOpen && fallbackReason) parts.push(`fallback=${fallbackReason}`);
  if (missingApiKey) parts.push("Jev is not configured");
  return parts.join(" ");
}

function formatHoldDetail(
  userRank: number,
  probability: number | null,
  reason: JevHoldReason
): string {
  const parts = [`verdict=held`, `rank=${userRank}`, `reason=${reason}`];
  if (probability !== null && probability !== undefined) parts.push(`p=${probability}`);
  return parts.join(" ");
}

function toOutcome(
  targets: readonly ResolvedComboTarget[],
  judged: JudgeJevResult,
  missingApiKey: boolean,
  confidence: number | null,
  answeringModel: string | null
): JevOrderOutcome {
  const byId = new Map(targets.map((target) => [target.executionKey, target]));
  const holds: JevHoldRecord[] = [];
  for (const row of judged.held) {
    const target = byId.get(row.id);
    if (!target || !row.reason) continue;
    holds.push({
      executionKey: target.executionKey,
      modelStr: target.modelStr,
      userRank: row.userRank,
      reason: row.reason,
      detail: formatHoldDetail(row.userRank, row.probability, row.reason),
    });
  }
  const admitted: JevAdmittedRecord[] = [];
  for (const row of judged.admitted) {
    const target = byId.get(row.id);
    if (!target) continue;
    admitted.push({
      executionKey: target.executionKey,
      modelStr: target.modelStr,
      userRank: row.userRank,
      probability: row.probability,
      detail: formatAdmittedDetail(
        row.userRank,
        row.probability,
        confidence,
        judged.fellOpen,
        judged.fallbackReason,
        missingApiKey,
        answeringModel
      ),
    });
  }
  return {
    targets: orderTargetsByJevVerdicts(targets, judged.admitted),
    holds,
    admitted,
    protectHead: true,
    fellOpen: judged.fellOpen,
    fallbackReason: judged.fallbackReason,
    missingApiKey,
  };
}
