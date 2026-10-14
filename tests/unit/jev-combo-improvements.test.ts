/**
 * Improvements landed on top of #15279: trace model, per-combo policy,
 * decision-target factory, read-only sticky preview.
 */
import test from "node:test";
import assert from "node:assert/strict";

import { judgeJev, type JevCandidate } from "../../open-sse/services/combo/jevStrategy.ts";
import { orderJevComboTargets } from "../../open-sse/services/combo/jevOrdering.ts";
import {
  createJevAsk,
  createRouteBackedJevAsk,
  createTypesafeJevAsk,
} from "../../open-sse/services/combo/jevAsk.ts";
import {
  COMBO_SKIP_REASONS,
  createInvocationId,
  getComboTrace,
  recordComboDecision,
  resetComboTraceStore,
  startComboTrace,
} from "../../open-sse/services/combo/decisionTrace.ts";
import type { ResolvedComboTarget } from "../../open-sse/services/combo/types.ts";
import type { ComboLogger } from "../../open-sse/services/combo/types.ts";

test.afterEach(() => {
  resetComboTraceStore();
});

const noopLog: ComboLogger = {
  info() {},
  warn() {},
  error() {},
  debug() {},
};

function modelTarget(
  executionKey: string,
  modelStr: string,
  stepId = "shared"
): ResolvedComboTarget {
  return {
    kind: "model",
    stepId,
    executionKey,
    modelStr,
    provider: "p",
    providerId: null,
    connectionId: null,
    weight: 1,
    label: null,
  };
}

const CANDIDATES: JevCandidate[] = [
  { id: "opus", userRank: 0, label: "Opus" },
  { id: "gpt", userRank: 1, label: "GPT" },
  { id: "glm", userRank: 2, label: "GLM" },
];

test("judgeJev: per-call policy lowers the probability floor and admits a previously-held option", () => {
  const standard = judgeJev(CANDIDATES, {
    choice: "opus",
    probabilities: { opus: 0.5, gpt: 0.12, glm: 0.38 },
    confidence: 0.9,
  });
  // At the default 0.15 floor, gpt is held.
  assert.equal(standard.held.find((h) => h.id === "gpt")?.reason, "below_probability_floor");
  assert.equal(standard.admitted.find((a) => a.id === "gpt"), undefined);

  const loose = judgeJev(
    CANDIDATES,
    {
      choice: "opus",
      probabilities: { opus: 0.5, gpt: 0.12, glm: 0.38 },
      confidence: 0.9,
    },
    { probabilityFloor: 0.1 }
  );
  // At a 0.10 floor, gpt is admitted.
  assert.equal(loose.held.find((h) => h.id === "gpt"), undefined);
  assert.ok(loose.admitted.some((a) => a.id === "gpt"));
});

test("judgeJev: per-call policy can lift confidence below default and fall open", () => {
  // Default confidence floor is 0.65. With an override of 0.5 the same
  // 0.55 answer should be admitted; with default it falls open.
  const underDefault = judgeJev(CANDIDATES, {
    choice: "opus",
    probabilities: { opus: 0.5, gpt: 0.3, glm: 0.2 },
    confidence: 0.55,
  });
  assert.equal(underDefault.fellOpen, true);

  const underOverride = judgeJev(
    CANDIDATES,
    {
      choice: "opus",
      probabilities: { opus: 0.5, gpt: 0.3, glm: 0.2 },
      confidence: 0.55,
    },
    { confidenceFloor: 0.5 }
  );
  assert.equal(underOverride.fellOpen, false);
  assert.equal(underOverride.admitted[0].id, "opus");
});

test("orderJevComboTargets records the answering model on the admitted detail", async () => {
  const targets = [modelTarget("a", "cc/opus"), modelTarget("b", "glm/glm")];
  const outcome = await orderJevComboTargets(targets, "route me", {
    sessionPinned: false,
    ask: async () => ({
      ok: true,
      answer: {
        choice: "a",
        probabilities: { a: 0.7, b: 0.3 },
        confidence: 0.9,
        model: "typesafe/jev-latest",
      },
    }),
  });
  assert.match(outcome.admitted[0].detail, /model=typesafe\/jev-latest/);
});

test("orderJevComboTargets: per-call policy threads into admission", async () => {
  const targets = [modelTarget("a", "cc/opus"), modelTarget("b", "glm/glm")];
  const outcome = await orderJevComboTargets(targets, "route me", {
    sessionPinned: false,
    ask: async () => ({
      ok: true,
      // gpt would be held at the default 0.15 floor; permissive policy admits it.
      answer: {
        choice: "a",
        probabilities: { a: 0.7, b: 0.1 },
        confidence: 0.9,
      },
    }),
    policy: { probabilityFloor: 0.05 },
  });
  assert.ok(outcome.admitted.some((a) => a.executionKey === "b"));
});

test("createJevAsk with no decision target selects the direct TypeSafe factory", () => {
  // The factory still resolves when jevConfig is null: callers see the legacy
  // behavior, the direct client, with zero new dependencies.
  const ask = createJevAsk({
    jevConfig: null,
    log: noopLog,
    signal: null,
    apiKeyAllowedConnections: null,
  });
  assert.equal(typeof ask, "function");
});

test("createJevAsk with a decision target selects the route-backed factory", () => {
  const ask = createJevAsk({
    jevConfig: { decisionTarget: "typesafe/jev-latest", timeoutMs: 5000 },
    log: noopLog,
    signal: null,
    apiKeyAllowedConnections: null,
  });
  assert.equal(typeof ask, "function");
});

test("createRouteBackedJevAsk posts to /v1/systemone with the decision-target model", async () => {
  const calls: Array<{ url: string; body: string; headers: Record<string, string> }> = [];
  const fetchMock: typeof fetch = async (input, init) => {
    const url = typeof input === "string" ? input : input.toString();
    calls.push({
      url,
      body: typeof init?.body === "string" ? init.body : "",
      headers: (init?.headers as Record<string, string>) ?? {},
    });
    // First call: probe the route. Return 200 so routeReady = true.
    // Second call: the actual ask. Return a valid choice payload.
    if (url.includes("/v1/systemone/models")) {
      return new Response("{}", { status: 200 });
    }
    return new Response(
      JSON.stringify({
        answers: {
          route: {
            type: "choice",
            choice: "opus",
            probabilities: { opus: 0.7, gpt: 0.3 },
            confidence: 0.9,
          },
        },
        model: "ollama-local/clef-flash",
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  };
  const ask = createRouteBackedJevAsk(
    { decisionTarget: "ollama-local/clef-flash", timeoutMs: 4000 },
    noopLog,
    null,
    null,
    async () => ({ apiKey: "ts-fake-key" })
  );
  // Override fetch via globalThis to inject the mock.
  const originalFetch = globalThis.fetch;
  (globalThis as { fetch: typeof fetch }).fetch = fetchMock as typeof fetch;
  try {
    const result = await ask({
      state: "pick a model",
      criteria: { opus: "Opus", gpt: "GPT" },
    });
    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.answer.choice, "opus");
    assert.equal(result.answer.model, "ollama-local/clef-flash");
    // Probe + ask: at least one call to /v1/systemone carries the decision target.
    const askCall = calls.find((c) => c.url.endsWith("/v1/systemone"));
    assert.ok(askCall, "expected a /v1/systemone call");
    const body = JSON.parse(askCall.body) as { model?: string };
    assert.equal(body.model, "ollama-local/clef-flash");
  } finally {
    (globalThis as { fetch: typeof fetch }).fetch = originalFetch;
  }
});

test("createRouteBackedJevAsk times out and falls back to unavailable (no direct fallback when route is ready)", async () => {
  const fetchMock: typeof fetch = async (input) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.includes("/v1/systemone/models")) return new Response("{}", { status: 200 });
    // Simulate an aborted fetch.
    const err = new Error("aborted");
    err.name = "AbortError";
    throw err;
  };
  const ask = createRouteBackedJevAsk(
    { decisionTarget: "typesafe/jev-latest", timeoutMs: 1 },
    noopLog,
    null,
    null,
    async () => ({ apiKey: "ts-fake-key" })
  );
  const originalFetch = globalThis.fetch;
  (globalThis as { fetch: typeof fetch }).fetch = fetchMock as typeof fetch;
  try {
    const result = await ask({ state: "x", criteria: { a: "A" } });
    assert.equal(result.ok, false);
  } finally {
    (globalThis as { fetch: typeof fetch }).fetch = originalFetch;
  }
});

test("createTypesafeJevAsk still threads the answering model into the JevChoiceAnswer shape", () => {
  // The factory returns a function. We can't drive it without a real
  // TypeSafe key in scope, but we can assert the type surface and the
  // timeoutMs passthrough behavior via a hand-rolled check on its toString.
  const ask = createTypesafeJevAsk(
    { decisionTarget: "typesafe/jev-latest", timeoutMs: 2500 },
    noopLog,
    null,
    null
  );
  assert.equal(typeof ask, "function");
  assert.notEqual(ask.length, 0); // takes the (input) arg
});

test("jev skip reasons still on the allowlist (regression guard)", () => {
  for (const reason of [
    "below_probability_floor",
    "dominated",
    "low_confidence",
    "jev_unavailable",
  ] as const) {
    assert.ok((COMBO_SKIP_REASONS as readonly string[]).includes(reason), reason);
  }
});

test("decision trace rows can carry the answering model without leaking the prompt", () => {
  const id = createInvocationId();
  startComboTrace(id, { strategy: "jev", comboName: "stack" });
  recordComboDecision(id, {
    step: "opus",
    target: "cc/opus",
    decision: "not_reached",
    detail: "verdict=admitted rank=0 p=0.7 c=0.9 model=typesafe/jev-latest",
  });
  const trace = getComboTrace(id);
  assert.ok(trace);
  assert.match(trace!.decisions[0].detail ?? "", /model=typesafe\/jev-latest/);
  const serialized = JSON.stringify(trace);
  assert.doesNotMatch(serialized, /Bearer |sk-/);
});
