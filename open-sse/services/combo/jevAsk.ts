/**
 * Ask factories for the `jev` combo strategy (#15276).
 *
 * Two implementations of the same `JevAsk` port:
 *
 * - `createTypesafeJevAsk`: the original direct client to api.typesafe.ai.
 * - `createRouteBackedJevAsk`: routes the decision call through OmniRoute's
 *   own `POST /v1/systemone` endpoint, so any registered System One backend
 *   (TypeSafe, OpenRouter, Ollama Clef / Clef Flash, Nimble, Tev) can serve
 *   the judge. The route itself is owned by #14667; when it is missing this
 *   factory falls back to the direct TypeSafe client transparently.
 *
 * `createJevAsk` chooses between them based on `config.jev.decisionTarget`.
 * When no decision target is configured, the direct TypeSafe client is used
 * (zero behavior change for the existing PR).
 */

import { evaluateSystemOneChoice, TYPESAFE_PROVIDER_ID } from "../typesafe/systemOne.ts";
import { getProviderCredentials } from "../../../src/sse/services/auth";
import type { JevAsk, JevAskResult } from "./jevOrdering.ts";
import type { ComboLogger } from "./types.ts";

export type JevComboConfig = {
  /** System One model id (or decision-combo name) to route the judge call to. */
  decisionTarget?: string;
  /** Probability floor below which a non-winner is held. */
  probabilityFloor?: number;
  /** Confidence floor below which the judge falls open. */
  confidenceFloor?: number;
  /** Hard ceiling on the System One call. */
  timeoutMs?: number;
  /** Cap on the admitted set. */
  maxAdmitted?: number;
};

export type CreateJevAskDeps = {
  jevConfig: JevComboConfig | null;
  log: ComboLogger;
  signal: AbortSignal | null;
  apiKeyAllowedConnections: string[] | null;
  /**
   * Optional override for the TypeSafe credential resolver. Defaults to
   * `getProviderCredentials`; tests inject a fake. The resolver returns
   * `null` when the operator has no TypeSafe key, which falls open.
   */
  credentialsResolver?: () => Promise<{ apiKey: string | null } | null>;
};

/**
 * Direct TypeSafe System One client. Honours the gateway connection
 * allowlist via `getProviderCredentials`'s third argument, so a restricted
 * key whose allowlist excludes TypeSafe cannot route the user's latest
 * text through that operator credential.
 */
export function createTypesafeJevAsk(
  jevConfig: JevComboConfig | null,
  log: ComboLogger,
  signal: AbortSignal | null,
  apiKeyAllowedConnections: string[] | null,
  credentialsResolver?: () => Promise<{ apiKey: string | null } | null>
): JevAsk {
  return async ({ state, criteria }) => {
    try {
      const credentials = await (credentialsResolver ?? (() =>
        getProviderCredentials(
          TYPESAFE_PROVIDER_ID,
          null,
          apiKeyAllowedConnections
        )))();
      const apiKey =
        credentials && typeof (credentials as { apiKey?: unknown }).apiKey === "string"
          ? (credentials as { apiKey: string }).apiKey
          : null;
      const result = await evaluateSystemOneChoice({
        apiKey,
        state,
        criteria,
        signal,
        timeoutMs: jevConfig?.timeoutMs,
      });
      if (result.ok) {
        return {
          ok: true as const,
          answer: {
            choice: result.choice,
            probabilities: result.probabilities,
            confidence: result.confidence,
            model: result.model,
          },
        };
      }
      log.info("COMBO", `[jev] System One unavailable: ${result.reason}`);
      if (result.reason === "missing_api_key") {
        return { ok: false as const, reason: "missing_api_key" as const };
      }
      return { ok: false as const, reason: "unavailable" as const };
    } catch {
      log.warn("COMBO", "[jev] System One call failed (non-fatal); falling open to user order");
      return { ok: false as const, reason: "unavailable" as const };
    }
  };
}

/**
 * Route the decision call through OmniRoute's own `POST /v1/systemone`
 * endpoint. The factory detects the route's availability once per process
 * (in-memory `routeReady`); on a 404 the direct TypeSafe client takes over
 * without disrupting the caller. The local-key path uses a TypeSafe key
 * scoped to the gateway allowlist, mirroring the direct factory.
 */
export function createRouteBackedJevAsk(
  jevConfig: JevComboConfig | null,
  log: ComboLogger,
  signal: AbortSignal | null,
  apiKeyAllowedConnections: string[] | null,
  credentialsResolver?: () => Promise<{ apiKey: string | null } | null>
): JevAsk {
  // Origin defaults to localhost on the dev server port. Overridable via
  // OMNIROUTE_BASE_URL for self-hosted or remote instances.
  const baseUrl = (process.env.OMNIROUTE_BASE_URL || "http://localhost:20128").replace(/\/+$/, "");
  const fetchImpl: typeof fetch = (input, init) => fetch(input, init);
  let routeReady: boolean | null = null;

  async function probeRoute(): Promise<boolean> {
    if (routeReady !== null) return routeReady;
    try {
      const probe = await fetchImpl(`${baseUrl}/v1/systemone/models`, { method: "GET" });
      routeReady = probe.status === 200;
    } catch {
      routeReady = false;
    }
    return routeReady;
  }

  const resolveCreds =
    credentialsResolver ?? (() =>
      getProviderCredentials(TYPESAFE_PROVIDER_ID, null, apiKeyAllowedConnections));
  return async ({ state, criteria }) => {
    if (!(await probeRoute())) {
      return createTypesafeJevAsk(
        jevConfig,
        log,
        signal,
        apiKeyAllowedConnections,
        credentialsResolver
      )({ state, criteria });
    }
    try {
      const credentials = await resolveCreds();
      const apiKey =
        credentials && typeof (credentials as { apiKey?: unknown }).apiKey === "string"
          ? (credentials as { apiKey: string }).apiKey
          : null;
      if (!apiKey) {
        return { ok: false as const, reason: "missing_api_key" as const };
      }
      const model = jevConfig?.decisionTarget;
      const response = await fetchImpl(`${baseUrl}/v1/systemone`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          state,
          questions: {
            route: { type: "choice", criteria },
          },
        }),
        signal: signal ?? undefined,
      });
      if (response.status === 404) {
        // Route was deleted between probe and call; degrade to direct and remember.
        routeReady = false;
        return createTypesafeJevAsk(
          jevConfig,
          log,
          signal,
          apiKeyAllowedConnections,
          credentialsResolver
        )({ state, criteria });
      }
      if (!response.ok) {
        log.info("COMBO", `[jev] /v1/systemone returned HTTP ${response.status}`);
        return { ok: false as const, reason: "unavailable" as const };
      }
      const json = (await response.json()) as {
        answers?: { route?: { choice?: string; probabilities?: Record<string, number>; confidence?: number } };
        model?: string;
      };
      const route = json.answers?.route;
      if (
        !route ||
        typeof route.choice !== "string" ||
        !route.probabilities ||
        typeof route.confidence !== "number"
      ) {
        return { ok: false as const, reason: "unavailable" as const };
      }
      return {
        ok: true as const,
        answer: {
          choice: route.choice,
          probabilities: route.probabilities,
          confidence: route.confidence,
          model: typeof json.model === "string" ? json.model : null,
        },
      };
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        log.info("COMBO", "[jev] /v1/systemone aborted");
        return { ok: false as const, reason: "unavailable" as const };
      }
      log.warn("COMBO", "[jev] /v1/systemone call failed (non-fatal); falling open to user order");
      return { ok: false as const, reason: "unavailable" as const };
    }
  };
}

/**
 * Pick the right factory for `jevConfig.decisionTarget`. Absent means the
 * default behavior (direct TypeSafe client), preserving the existing PR.
 */
export function createJevAsk(deps: CreateJevAskDeps): JevAsk {
  if (deps.jevConfig?.decisionTarget) {
    return createRouteBackedJevAsk(
      deps.jevConfig,
      deps.log,
      deps.signal,
      deps.apiKeyAllowedConnections,
      deps.credentialsResolver
    );
  }
  return createTypesafeJevAsk(
    deps.jevConfig,
    deps.log,
    deps.signal,
    deps.apiKeyAllowedConnections,
    deps.credentialsResolver
  );
}

// Re-export for callers that want a narrow surface.
export type { JevAsk, JevAskResult };
