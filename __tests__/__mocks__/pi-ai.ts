// Stub for @earendil-works/pi-ai (and /compat) peer dependency.
//
// Runtime-shape-compatible exports for tests. Records streamOpenAICompletions
// calls and lets tests override clampThinkingLevel via __setClamp, so tests can
// assert how streamMakora forwards the user's thinking selection
// (reasoning → reasoningEffort) and injects chat_template_kwargs via onPayload
// without depending on real pi-ai internals or making HTTP requests.

export interface SimpleStreamOptions {
  apiKey?: string;
  reasoning?: string;
  reasoningEffort?: string;
  maxTokens?: number;
  onPayload?: (params: any, model: any) => any | Promise<any>;
}

export interface AssistantMessageEventStream {
  end: (result?: any) => void;
}

export const __streamCalls: Array<{
  entrypoint: "raw" | "simple";
  model: any;
  context: any;
  options: any;
}> = [];

export function __resetStreamCalls(): void {
  __streamCalls.length = 0;
}

let __clampImpl: (model: any, level: any) => any = (_model, level) => level;

export function __setClamp(fn: (model: any, level: any) => any): void {
  __clampImpl = fn;
}

export function clampThinkingLevel(model: any, level: any): any {
  return __clampImpl(model, level);
}

export function streamOpenAICompletions(
  model: any,
  context: any,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  __streamCalls.push({ entrypoint: "raw", model, context, options });
  return {
    end() {},
  };
}

export function openAICompletionsApi() {
  return {
    streamSimple(model: any, context: any, options?: SimpleStreamOptions): AssistantMessageEventStream {
      const clampedReasoning = options?.reasoning
        ? clampThinkingLevel(model, options.reasoning)
        : undefined;
      const reasoningEffort = clampedReasoning === "off" ? undefined : clampedReasoning;
      const maxTokens = Math.min(options?.maxTokens ?? model.maxTokens, model.contextWindow);
      const { reasoning: _reasoning, ...streamOptions } = options ?? {};
      __streamCalls.push({
        entrypoint: "simple",
        model,
        context,
        options: { ...streamOptions, maxTokens, reasoningEffort },
      });
      return {
        end() {},
      };
    },
  };
}
