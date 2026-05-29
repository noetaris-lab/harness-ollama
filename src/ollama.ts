import type { LLM, Message, Tool, ToolCall, LLMResponse, LLMUsageEvent } from '@noetaris/harness-types'
import type { ObserverAware, Observer, StepContext } from '@noetaris/harness'
import { randomUUID } from 'node:crypto'

/** Options for {@link Ollama}. */
export interface OllamaOptions {
  /** Base URL of the Ollama server. Defaults to `http://localhost:11434`. */
  baseUrl?: string
}

/**
 * Thrown by {@link Ollama.invoke} when the Ollama REST API returns a non-2xx
 * HTTP status code.
 */
export class OllamaApiError extends Error {
  /** The HTTP status code returned by the Ollama server. */
  readonly status: number

  constructor(status: number, body: string) {
    super(`Ollama API error: HTTP ${status} — ${body}`)
    this.name = 'OllamaApiError'
    this.status = status
  }
}

type OllamaToolCall = {
  function: {
    name: string
    arguments: Record<string, unknown>
  }
}

type OllamaMessage =
  | { role: 'user'; content: string }
  | { role: 'assistant'; content: string; tool_calls?: OllamaToolCall[] }
  | { role: 'tool'; content: string }

type OllamaTool = {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: Record<string, unknown>
  }
}

type OllamaResponse = {
  model: string
  message: {
    role: 'assistant'
    content: string
    tool_calls?: OllamaToolCall[]
  }
  done: boolean
  done_reason?: string
  prompt_eval_count?: number
  eval_count?: number
}

function translateMessages(messages: Message[]): OllamaMessage[] {
  return messages.map((msg): OllamaMessage => {
    if (msg.role === 'user') {
      return { role: 'user', content: msg.content }
    }
    if (msg.role === 'assistant') {
      if (msg.toolCalls && msg.toolCalls.length > 0) {
        const tool_calls: OllamaToolCall[] = msg.toolCalls.map((tc) => ({
          function: {
            name: tc.name,
            // as: ToolCall.input typed as unknown per harness-types contract; Ollama requires Record<string,unknown>
            arguments: tc.input as Record<string, unknown>,
          },
        }))
        return { role: 'assistant', content: msg.content ?? '', tool_calls }
      }
      return { role: 'assistant', content: msg.content ?? '' }
    }
    // role === 'tool'
    return { role: 'tool', content: msg.content }
  })
}

function translateTools(tools: Tool[]): OllamaTool[] {
  return tools.map((t) => ({
    type: 'function',
    function: {
      name: t.name,
      description: t.description,
      parameters: t.inputSchema,
    },
  }))
}

function mapStopReason(toolCalls: OllamaToolCall[], doneReason: string | undefined): LLMResponse['stopReason'] {
  if (toolCalls.length > 0) return 'tool_use'
  if (doneReason === 'length') return 'max_tokens'
  return 'end'
}

function normalizeResponse(response: OllamaResponse): LLMResponse {
  const rawToolCalls = response.message.tool_calls ?? []
  const toolCalls: ToolCall[] = rawToolCalls.map((tc) => ({
    id: randomUUID(),
    name: tc.function.name,
    input: tc.function.arguments,
  }))

  return {
    text: response.message.content ?? '',
    toolCalls,
    stopReason: mapStopReason(rawToolCalls, response.done_reason),
  }
}

const ZEROED_STEP_CONTEXT: StepContext = { agentId: '', sessionId: '', stepName: '' }
const DEFAULT_BASE_URL = 'http://localhost:11434'

/**
 * {@link LLM} adapter for a locally-running Ollama server (`/api/chat`).
 *
 * Implements {@link ObserverAware} — emits an `'llm.response'` event with an
 * `LLMUsageEvent` payload after each successful invocation.
 *
 * @throws {@link OllamaApiError} when the server returns a non-2xx response.
 *
 * @example
 * ```ts
 * const llm = new Ollama('llama3.2', { baseUrl: 'http://localhost:11434' })
 * const response = await llm.invoke(messages)
 * ```
 */
export class Ollama implements LLM, ObserverAware {
  private readonly model: string
  private readonly baseUrl: string
  private observer: Observer = {}
  private stepContext: StepContext = ZEROED_STEP_CONTEXT

  /**
   * @param model - Ollama model tag, e.g. `'llama3.2'`.
   * @param options - Optional base URL override (defaults to `http://localhost:11434`).
   */
  constructor(model: string, options?: OllamaOptions) {
    this.model = model
    this.baseUrl = options?.baseUrl ?? DEFAULT_BASE_URL
  }

  bindObserver(observer: Observer): void {
    this.observer = observer
  }

  setStepContext(ctx: StepContext): void {
    this.stepContext = ctx
  }

  async invoke(messages: Message[], options?: { tools?: Tool[] }): Promise<LLMResponse> {
    const translatedMessages = translateMessages(messages)
    const tools = options?.tools

    const requestBody = {
      model: this.model,
      messages: translatedMessages,
      stream: false,
      ...(tools !== undefined ? { tools: translateTools(tools) } : {}),
    }

    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    })

    if (!response.ok) {
      const body = await response.text()
      throw new OllamaApiError(response.status, body)
    }

    const data = await response.json() as OllamaResponse
    const result = normalizeResponse(data)

    const event: LLMUsageEvent = {
      tokens: { input: data.prompt_eval_count ?? 0, output: data.eval_count ?? 0 },
      modelId: this.model,
      stopReason: result.stopReason,
      providerName: 'ollama',
    }
    this.observer.onEvent?.(this.stepContext, 'llm.response', event)

    return result
  }
}
