import type { LLM, Message, Tool, LLMResponse, LLMUsageEvent } from '@noetaris/harness-types'
import type { ObserverAware, Observer, StepContext } from '@noetaris/harness'

export class MockOllamaEmptyQueueError extends Error {
  constructor() {
    super('MockOllama has no responses configured — call new MockOllama(response) or enqueue(response) before invoke')
    this.name = 'MockOllamaEmptyQueueError'
  }
}

const ZEROED_STEP_CONTEXT: StepContext = { agentId: '', sessionId: '', stepName: '' }

export class MockOllama implements LLM, ObserverAware {
  lastMessages: Message[] = []

  private queue: LLMResponse[] = []
  private observer: Observer = {}
  private stepContext: StepContext = ZEROED_STEP_CONTEXT

  constructor(responses?: LLMResponse | LLMResponse[]) {
    if (responses !== undefined) {
      this.enqueue(responses)
    }
  }

  enqueue(response: LLMResponse | LLMResponse[]): void {
    const items = Array.isArray(response) ? response : [response]
    this.queue.push(...items)
  }

  bindObserver(observer: Observer): void {
    this.observer = observer
  }

  setStepContext(ctx: StepContext): void {
    this.stepContext = ctx
  }

  async invoke(messages: Message[], options?: { tools?: Tool[] }): Promise<LLMResponse> {
    void options
    if (this.queue.length === 0) {
      throw new MockOllamaEmptyQueueError()
    }

    // sticky-last: dequeue only when more than one element remains
    const response: LLMResponse = this.queue.length > 1
      ? (this.queue.shift() as LLMResponse) // as: shift() on non-empty array is always defined; length > 1 is checked above
      : (this.queue[0] as LLMResponse) // as: queue.length === 1 guaranteed by the empty check; index 0 is always defined

    this.lastMessages = messages

    const event: LLMUsageEvent = {
      tokens:     { input: 0, output: 0 },
      modelId:    'mock',
      stopReason: response.stopReason,
    }
    this.observer.onEvent?.(this.stepContext, 'llm.response', event)

    return response
  }
}
