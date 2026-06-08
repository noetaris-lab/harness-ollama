import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { Message } from '@noetaris/harness-types'
import { Ollama } from './ollama.js'
import { MockOllama } from './mock-ollama.js'

const messages: Message[] = [{ role: 'user', content: 'Hello' }]

function makeOllamaFetch(overrides?: Partial<{
  content: string
  prompt_eval_count: number
  eval_count: number
}>) {
  return vi.fn().mockResolvedValue({
    ok: true,
    json: vi.fn().mockResolvedValue({
      model: 'llama3.2',
      message: { role: 'assistant', content: overrides?.content ?? 'Hi', tool_calls: [] },
      done: true,
      done_reason: 'stop',
      prompt_eval_count: overrides?.prompt_eval_count ?? 25,
      eval_count: overrides?.eval_count ?? 10,
    }),
  })
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('Ollama — AdapterUsageF52', () => {

  describe('Group 5: numCtx Option Pass-Through', () => {

    it('returns contextWindowSize equal to numCtx when option is provided; event includes it', async () => {
      // arrange
      const mockFetch = makeOllamaFetch({ prompt_eval_count: 25, eval_count: 10 })
      vi.stubGlobal('fetch', mockFetch)
      const observer = { onEvent: vi.fn() }
      const ollama = new Ollama('llama3.2', { numCtx: 8192 })
      ollama.bindObserver(observer)

      // act
      const result = await ollama.invoke(messages)

      // assert
      expect(result.usage.contextWindowSize).toBe(8192)
      expect(result.usage.inputTokens).toBe(25)
      expect(result.usage.outputTokens).toBe(10)
      const event = observer.onEvent.mock.calls.find((call) => call[1] === 'llm.response')?.[2]
      expect(event.contextWindowSize).toBe(8192)
    })

    it('returns contextWindowSize as undefined when numCtx is not provided; event has no contextWindowSize field', async () => {
      // arrange
      const mockFetch = makeOllamaFetch()
      vi.stubGlobal('fetch', mockFetch)
      const observer = { onEvent: vi.fn() }
      const ollama = new Ollama('llama3.2', {})
      ollama.bindObserver(observer)

      // act
      const result = await ollama.invoke(messages)

      // assert
      expect(result.usage.contextWindowSize).toBeUndefined()
      const event = observer.onEvent.mock.calls.find((call) => call[1] === 'llm.response')?.[2]
      expect(event).not.toHaveProperty('contextWindowSize')
    })

  })

})

describe('MockOllama — AdapterUsageF52', () => {

  describe('Group 6: Fixed Zero Usage', () => {

    it('invoke returns usage = { inputTokens: 0, outputTokens: 0 } with no contextWindowSize; emitted event has no contextWindowSize', async () => {
      // arrange
      const observer = { onEvent: vi.fn() }
      const mockOllama = new MockOllama({
        text: 'Ok',
        toolCalls: [],
        stopReason: 'end',
        usage: { inputTokens: 0, outputTokens: 0 },
      })
      mockOllama.bindObserver(observer)

      // act
      const result = await mockOllama.invoke(messages)

      // assert
      expect(result.usage).toEqual({ inputTokens: 0, outputTokens: 0 })
      expect(result.usage.contextWindowSize).toBeUndefined()
      const event = observer.onEvent.mock.calls.find((call) => call[1] === 'llm.response')?.[2]
      expect(event).not.toHaveProperty('contextWindowSize')
    })

  })

})
