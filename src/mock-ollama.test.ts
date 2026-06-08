import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MockOllama, MockOllamaEmptyQueueError } from './mock-ollama.js'
import type { LLMResponse, Message } from '@noetaris/harness-types'
import type { StepContext } from '@noetaris/harness'

describe('MockOllama', () => {

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('queue — single response (sticky last)', () => {

    it('returns the same response on every call when constructed with a single response', async () => {
      // arrange
      const response: LLMResponse = { text: 'hello', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const mock = new MockOllama(response)

      // act
      const r1 = await mock.invoke([])
      const r2 = await mock.invoke([])
      const r3 = await mock.invoke([])

      // assert
      expect(r1).toEqual(response)
      expect(r2).toEqual(response)
      expect(r3).toEqual(response)
    })

  })

  describe('queue — multi-response FIFO with sticky last', () => {

    it('returns responses in FIFO order when constructed with three responses', async () => {
      // arrange
      const r1: LLMResponse = { text: 'one', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const r2: LLMResponse = { text: 'two', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const r3: LLMResponse = { text: 'three', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const mock = new MockOllama([r1, r2, r3])

      // act
      const a = await mock.invoke([])
      const b = await mock.invoke([])
      const c = await mock.invoke([])

      // assert
      expect(a).toEqual(r1)
      expect(b).toEqual(r2)
      expect(c).toEqual(r3)
    })

    it('last response is sticky after all preceding responses are consumed', async () => {
      // arrange
      const r1: LLMResponse = { text: 'one', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const r2: LLMResponse = { text: 'two', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const r3: LLMResponse = { text: 'three', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const mock = new MockOllama([r1, r2, r3])
      await mock.invoke([])
      await mock.invoke([])
      await mock.invoke([])

      // act
      const d = await mock.invoke([])

      // assert
      expect(d).toEqual(r3)
    })

    it('calls beyond queue size all return the last response', async () => {
      // arrange
      const r1: LLMResponse = { text: 'one', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const r2: LLMResponse = { text: 'two', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const mock = new MockOllama([r1, r2])

      // act
      const a = await mock.invoke([])
      const b = await mock.invoke([])
      const c = await mock.invoke([])
      const d = await mock.invoke([])

      // assert
      expect(a).toEqual(r1)
      expect(b).toEqual(r2)
      expect(c).toEqual(r2)
      expect(d).toEqual(r2)
    })

  })

  describe('enqueue — appending responses after construction', () => {

    it('returns enqueued response after starting with an empty queue', async () => {
      // arrange
      const response: LLMResponse = { text: 'added', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const mock = new MockOllama()
      mock.enqueue(response)

      // act
      const result = await mock.invoke([])

      // assert
      expect(result).toEqual(response)
    })

    it('returns all responses in order when enqueue is called after construction', async () => {
      // arrange
      const r1: LLMResponse = { text: 'one', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const r2: LLMResponse = { text: 'two', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const r3: LLMResponse = { text: 'three', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const mock = new MockOllama([r1])
      mock.enqueue([r2, r3])

      // act
      const a = await mock.invoke([])
      const b = await mock.invoke([])
      const c = await mock.invoke([])

      // assert
      expect(a).toEqual(r1)
      expect(b).toEqual(r2)
      expect(c).toEqual(r3)
    })

  })

  describe('observer integration', () => {

    it('fires onEvent with correct event type, payload, and stopReason after invoke', async () => {
      // arrange
      const response: LLMResponse = { text: 'hi', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const mock = new MockOllama(response)
      const observer = { onEvent: vi.fn() }
      mock.bindObserver(observer)
      const ctx: StepContext = { agentId: 'a1', sessionId: 's1', stepName: 'step1' }
      mock.setStepContext(ctx)

      // act
      await mock.invoke([])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(ctx, 'llm.response', { tokens: { input: 0, output: 0 }, modelId: 'mock', stopReason: 'end', providerName: 'mock' })
    })

    it('passes the StepContext from setStepContext to onEvent', async () => {
      // arrange
      const mock = new MockOllama({ text: 'hi', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } })
      const observer = { onEvent: vi.fn() }
      mock.bindObserver(observer)
      const ctx: StepContext = { agentId: 'agent-99', sessionId: 'sess-42', stepName: 'my-step' }
      mock.setStepContext(ctx)

      // act
      await mock.invoke([])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(ctx, 'llm.response', expect.any(Object))
    })

    it('passes default StepContext to onEvent when setStepContext is never called', async () => {
      // arrange
      const mock = new MockOllama({ text: 'hi', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } })
      const observer = { onEvent: vi.fn() }
      mock.bindObserver(observer)

      // act
      await mock.invoke([])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith({ agentId: '', sessionId: '', stepName: '' }, 'llm.response', expect.any(Object))
    })

    it('does not throw when observer is a NOOP object with no onEvent method', async () => {
      // arrange
      const mock = new MockOllama({ text: 'hi', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } })
      mock.bindObserver({})

      // act
      const result = await mock.invoke([])

      // assert
      expect(result).toBeDefined()
    })

  })

  describe('lastMessages tracking', () => {

    it('lastMessages reflects the messages array from the most recent invoke', async () => {
      // arrange
      const mock = new MockOllama({ text: 'hi', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } })
      const msgs: Message[] = [{ role: 'user', content: 'hello' }]

      // act
      await mock.invoke(msgs)

      // assert
      expect(mock.lastMessages).toEqual(msgs)
    })

  })

  describe('error — empty queue', () => {

    it('throws MockOllamaEmptyQueueError when queue is empty at invoke time', async () => {
      // arrange
      const mock = new MockOllama()

      // act
      const p = mock.invoke([])

      // assert
      await expect(p).rejects.toThrow(MockOllamaEmptyQueueError)
    })

  })

  describe('"llm.request" emission', () => {

    it('emits "llm.request" with modelId: "mock" and providerName: "mock" before dequeue', async () => {
      // arrange
      const response: LLMResponse = { text: 'hi', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const adapter = new MockOllama(response)
      const mockObserver = { onEvent: vi.fn() }
      adapter.bindObserver(mockObserver)

      // act
      await adapter.invoke([])

      // assert
      expect(mockObserver.onEvent.mock.calls[0]?.[1]).toBe('llm.request')
      expect(mockObserver.onEvent.mock.calls[0]?.[2]).toEqual({ modelId: 'mock', providerName: 'mock' })
    })

    it('emits "llm.request" before MockOllamaEmptyQueueError throw and does not emit "llm.response"', async () => {
      // arrange
      const adapter = new MockOllama()
      const mockObserver = { onEvent: vi.fn() }
      adapter.bindObserver(mockObserver)

      // act
      await expect(adapter.invoke([])).rejects.toThrow(MockOllamaEmptyQueueError)

      // assert
      expect(mockObserver.onEvent).toHaveBeenCalledTimes(1)
      expect(mockObserver.onEvent.mock.calls[0]?.[1]).toBe('llm.request')
    })

  })

  describe('edge cases', () => {

    it('enqueue with a single non-array LLMResponse treats it as a one-element queue', async () => {
      // arrange
      const mock = new MockOllama()
      const response: LLMResponse = { text: 'single', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      mock.enqueue(response)

      // act
      const result = await mock.invoke([])

      // assert
      expect(result).toEqual(response)
    })

    it('onEvent fires on the most recently bound observer when bindObserver is called multiple times', async () => {
      // arrange
      const mock = new MockOllama({ text: 'hi', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } })
      const obs1 = { onEvent: vi.fn() }
      const obs2 = { onEvent: vi.fn() }
      mock.bindObserver(obs1)
      mock.bindObserver(obs2)

      // act
      await mock.invoke([])

      // assert
      expect(obs2.onEvent).toHaveBeenCalled()
      expect(obs1.onEvent).not.toHaveBeenCalled()
    })

    it('returns configured response and sets lastMessages to empty array when invoked with empty messages', async () => {
      // arrange
      const response: LLMResponse = { text: 'hi', toolCalls: [], stopReason: 'end', usage: { inputTokens: 0, outputTokens: 0 } }
      const mock = new MockOllama(response)

      // act
      const result = await mock.invoke([])

      // assert
      expect(result).toEqual(response)
      expect(mock.lastMessages).toEqual([])
    })

  })

})
