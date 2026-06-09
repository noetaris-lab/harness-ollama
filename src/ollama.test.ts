import { describe, it, expect, vi, beforeEach } from 'vitest'
import { Ollama, OllamaApiError } from './ollama.js'
import type { Message } from '@noetaris/harness-types'

// Helper to extract the parsed request body from the first fetch call
function getRequestBody(mockFetch: ReturnType<typeof vi.fn>): Record<string, unknown> {
  const call = mockFetch.mock.calls[0] as [string, { body: string }]
  return JSON.parse(call[1].body) as Record<string, unknown>
}

describe('Ollama', () => {

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('basic invocation and request shape', () => {

    it('sends correct URL, model, user message, stream:false and returns normalized LLMResponse', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'Hello back' },
          done: true,
          done_reason: 'stop',
          prompt_eval_count: 10,
          eval_count: 5,
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)

      // act
      const result = await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(mockFetch).toHaveBeenCalledWith('http://localhost:11434/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'llama3.2',
          messages: [{ role: 'user', content: 'hello' }],
          stream: false,
        }),
      })
      expect(result.text).toBe('Hello back')
      expect(result.toolCalls).toEqual([])
      expect(result.stopReason).toBe('end')
    })

    it('uses custom baseUrl in fetch URL', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'hi' },
          done: true,
          done_reason: 'stop',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2', { baseUrl: 'http://my-server:11434' })

      // act
      await adapter.invoke([{ role: 'user', content: 'ping' }])

      // assert
      expect(mockFetch).toHaveBeenCalledWith('http://my-server:11434/api/chat', expect.any(Object))
    })

  })

  describe('message and tool translation', () => {

    it('translates Tool array to Ollama tools format', async () => {
      // arrange
      const tool = {
        name: 'get_weather',
        description: 'Get current weather',
        inputSchema: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
      }
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: '', tool_calls: [{ function: { name: 'get_weather', arguments: { city: 'Paris' } } }] },
          done: true,
          done_reason: 'tool_calls',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act
      await adapter.invoke([{ role: 'user', content: 'weather?' }], { tools: [tool] })

      // assert
      const body = getRequestBody(mockFetch)
      expect(body.tools).toEqual([{
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Get current weather',
          parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
        },
      }])
    })

    it('omits tools field from request body when options not provided', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'ok' },
          done: true,
          done_reason: 'stop',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act
      await adapter.invoke([{ role: 'user', content: 'ping' }])

      // assert
      const body = getRequestBody(mockFetch)
      expect(body).not.toHaveProperty('tools')
    })

    it('translates assistant message with toolCalls only — content is ""', async () => {
      // arrange
      // Message type with exactOptionalPropertyTypes: content must be omitted, not set to undefined
      const assistantMsg: Message = { role: 'assistant', toolCalls: [{ id: 'tc1', name: 'get_weather', input: { city: 'Paris' } }] }
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'done' },
          done: true,
          done_reason: 'stop',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act
      await adapter.invoke([assistantMsg])

      // assert
      const body = getRequestBody(mockFetch)
      const messages = body.messages as Array<Record<string, unknown>>
      expect(messages[0]).toEqual({
        role: 'assistant',
        content: '',
        tool_calls: [{ function: { name: 'get_weather', arguments: { city: 'Paris' } } }],
      })
    })

    it('translates assistant message with both content and toolCalls', async () => {
      // arrange
      const assistantMsg: Message = {
        role: 'assistant',
        content: 'Let me check',
        toolCalls: [{ id: 'tc2', name: 'lookup', input: { q: 'test' } }],
      }
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'ok' },
          done: true,
          done_reason: 'stop',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act
      await adapter.invoke([assistantMsg])

      // assert
      const body = getRequestBody(mockFetch)
      const messages = body.messages as Array<Record<string, unknown>>
      expect(messages[0]).toEqual({
        role: 'assistant',
        content: 'Let me check',
        tool_calls: [{ function: { name: 'lookup', arguments: { q: 'test' } } }],
      })
    })

    it('translates a single tool message to { role: "tool", content }', async () => {
      // arrange
      const toolMsg: Message = { role: 'tool', toolCallId: 'tc1', content: '{"temp":22}' }
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'The temp is 22C' },
          done: true,
          done_reason: 'stop',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act
      await adapter.invoke([toolMsg])

      // assert
      const body = getRequestBody(mockFetch)
      const messages = body.messages as Array<Record<string, unknown>>
      expect(messages[0]).toEqual({ role: 'tool', content: '{"temp":22}' })
    })

    it('translates multiple consecutive tool messages as separate messages — no grouping', async () => {
      // arrange
      const msgs: Message[] = [
        { role: 'tool', toolCallId: 'tc1', content: 'result1' },
        { role: 'tool', toolCallId: 'tc2', content: 'result2' },
      ]
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'got both' },
          done: true,
          done_reason: 'stop',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act
      await adapter.invoke(msgs)

      // assert
      const body = getRequestBody(mockFetch)
      expect(body.messages).toEqual([
        { role: 'tool', content: 'result1' },
        { role: 'tool', content: 'result2' },
      ])
    })

  })

  describe('response normalization', () => {

    it('normalizes single tool call response — name, parsed input, UUID id, stopReason "tool_use"', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: {
            role: 'assistant',
            content: '',
            tool_calls: [{ function: { name: 'get_weather', arguments: { city: 'Paris' } } }],
          },
          done: true,
          done_reason: 'tool_calls',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act
      const result = await adapter.invoke(
        [{ role: 'user', content: 'weather?' }],
        { tools: [{ name: 'get_weather', description: 'Get weather', inputSchema: {} }] },
      )

      // assert
      expect(result.toolCalls).toHaveLength(1)
      const tc = result.toolCalls[0]
      expect(tc?.name).toBe('get_weather')
      expect(tc?.input).toEqual({ city: 'Paris' })
      expect(tc?.id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i)
      expect(result.text).toBe('')
      expect(result.stopReason).toBe('tool_use')
    })

    it('normalizes response with both content and tool_calls — text and toolCalls both populated', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: {
            role: 'assistant',
            content: 'Let me check',
            tool_calls: [{ function: { name: 'lookup', arguments: { q: 'test' } } }],
          },
          done: true,
          done_reason: 'tool_calls',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act
      const result = await adapter.invoke(
        [{ role: 'user', content: 'go' }],
        { tools: [{ name: 'lookup', description: 'search', inputSchema: {} }] },
      )

      // assert
      expect(result.text).toBe('Let me check')
      expect(result.toolCalls).toHaveLength(1)
      expect(result.toolCalls[0]?.name).toBe('lookup')
      expect(result.stopReason).toBe('tool_use')
    })

    it('normalizes multiple tool calls — one ToolCall per entry, each id is a distinct UUID', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: {
            role: 'assistant',
            content: '',
            tool_calls: [
              { function: { name: 'tool_a', arguments: { x: 1 } } },
              { function: { name: 'tool_b', arguments: { y: 2 } } },
            ],
          },
          done: true,
          done_reason: 'tool_calls',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act
      const result = await adapter.invoke(
        [{ role: 'user', content: 'run both' }],
        { tools: [{ name: 'tool_a', description: 'A', inputSchema: {} }, { name: 'tool_b', description: 'B', inputSchema: {} }] },
      )

      // assert
      expect(result.toolCalls).toHaveLength(2)
      const tc0 = result.toolCalls[0]
      const tc1 = result.toolCalls[1]
      expect(tc0?.name).toBe('tool_a')
      expect(tc0?.input).toEqual({ x: 1 })
      expect(tc1?.name).toBe('tool_b')
      expect(tc1?.input).toEqual({ y: 2 })
      expect(tc0?.id).not.toBe(tc1?.id)
      expect(tc0?.id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i)
    })

    it('maps done_reason "length" to stopReason "max_tokens"', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'truncated...' },
          done: true,
          done_reason: 'length',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act
      const result = await adapter.invoke([{ role: 'user', content: 'write a lot' }])

      // assert
      expect(result.stopReason).toBe('max_tokens')
    })

    it('maps unrecognized done_reason to stopReason "end" (safe fallback)', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'partial' },
          done: true,
          done_reason: 'cancelled',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act
      const result = await adapter.invoke([{ role: 'user', content: 'test' }])

      // assert
      expect(result.stopReason).toBe('end')
    })

    it('treats empty tool_calls array as no tool calls — toolCalls is [], stopReason from done_reason', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'done', tool_calls: [] },
          done: true,
          done_reason: 'stop',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act
      const result = await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(result.toolCalls).toEqual([])
      expect(result.stopReason).toBe('end')
    })

  })

  describe('observer wiring and StepContext', () => {

    it('calls observer.onEvent with correct payload including token counts', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'hi' },
          done: true,
          done_reason: 'stop',
          prompt_eval_count: 50,
          eval_count: 120,
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)
      adapter.setStepContext({ agentId: 'agent-1', sessionId: 'sess-1', stepName: 'step-1' })

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(
        { agentId: 'agent-1', sessionId: 'sess-1', stepName: 'step-1' },
        'llm.response',
        { tokens: { input: 50, output: 120 }, modelId: 'llama3.2', stopReason: 'end', providerName: 'ollama' },
      )
    })

    it('defaults token counts to 0 when prompt_eval_count and eval_count are absent', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'hi' },
          done: true,
          done_reason: 'stop',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(
        expect.any(Object),
        'llm.response',
        expect.objectContaining({ tokens: { input: 0, output: 0 } }),
      )
    })

    it('does not throw when observer is NOOP ({} with no onEvent method)', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'ok' },
          done: true,
          done_reason: 'stop',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')
      adapter.bindObserver({})

      // act / assert
      await expect(adapter.invoke([{ role: 'user', content: 'hi' }])).resolves.not.toThrow()
    })

    it('uses zeroed StepContext when setStepContext was never called', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'hi' },
          done: true,
          done_reason: 'stop',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(
        { agentId: '', sessionId: '', stepName: '' },
        'llm.response',
        expect.any(Object),
      )
    })

    it('uses StepContext from setStepContext in observer event', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'done' },
          done: true,
          done_reason: 'stop',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)
      adapter.setStepContext({ agentId: 'my-agent', sessionId: 'my-sess', stepName: 'my-step' })

      // act
      await adapter.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(
        { agentId: 'my-agent', sessionId: 'my-sess', stepName: 'my-step' },
        'llm.response',
        expect.any(Object),
      )
    })

  })

  describe('error propagation', () => {

    it('throws OllamaApiError with status code on non-2xx response — llm.response not emitted', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({ ok: false, status: 404, text: vi.fn().mockResolvedValue('Not Found') })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)

      // act / assert
      await expect(adapter.invoke([{ role: 'user', content: 'hi' }])).rejects.toThrow(OllamaApiError)

      let caught!: OllamaApiError
      try {
        await adapter.invoke([{ role: 'user', content: 'hi' }])
      } catch (e) {
        caught = e as OllamaApiError
      }
      expect(caught.status).toBe(404)
      const eventTypes = observer.onEvent.mock.calls.map((c: unknown[]) => c[1])
      expect(eventTypes).not.toContain('llm.response')
    })

    it('throws OllamaApiError with status 500 — error includes status and body text', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({ ok: false, status: 500, text: vi.fn().mockResolvedValue('Internal Server Error') })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act / assert
      await expect(adapter.invoke([{ role: 'user', content: 'hi' }])).rejects.toThrow(OllamaApiError)

      let caught!: OllamaApiError
      try {
        await adapter.invoke([{ role: 'user', content: 'hi' }])
      } catch (e) {
        caught = e as OllamaApiError
      }
      expect(caught.status).toBe(500)
      expect(caught.message).toContain('500')
      expect(caught.message).toContain('Internal Server Error')
      expect(caught.name).toBe('OllamaApiError')
    })

    it('propagates network error from fetch unchanged — llm.response not emitted', async () => {
      // arrange
      const networkError = new Error('ECONNREFUSED connect ECONNREFUSED 127.0.0.1:11434')
      const mockFetch = vi.fn().mockRejectedValue(networkError)
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)

      // act / assert
      await expect(adapter.invoke([{ role: 'user', content: 'hi' }])).rejects.toThrow('ECONNREFUSED')
      const eventTypes = observer.onEvent.mock.calls.map((c: unknown[]) => c[1])
      expect(eventTypes).not.toContain('llm.response')
    })

  })

  describe('Group 5: Ollama — absent params produce no options key in request body', () => {

    it('sends request body without options key when no generation params are set', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ message: { role: 'assistant', content: 'hi', tool_calls: undefined }, done_reason: 'stop', prompt_eval_count: 5, eval_count: 2 }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      const fetchInit = mockFetch.mock.calls[0]?.[1] as { body: string }
      const parsedBody = JSON.parse(fetchInit.body) as Record<string, unknown>
      expect(parsedBody).not.toHaveProperty('options')
      expect(parsedBody).toMatchObject({ model: 'llama3.2', stream: false })
    })

  })

  describe('Group 6: Ollama — individual and combined params forwarded with correct field names', () => {

    it('puts temperature under options.temperature when set', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ message: { role: 'assistant', content: 'hi', tool_calls: undefined }, done_reason: 'stop', prompt_eval_count: 5, eval_count: 2 }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2', { temperature: 0.6 })

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      const fetchInit = mockFetch.mock.calls[0]?.[1] as { body: string }
      const parsedBody = JSON.parse(fetchInit.body) as Record<string, unknown>
      expect(parsedBody.options).toMatchObject({ temperature: 0.6 })
    })

    it('puts maxTokens under options.num_predict when set', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ message: { role: 'assistant', content: 'hi', tool_calls: undefined }, done_reason: 'stop', prompt_eval_count: 5, eval_count: 2 }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2', { maxTokens: 256 })

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      const fetchInit = mockFetch.mock.calls[0]?.[1] as { body: string }
      const parsedBody = JSON.parse(fetchInit.body) as Record<string, unknown>
      const opts = parsedBody.options as Record<string, unknown>
      expect(opts).toMatchObject({ num_predict: 256 })
      expect(opts).not.toHaveProperty('maxTokens')
      expect(opts).not.toHaveProperty('max_tokens')
    })

    it('puts topP under options.top_p when set', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ message: { role: 'assistant', content: 'hi', tool_calls: undefined }, done_reason: 'stop', prompt_eval_count: 5, eval_count: 2 }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2', { topP: 0.9 })

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      const fetchInit = mockFetch.mock.calls[0]?.[1] as { body: string }
      const parsedBody = JSON.parse(fetchInit.body) as Record<string, unknown>
      const opts = parsedBody.options as Record<string, unknown>
      expect(opts).toMatchObject({ top_p: 0.9 })
      expect(opts).not.toHaveProperty('topP')
    })

    it('puts topK under options.top_k when set', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ message: { role: 'assistant', content: 'hi', tool_calls: undefined }, done_reason: 'stop', prompt_eval_count: 5, eval_count: 2 }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2', { topK: 50 })

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      const fetchInit = mockFetch.mock.calls[0]?.[1] as { body: string }
      const parsedBody = JSON.parse(fetchInit.body) as Record<string, unknown>
      const opts = parsedBody.options as Record<string, unknown>
      expect(opts).toMatchObject({ top_k: 50 })
      expect(opts).not.toHaveProperty('topK')
    })

    it('puts all four params under options when all are set', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ message: { role: 'assistant', content: 'hi', tool_calls: undefined }, done_reason: 'stop', prompt_eval_count: 5, eval_count: 2 }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2', { temperature: 0.7, maxTokens: 128, topP: 0.85, topK: 40 })

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      const fetchInit = mockFetch.mock.calls[0]?.[1] as { body: string }
      const parsedBody = JSON.parse(fetchInit.body) as Record<string, unknown>
      expect(parsedBody.options).toEqual({ temperature: 0.7, num_predict: 128, top_p: 0.85, top_k: 40 })
    })

  })

  describe('Group 7: Ollama — absent params excluded; options key absent when all are omitted', () => {

    it('excludes options key entirely when all generation params are omitted', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ message: { role: 'assistant', content: 'hi', tool_calls: undefined }, done_reason: 'stop', prompt_eval_count: 5, eval_count: 2 }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2', {})

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      const fetchInit = mockFetch.mock.calls[0]?.[1] as { body: string }
      const parsedBody = JSON.parse(fetchInit.body) as Record<string, unknown>
      expect(parsedBody).not.toHaveProperty('options')
    })

  })

  describe('Group 8: Ollama — observer event integrity with generation params', () => {

    it('emits llm.response event with correct fields when generation params are set', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ message: { role: 'assistant', content: 'reply', tool_calls: undefined }, done_reason: 'stop', prompt_eval_count: 8, eval_count: 3 }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2', { temperature: 0.4, topK: 20 })
      const onEvent = vi.fn()
      adapter.bindObserver({ onEvent })

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(onEvent).toHaveBeenCalledWith(
        expect.any(Object),
        'llm.response',
        expect.objectContaining({ tokens: { input: 8, output: 3 }, modelId: 'llama3.2', stopReason: 'end', providerName: 'ollama' }),
      )
    })

  })

  describe('Group 9: Ollama — non-2xx response throws OllamaApiError', () => {

    it('throws OllamaApiError when fetch returns a non-2xx response', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        text: () => Promise.resolve('Service Unavailable'),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act / assert
      await expect(adapter.invoke([{ role: 'user', content: 'hello' }])).rejects.toThrow(OllamaApiError)
      await expect(adapter.invoke([{ role: 'user', content: 'hello' }])).rejects.toMatchObject({ statusCode: 503 })
    })

  })

  describe('"llm.request" emission', () => {

    it('emits "llm.request" with modelId and providerName: "ollama" before fetch', async () => {
      // arrange
      const mockFetch = vi.fn()
      const minimalOllamaResponse = { model: 'llama3.2', message: { role: 'assistant', content: 'ok' }, done: true }
      mockFetch.mockResolvedValue({ ok: true, json: async () => minimalOllamaResponse })
      vi.stubGlobal('fetch', mockFetch)
      const mockObserver = { onEvent: vi.fn() }
      const adapter = new Ollama('llama3.2')
      adapter.bindObserver(mockObserver)

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(mockObserver.onEvent.mock.calls[0]?.[1]).toBe('llm.request')
      expect(mockObserver.onEvent.mock.calls[0]?.[2]).toEqual({ modelId: 'llama3.2', providerName: 'ollama' })
      expect(mockFetch).toHaveBeenCalledOnce()
      expect(mockObserver.onEvent.mock.invocationCallOrder[0] ?? 0).toBeLessThan(mockFetch.mock.invocationCallOrder[0] ?? 0)
    })

    it('emits "llm.request" before "llm.response" on success; no optional content fields', async () => {
      // arrange
      const mockFetch = vi.fn()
      const minimalOllamaResponse = { model: 'llama3.2', message: { role: 'assistant', content: 'ok' }, done: true }
      mockFetch.mockResolvedValue({ ok: true, json: async () => minimalOllamaResponse })
      vi.stubGlobal('fetch', mockFetch)
      const mockObserver = { onEvent: vi.fn() }
      const adapter = new Ollama('llama3.2')
      adapter.bindObserver(mockObserver)

      // act
      await adapter.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockObserver.onEvent).toHaveBeenCalledTimes(2)
      expect(mockObserver.onEvent.mock.calls[0]?.[1]).toBe('llm.request')
      expect(mockObserver.onEvent.mock.calls[1]?.[1]).toBe('llm.response')
      expect(mockObserver.onEvent.mock.calls[0]?.[2]).not.toHaveProperty('messages')
      expect(mockObserver.onEvent.mock.calls[0]?.[2]).not.toHaveProperty('tools')
      expect(mockObserver.onEvent.mock.calls[1]?.[2]).not.toHaveProperty('output')
    })

    it('emits "llm.request" before fetch throw and does not emit "llm.response" on error', async () => {
      // arrange
      const mockFetch = vi.fn()
      mockFetch.mockRejectedValue(new Error('ECONNREFUSED'))
      vi.stubGlobal('fetch', mockFetch)
      const mockObserver = { onEvent: vi.fn() }
      const adapter = new Ollama('llama3.2')
      adapter.bindObserver(mockObserver)

      // act
      await expect(adapter.invoke([{ role: 'user', content: 'hi' }])).rejects.toThrow('ECONNREFUSED')

      // assert
      expect(mockObserver.onEvent).toHaveBeenCalledTimes(1)
      expect(mockObserver.onEvent.mock.calls[0]?.[1]).toBe('llm.request')
    })

  })

  describe('edge cases and repeated calls', () => {

    it('passes empty messages array through to fetch body and returns LLMResponse normally', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'ok' },
          done: true,
          done_reason: 'stop',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')

      // act
      const result = await adapter.invoke([])

      // assert
      const body = getRequestBody(mockFetch)
      expect(body.messages).toEqual([])
      expect(result.text).toBe('ok')
      expect(result.toolCalls).toEqual([])
      expect(result.stopReason).toBe('end')
    })

    it('second bindObserver call replaces first — subsequent invokes use the new observer', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'hi' },
          done: true,
          done_reason: 'stop',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')
      const firstObserver = { onEvent: vi.fn() }
      const secondObserver = { onEvent: vi.fn() }
      adapter.bindObserver(firstObserver)
      adapter.bindObserver(secondObserver)

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(secondObserver.onEvent).toHaveBeenCalled()
      expect(firstObserver.onEvent).not.toHaveBeenCalled()
    })

    it('last setStepContext call wins when called multiple times before invoke', async () => {
      // arrange
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          model: 'llama3.2',
          message: { role: 'assistant', content: 'ok' },
          done: true,
          done_reason: 'stop',
        }),
      })
      vi.stubGlobal('fetch', mockFetch)
      const adapter = new Ollama('llama3.2')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)
      adapter.setStepContext({ agentId: 'a1', sessionId: 's1', stepName: 'step-old' })
      adapter.setStepContext({ agentId: 'a1', sessionId: 's1', stepName: 'step-new' })

      // act
      await adapter.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(
        { agentId: 'a1', sessionId: 's1', stepName: 'step-new' },
        'llm.response',
        expect.any(Object),
      )
    })

  })

})
