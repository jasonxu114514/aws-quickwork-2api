package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"quickwork-openai/internal/accounts"
)

// ResponsesRequest 代表 OpenAI Responses API (/v1/responses) 的請求結構。
type ResponsesRequest struct {
	Model             string          `json:"model"`
	Instructions      string          `json:"instructions,omitempty"`
	Input             any             `json:"input"`
	Stream            bool            `json:"stream"`
	Tools             []responsesTool `json:"tools,omitempty"`
	ToolChoice        any             `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`
	MaxOutputTokens   int             `json:"max_output_tokens,omitempty"`
	Temperature       *float64        `json:"temperature,omitempty"`
	Store             *bool           `json:"store,omitempty"`
	PreviousRespID    string          `json:"previous_response_id,omitempty"`
}

// responsesTool 同時相容 Responses API 扁平格式與 Chat Completions 巢狀格式。
type responsesTool struct {
	Type        string        `json:"type"`
	Name        string        `json:"name,omitempty"`
	Description string        `json:"description,omitempty"`
	Parameters  any           `json:"parameters,omitempty"`
	Strict      bool          `json:"strict,omitempty"`
	Function    *ToolFunction `json:"function,omitempty"`
}

// toChatTools 將 Responses API 的 tools 轉換為內部統一使用的 []Tool。
func toChatTools(tools []responsesTool) []Tool {
	out := make([]Tool, 0, len(tools))
	for _, t := range tools {
		typ := strings.ToLower(strings.TrimSpace(t.Type))
		if typ != "" && typ != "function" {
			// 非 function 類型的內建工具（如 web_search）目前無法由客戶端橋接執行，略過之。
			continue
		}
		if t.Function != nil && strings.TrimSpace(t.Function.Name) != "" {
			out = append(out, Tool{Type: "function", Function: *t.Function})
			continue
		}
		if strings.TrimSpace(t.Name) != "" {
			out = append(out, Tool{
				Type: "function",
				Function: ToolFunction{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.Parameters,
					Strict:      t.Strict,
				},
			})
		}
	}
	return out
}

// parseResponsesMessages 將 Responses API 的 instructions 與 input 轉換為 []Message。
func parseResponsesMessages(instructions string, input any) ([]Message, error) {
	var msgs []Message
	if strings.TrimSpace(instructions) != "" {
		msgs = append(msgs, Message{
			Role:    "system",
			Content: instructions,
		})
	}

	switch v := input.(type) {
	case nil:
		return nil, fmt.Errorf("input must not be empty")
	case string:
		if strings.TrimSpace(v) == "" {
			return nil, fmt.Errorf("input must not be empty")
		}
		msgs = append(msgs, Message{Role: "user", Content: v})
		return msgs, nil
	case []any:
		for _, rawItem := range v {
			item, ok := rawItem.(map[string]any)
			if !ok {
				continue
			}
			itemType, _ := item["type"].(string)
			itemType = strings.ToLower(strings.TrimSpace(itemType))

			switch itemType {
			case "", "message":
				role, _ := item["role"].(string)
				role = strings.TrimSpace(role)
				if role == "" {
					role = "user"
				}
				text := extractResponsesContent(item["content"])
				if strings.TrimSpace(text) != "" {
					msgs = append(msgs, Message{Role: role, Content: text})
				}
			case "function_call":
				callID, _ := item["call_id"].(string)
				if strings.TrimSpace(callID) == "" {
					callID, _ = item["id"].(string)
				}
				name, _ := item["name"].(string)
				args := stringifyAny(item["arguments"])
				if strings.TrimSpace(args) == "" {
					args = "{}"
				}
				msgs = append(msgs, Message{
					Role: "assistant",
					ToolCalls: []ToolCall{{
						ID:   callID,
						Type: "function",
						Function: FunctionCall{
							Name:      name,
							Arguments: args,
						},
					}},
				})
			case "function_call_output":
				callID, _ := item["call_id"].(string)
				output := extractResponsesContent(item["output"])
				msgs = append(msgs, Message{
					Role:       "tool",
					ToolCallID: callID,
					Content:    output,
				})
			case "reasoning":
				// 忽略多輪上下文回傳的 reasoning 項目。
				continue
			default:
				// 其他未知項目若有 content 或 text，嘗試作為 user 內容保留。
				if text := extractResponsesContent(item["content"]); strings.TrimSpace(text) != "" {
					msgs = append(msgs, Message{Role: "user", Content: text})
				}
			}
		}
	default:
		// 若為單一物件，嘗試作為單一 message 處理。
		if item, ok := v.(map[string]any); ok {
			return parseResponsesMessages(instructions, []any{item})
		}
		return nil, fmt.Errorf("unsupported input type")
	}

	if len(msgs) == 0 {
		return nil, fmt.Errorf("input contains no messages")
	}
	return msgs, nil
}

// extractResponsesContent 解析 Responses API 的 content / output 欄位為純文字。
func extractResponsesContent(c any) string {
	switch v := c.(type) {
	case nil:
		return ""
	case string:
		return v
	case []any:
		var b strings.Builder
		for _, elem := range v {
			part, ok := elem.(map[string]any)
			if !ok {
				if s, ok := elem.(string); ok {
					b.WriteString(s)
				}
				continue
			}
			typ, _ := part["type"].(string)
			switch strings.ToLower(strings.TrimSpace(typ)) {
			case "input_text", "output_text", "text":
				if t, ok := part["text"].(string); ok {
					b.WriteString(t)
				}
			case "refusal":
				if r, ok := part["refusal"].(string); ok {
					b.WriteString(r)
				}
			case "input_image":
				b.WriteString("[image]")
			case "input_audio":
				b.WriteString("[audio]")
			case "input_file":
				b.WriteString("[file]")
			default:
				if t, ok := part["text"].(string); ok {
					b.WriteString(t)
				} else if body, err := json.Marshal(part); err == nil {
					b.Write(body)
				}
			}
		}
		return b.String()
	default:
		return stringifyAny(v)
	}
}

func stringifyAny(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	body, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(body)
}

func responsesUsageMap(u Usage) map[string]any {
	completion := u.CompletionTokens
	total := u.TotalTokens
	if total == 0 {
		total = u.PromptTokens + u.CacheReadTokens + u.CacheWriteTokens + completion
	}
	prompt := total - completion
	if prompt < 0 {
		prompt = u.PromptTokens + u.CacheReadTokens + u.CacheWriteTokens
	}
	return map[string]any{
		"input_tokens": prompt,
		"input_tokens_details": map[string]any{
			"cached_tokens": u.CacheReadTokens,
		},
		"output_tokens": completion,
		"output_tokens_details": map[string]any{
			"reasoning_tokens": 0,
		},
		"total_tokens": total,
	}
}

func buildResponsesOutputItems(msgID, text string, toolCalls []ToolCall) []map[string]any {
	output := make([]map[string]any, 0, 1+len(toolCalls))
	if strings.TrimSpace(text) != "" {
		output = append(output, map[string]any{
			"id":     msgID,
			"type":   "message",
			"status": "completed",
			"role":   "assistant",
			"content": []map[string]any{{
				"type":        "output_text",
				"text":        text,
				"annotations": []any{},
			}},
		})
	}
	for _, tc := range toolCalls {
		args := strings.TrimSpace(tc.Function.Arguments)
		if args == "" {
			args = "{}"
		}
		output = append(output, map[string]any{
			"id":        "fc_" + strings.ReplaceAll(uuid(), "-", ""),
			"type":      "function_call",
			"status":    "completed",
			"call_id":   tc.ID,
			"name":      tc.Function.Name,
			"arguments": args,
		})
	}
	return output
}

func buildResponseObject(respID, model, instructions, status string, created int64, output []map[string]any, usage map[string]any) map[string]any {
	if output == nil {
		output = []map[string]any{}
	}
	resp := map[string]any{
		"id":                 respID,
		"object":             "response",
		"created_at":         created,
		"status":             status,
		"error":              nil,
		"incomplete_details": nil,
		"model":              model,
		"output":             output,
	}
	if strings.TrimSpace(instructions) != "" {
		resp["instructions"] = instructions
	} else {
		resp["instructions"] = nil
	}
	if usage != nil {
		resp["usage"] = usage
	}
	return resp
}

// handleResponses 處理 OpenAI Responses API (POST /v1/responses)。
func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeOpenAIError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	if !s.tokens.Allowed(r) {
		writeOpenAIError(w, http.StatusUnauthorized, "invalid_api_key", "invalid proxy API key")
		return
	}

	var req ResponsesRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8<<20))
	if err := dec.Decode(&req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON body: "+err.Error())
		return
	}

	msgs, err := parseResponsesMessages(req.Instructions, req.Input)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	prompt, err := formatPrompt(msgs)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	chatTools := toChatTools(req.Tools)
	chatReq := ChatRequest{
		Tools:      chatTools,
		ToolChoice: req.ToolChoice,
	}
	allowedTools := requestedToolNames(&chatReq)
	choiceMode, choiceName := parseToolChoice(req.ToolChoice)
	if choiceMode == "function" && choiceName != "" {
		allowedTools[choiceName] = true
	}
	parallelTools := true
	if req.ParallelToolCalls != nil {
		parallelTools = *req.ParallelToolCalls
	}

	systemPromptText := systemPrompt(msgs)
	bridgePrompt := buildToolBridgePrompt(chatTools, nil, choiceMode, choiceName, parallelTools)
	bridgeActive := bridgePrompt != ""
	if bridgeActive {
		systemPromptText += "\n\n" + bridgePrompt
	}
	bridgeAllowed := allowedTools
	if choiceMode == "function" && choiceName != "" {
		bridgeAllowed = map[string]bool{choiceName: true}
	}

	model := s.mapModel(req.Model)
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.RequestTimeout.Duration())
	defer cancel()

	threadID := strings.TrimSpace(r.Header.Get("X-Thread-Id"))
	respID := "resp_" + strings.ReplaceAll(uuid(), "-", "")
	msgID := "msg_" + strings.ReplaceAll(uuid(), "-", "")
	created := time.Now().Unix()

	in := ChatInput{
		Prompt:       prompt,
		Model:        model,
		ThreadID:     threadID,
		Title:        titleFor(lastUserText(msgs)),
		SystemPrompt: systemPromptText,
	}

	if !req.Stream {
		res, lastErr := s.execNonStream(ctx, in)
		if lastErr != nil {
			status, code := http.StatusBadGateway, "upstream_error"
			if errors.Is(lastErr, accounts.ErrNoAccounts) {
				status, code = http.StatusUnauthorized, "upstream_session_unavailable"
			}
			writeOpenAIError(w, status, code, lastErr.Error())
			return
		}
		text := res.Text
		var bridgeReplies []ToolCallReply
		if bridgeActive {
			text, bridgeReplies = parseClientToolCalls(text, bridgeAllowed)
			if !parallelTools && len(bridgeReplies) > 1 {
				bridgeReplies = bridgeReplies[:1]
			}
		}
		toolCalls := replyToolCalls(bridgeReplies)
		if len(toolCalls) == 0 {
			toolCalls = filterToolCalls(res.ToolCalls, allowedTools, choiceMode, choiceName, parallelTools)
		}
		if strings.TrimSpace(text) == "" && len(toolCalls) == 0 {
			writeOpenAIError(w, http.StatusBadGateway, "upstream_error", "upstream returned no text or tool calls")
			return
		}
		output := buildResponsesOutputItems(msgID, text, toolCalls)
		writeJSON(w, http.StatusOK, buildResponseObject(
			respID, model, req.Instructions, "completed", created, output, responsesUsageMap(res.Usage),
		))
		return
	}

	// 串流模式：發送標準 OpenAI Responses API SSE 事件。
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeOpenAIError(w, http.StatusInternalServerError, "streaming_unavailable", "response writer does not support flushing")
		return
	}

	started := false
	seq := 0
	writeSSE := func(eventType string, payload map[string]any) {
		payload["type"] = eventType
		payload["sequence_number"] = seq
		seq++
		body, _ := json.Marshal(payload)
		_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, body)
		flusher.Flush()
	}

	ensureStarted := func() {
		if started {
			return
		}
		started = true
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		if threadID != "" {
			w.Header().Set("X-Thread-Id", threadID)
		}
		w.WriteHeader(http.StatusOK)

		inProg := buildResponseObject(respID, model, req.Instructions, "in_progress", created, nil, nil)
		writeSSE("response.created", map[string]any{"response": inProg})
		writeSSE("response.in_progress", map[string]any{"response": inProg})
	}

	nextOutputIdx := 0
	textItemStarted := false
	textOutputIdx := -1
	var textBuf strings.Builder

	ensureTextItem := func() {
		if textItemStarted {
			return
		}
		ensureStarted()
		textItemStarted = true
		textOutputIdx = nextOutputIdx
		nextOutputIdx++
		writeSSE("response.output_item.added", map[string]any{
			"output_index": textOutputIdx,
			"item": map[string]any{
				"id":      msgID,
				"type":    "message",
				"status":  "in_progress",
				"role":    "assistant",
				"content": []any{},
			},
		})
		writeSSE("response.content_part.added", map[string]any{
			"item_id":       msgID,
			"output_index":  textOutputIdx,
			"content_index": 0,
			"part": map[string]any{
				"type":        "output_text",
				"text":        "",
				"annotations": []any{},
			},
		})
	}

	closeTextItem := func() map[string]any {
		fullText := textBuf.String()
		writeSSE("response.output_text.done", map[string]any{
			"item_id":       msgID,
			"output_index":  textOutputIdx,
			"content_index": 0,
			"text":          fullText,
		})
		part := map[string]any{
			"type":        "output_text",
			"text":        fullText,
			"annotations": []any{},
		}
		writeSSE("response.content_part.done", map[string]any{
			"item_id":       msgID,
			"output_index":  textOutputIdx,
			"content_index": 0,
			"part":          part,
		})
		item := map[string]any{
			"id":      msgID,
			"type":    "message",
			"status":  "completed",
			"role":    "assistant",
			"content": []map[string]any{part},
		}
		writeSSE("response.output_item.done", map[string]any{
			"output_index": textOutputIdx,
			"item":         item,
		})
		return item
	}

	type streamToolState struct {
		fcID      string
		callID    string
		name      string
		outputIdx int
		args      string
	}
	toolMap := make(map[string]*streamToolState)
	var toolOrder []string
	var bridgeBuffer strings.Builder

	emitNativeTool := func(tc ToolCallState) {
		if !toolAllowed(tc, allowedTools, choiceMode, choiceName) {
			return
		}
		st, seen := toolMap[tc.ID]
		if !seen {
			if !parallelTools && len(toolOrder) > 0 {
				return
			}
			ensureStarted()
			fcID := "fc_" + strings.ReplaceAll(uuid(), "-", "")
			idx := nextOutputIdx
			nextOutputIdx++
			st = &streamToolState{
				fcID:      fcID,
				callID:    tc.ID,
				name:      tc.Name,
				outputIdx: idx,
			}
			toolMap[tc.ID] = st
			toolOrder = append(toolOrder, tc.ID)
			writeSSE("response.output_item.added", map[string]any{
				"output_index": idx,
				"item": map[string]any{
					"id":        fcID,
					"type":      "function_call",
					"status":    "in_progress",
					"call_id":   tc.ID,
					"name":      tc.Name,
					"arguments": "",
				},
			})
		}
		args := strings.TrimSpace(tc.Arguments)
		if args == "" {
			args = "{}"
		}
		delta := args
		if strings.HasPrefix(args, st.args) {
			delta = args[len(st.args):]
		}
		st.args = args
		if delta != "" {
			writeSSE("response.function_call_arguments.delta", map[string]any{
				"item_id":      st.fcID,
				"output_index": st.outputIdx,
				"delta":        delta,
			})
		}
	}

	var res ChatResult
	var lastErr error
	for range s.maxAttempts(ctx) {
		acct, err := s.accounts.Next(ctx)
		if err != nil {
			lastErr = err
			break
		}
		sess := SessionFromAccount(acct, s.cfg)
		if err := sess.validate(); err != nil {
			s.accounts.ReportFailure(ctx, acct.ID, err)
			lastErr = err
			continue
		}
		res, err = s.upstream.Chat(ctx, sess, in, func(ev StreamEvent) error {
			switch ev.Type {
			case "text":
				if ev.Text == "" {
					return nil
				}
				if bridgeActive {
					bridgeBuffer.WriteString(ev.Text)
					return nil
				}
				ensureTextItem()
				textBuf.WriteString(ev.Text)
				writeSSE("response.output_text.delta", map[string]any{
					"item_id":       msgID,
					"output_index":  textOutputIdx,
					"content_index": 0,
					"delta":         ev.Text,
				})
			case "tool_call", "tool_call_update":
				if bridgeActive {
					return nil
				}
				if ev.ToolCall != nil {
					emitNativeTool(*ev.ToolCall)
				}
			}
			return nil
		})
		if err == nil {
			lastErr = nil
			break
		}
		lastErr = err
		s.accounts.ReportFailure(ctx, acct.ID, err)
		if started || ctx.Err() != nil {
			break
		}
		// 尚未輸出給客戶端前可安全重置並換號重試。
		nextOutputIdx = 0
		textItemStarted = false
		textOutputIdx = -1
		textBuf.Reset()
		bridgeBuffer.Reset()
		toolMap = make(map[string]*streamToolState)
		toolOrder = nil
	}

	if lastErr != nil {
		if !started {
			status, code := http.StatusBadGateway, "upstream_error"
			if errors.Is(lastErr, accounts.ErrNoAccounts) {
				status, code = http.StatusUnauthorized, "upstream_session_unavailable"
			}
			writeOpenAIError(w, status, code, lastErr.Error())
			return
		}
		writeSSE("error", map[string]any{
			"code":    "upstream_error",
			"message": lastErr.Error(),
		})
		return
	}

	var completedItems []map[string]any
	if bridgeActive {
		text, replies := parseClientToolCalls(bridgeBuffer.String(), bridgeAllowed)
		if !parallelTools && len(replies) > 1 {
			replies = replies[:1]
		}
		if strings.TrimSpace(text) == "" && len(replies) == 0 {
			if !started {
				writeOpenAIError(w, http.StatusBadGateway, "upstream_error", "upstream returned no text or tool calls")
				return
			}
			writeSSE("error", map[string]any{
				"code":    "upstream_error",
				"message": "upstream returned no text or tool calls",
			})
			return
		}
		ensureStarted()
		if strings.TrimSpace(text) != "" {
			ensureTextItem()
			textBuf.WriteString(text)
			writeSSE("response.output_text.delta", map[string]any{
				"item_id":       msgID,
				"output_index":  textOutputIdx,
				"content_index": 0,
				"delta":         text,
			})
			completedItems = append(completedItems, closeTextItem())
		}
		for _, tc := range replyToolCalls(replies) {
			fcID := "fc_" + strings.ReplaceAll(uuid(), "-", "")
			idx := nextOutputIdx
			nextOutputIdx++
			args := strings.TrimSpace(tc.Function.Arguments)
			if args == "" {
				args = "{}"
			}
			writeSSE("response.output_item.added", map[string]any{
				"output_index": idx,
				"item": map[string]any{
					"id":        fcID,
					"type":      "function_call",
					"status":    "in_progress",
					"call_id":   tc.ID,
					"name":      tc.Function.Name,
					"arguments": "",
				},
			})
			writeSSE("response.function_call_arguments.delta", map[string]any{
				"item_id":      fcID,
				"output_index": idx,
				"delta":        args,
			})
			writeSSE("response.function_call_arguments.done", map[string]any{
				"item_id":      fcID,
				"output_index": idx,
				"arguments":    args,
			})
			fcItem := map[string]any{
				"id":        fcID,
				"type":      "function_call",
				"status":    "completed",
				"call_id":   tc.ID,
				"name":      tc.Function.Name,
				"arguments": args,
			}
			writeSSE("response.output_item.done", map[string]any{
				"output_index": idx,
				"item":         fcItem,
			})
			completedItems = append(completedItems, fcItem)
		}
	} else {
		if !textItemStarted && len(toolOrder) == 0 {
			if !started {
				writeOpenAIError(w, http.StatusBadGateway, "upstream_error", "upstream returned no text or tool calls")
				return
			}
			writeSSE("error", map[string]any{
				"code":    "upstream_error",
				"message": "upstream returned no text or tool calls",
			})
			return
		}
		if textItemStarted {
			completedItems = append(completedItems, closeTextItem())
		}
		for _, id := range toolOrder {
			st := toolMap[id]
			writeSSE("response.function_call_arguments.done", map[string]any{
				"item_id":      st.fcID,
				"output_index": st.outputIdx,
				"arguments":    st.args,
			})
			fcItem := map[string]any{
				"id":        st.fcID,
				"type":      "function_call",
				"status":    "completed",
				"call_id":   st.callID,
				"name":      st.name,
				"arguments": st.args,
			}
			writeSSE("response.output_item.done", map[string]any{
				"output_index": st.outputIdx,
				"item":         fcItem,
			})
			completedItems = append(completedItems, fcItem)
		}
	}

	completedResp := buildResponseObject(
		respID, model, req.Instructions, "completed", created, completedItems, responsesUsageMap(res.Usage),
	)
	writeSSE("response.completed", map[string]any{"response": completedResp})
}
