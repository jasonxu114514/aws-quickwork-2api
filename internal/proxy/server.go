package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"quickwork-openai/internal/accounts"
	"quickwork-openai/internal/config"
	"quickwork-openai/internal/store"
)

type Server struct {
	cfg      *config.Config
	accounts *accounts.Manager
	tokens   *APIKeyAuth
	upstream *Upstream
	sessions *conversationStore
}

func NewServer(cfg *config.Config, mgr *accounts.Manager) *Server {
	return &Server{
		cfg:      cfg,
		accounts: mgr,
		tokens:   NewAPIKeyAuth(cfg.APIKeys),
		upstream: NewUpstream(cfg),
		sessions: newConversationStore(cfg.SessionTTL.Duration(), 256),
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	return s.withCORS(mux)
}

func (s *Server) withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Api-Key, X-Thread-Id, X-Conversation-Id")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "service": "quickwork-openai"})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeOpenAIError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET required")
		return
	}
	if !s.tokens.Allowed(r) {
		writeOpenAIError(w, http.StatusUnauthorized, "invalid_api_key", "invalid proxy API key")
		return
	}
	created := time.Now().Unix()
	data := make([]map[string]any, 0, len(s.cfg.Models))
	for _, id := range s.cfg.Models {
		data = append(data, map[string]any{
			"id":       id,
			"object":   "model",
			"created":  created,
			"owned_by": "quickwork",
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeOpenAIError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	if !s.tokens.Allowed(r) {
		writeOpenAIError(w, http.StatusUnauthorized, "invalid_api_key", "invalid proxy API key")
		return
	}

	var req ChatRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8<<20))
	if err := dec.Decode(&req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON body: "+err.Error())
		return
	}
	if req.N > 1 {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "n > 1 is not supported")
		return
	}

	prompt, err := formatPrompt(req.Messages)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	model := s.mapModel(req.Model)
	allowedTools := requestedToolNames(&req)
	choiceMode, choiceName := parseToolChoice(req.ToolChoice)
	if choiceMode == "function" && choiceName != "" {
		allowedTools[choiceName] = true
	}
	parallelTools := true
	if req.ParallelToolCalls != nil {
		parallelTools = *req.ParallelToolCalls
	}
	bridgePrompt := buildToolBridgePrompt(&req, choiceMode, choiceName, parallelTools)
	bridgeActive := bridgePrompt != ""
	systemPromptText := systemPrompt(req.Messages, bridgePrompt)
	// QuickWork has no per-turn system prompt channel, so the caller's
	// instructions are attached at the head of the user turn by the upstream
	// layer, while the transcript stays a separate field for attachment upload.
	instructions := ""
	if bridgeActive || hasCallerInstructions(req.Messages) {
		instructions = systemPromptText
	}

	conv := s.sessions.acquire(conversationKey(r, &req, systemPromptText))
	defer s.sessions.release(conv)

	resume := false
	promptText := prompt
	promptMessages := req.Messages
	if s.cfg.SessionReuse && conv.systemPrompt == systemPromptText {
		if delta, ok := conv.delta(req.Messages); ok {
			if deltaPrompt, err := formatPrompt(delta); err == nil && hasVisibleText(deltaPrompt) {
				promptText = deltaPrompt
				promptMessages = delta
				resume = true
			}
		}
	}

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.RequestTimeout.Duration())
	defer cancel()

	threadID := strings.TrimSpace(r.Header.Get("X-Thread-Id"))
	id := "chatcmpl-" + strings.ReplaceAll(uuid(), "-", "")
	created := time.Now().Unix()
	in := ChatInput{
		Prompt:        promptText,
		Messages:      promptMessages,
		Model:         model,
		ThreadID:      threadID,
		Title:         titleFor(lastUserText(req.Messages)),
		SystemPrompt:  instructions,
		ChatAgentID:   s.cfg.ChatAgentID,
		CreateSession: true,
		DeleteSession: !s.cfg.SessionReuse && !s.cfg.KeepSessions,
	}
	if resume {
		in.ThreadID = conv.threadID
		in.CreateSession = false
	}

	startedAt := time.Now()
	maxAttempts := s.maxAttempts(ctx)
	if resume {
		maxAttempts++
	}
	attempts := 0
	var usedAccount *store.Account
	requestedModel := strings.TrimSpace(req.Model)
	if requestedModel == "" {
		requestedModel = "(default)"
	}
	log.Printf("chat %s start model=%s upstream_model=%s stream=%t messages=%d prompt_chars=%d system_chars=%d reuse=%t thread=%q",
		id, requestedModel, model, req.Stream, len(req.Messages), len(promptText), len(systemPromptText), resume, in.ThreadID)

	selectAccount := func(attempt int) (*store.Account, error) {
		if resume && attempt == 1 && conv.accountID != 0 {
			acct, err := s.accounts.NextPinned(ctx, conv.accountID)
			if err == nil {
				return acct, nil
			}
			log.Printf("chat %s pinned account %d unavailable: %v; starting a new session",
				id, conv.accountID, err)
			conv.reset()
			resume = false
			in.ThreadID = ""
			in.CreateSession = true
			in.Prompt = prompt
			in.Messages = req.Messages
		}
		return s.accounts.Next(ctx)
	}

	if !req.Stream {
		var res ChatResult
		var lastErr error
		for attempt := 1; attempt <= maxAttempts; attempt++ {
			attempts = attempt
			acct, err := selectAccount(attempt)
			if err != nil {
				lastErr = err
				log.Printf("chat %s attempt=%d/%d account=none error=%v", id, attempt, maxAttempts, err)
				break
			}
			usedAccount = acct
			sess := SessionFromAccount(acct, s.cfg)
			if err := sess.validate(); err != nil {
				s.accounts.ReportFailure(ctx, acct.ID, err)
				lastErr = err
				log.Printf("chat %s attempt=%d/%d account=%s invalid_session error=%v",
					id, attempt, maxAttempts, accountLogLabel(acct), err)
				continue
			}
			log.Printf("chat %s attempt=%d/%d account=%s", id, attempt, maxAttempts, accountLogLabel(acct))
			res, err = s.upstream.Chat(ctx, sess, in, nil)
			if err == nil {
				lastErr = nil
				break
			}
			lastErr = err
			if !errors.Is(err, ErrUserMessageTooLarge) && ctx.Err() == nil {
				s.accounts.ReportFailure(ctx, acct.ID, err)
			}
			log.Printf("chat %s attempt=%d/%d account=%s failed error=%v",
				id, attempt, maxAttempts, accountLogLabel(acct), err)
			if errors.Is(err, ErrUserMessageTooLarge) {
				break
			}
			if resume {
				conv.reset()
				resume = false
				in.ThreadID = ""
				in.CreateSession = true
				in.Prompt = prompt
				in.Messages = req.Messages
				log.Printf("chat %s session resume failed; retrying with full transcript", id)
				continue
			}
			if ctx.Err() != nil {
				log.Printf("chat %s request context ended: %v", id, ctx.Err())
				break
			}
		}
		if lastErr != nil {
			conv.reset()
			status, code := http.StatusBadGateway, "upstream_error"
			switch {
			case errors.Is(lastErr, accounts.ErrNoAccounts):
				status, code = http.StatusUnauthorized, "upstream_session_unavailable"
			case errors.Is(lastErr, ErrUserMessageTooLarge):
				status, code = http.StatusRequestEntityTooLarge, "context_length_exceeded"
			}
			logChatDone(id, startedAt, in, usedAccount, attempts, "failed", "", res.Usage, 0, lastErr)
			writeOpenAIError(w, status, code, lastErr.Error())
			return
		}
		if s.cfg.SessionReuse && res.ThreadID != "" {
			conv.threadID = res.ThreadID
			conv.accountID = usedAccount.ID
			conv.messages = append([]Message(nil), req.Messages...)
			conv.systemPrompt = systemPromptText
		}
		text := res.Text
		var replies []ToolCallReply
		if bridgeActive {
			text, replies = parseClientToolCalls(text, allowedTools)
			if len(replies) == 0 {
				replies = nativeToolReplies(res.ToolCalls, allowedTools)
			}
			replies = limitToolReplies(replies, choiceMode, choiceName, parallelTools)
		}
		toolCalls := replyToolCalls(replies)
		if !hasVisibleText(text) && len(toolCalls) == 0 {
			errEmpty := fmt.Errorf("upstream returned no visible text or tool calls (completion_tokens=%d)", res.Usage.CompletionTokens)
			logChatDone(id, startedAt, in, usedAccount, attempts, "empty", "", res.Usage, 0, errEmpty)
			writeOpenAIError(w, http.StatusBadGateway, "upstream_error", errEmpty.Error())
			return
		}
		message := map[string]any{"role": "assistant"}
		if !hasVisibleText(text) {
			message["content"] = nil
		} else {
			message["content"] = text
		}
		if len(toolCalls) > 0 {
			message["tool_calls"] = toolCalls
		}
		finish := "stop"
		if len(toolCalls) > 0 {
			finish = "tool_calls"
		}
		logChatDone(id, startedAt, in, usedAccount, attempts, "ok", finish, res.Usage, len(toolCalls), nil)
		writeJSON(w, http.StatusOK, map[string]any{
			"id":      id,
			"object":  "chat.completion",
			"created": created,
			"model":   model,
			"choices": []map[string]any{{
				"index":         0,
				"message":       message,
				"finish_reason": finish,
			}},
			"usage": usageMap(res.Usage),
		})
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeOpenAIError(w, http.StatusInternalServerError, "streaming_unavailable", "response writer does not support flushing")
		return
	}
	started := false
	writeRaw := func(payload map[string]any) {
		body, _ := json.Marshal(payload)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", body)
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
		writeRaw(map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{{
				"index":         0,
				"delta":         map[string]any{"role": "assistant"},
				"finish_reason": nil,
			}},
		})
	}
	writePayload := func(payload map[string]any) {
		ensureStarted()
		writeRaw(payload)
	}
	writeChunk := func(delta map[string]any, finish any) {
		writePayload(map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{{
				"index":         0,
				"delta":         delta,
				"finish_reason": finish,
			}},
		})
	}
	writeStreamError := func(message string) {
		writePayload(map[string]any{"error": map[string]any{
			"message": message,
			"type":    "upstream_error",
			"code":    "upstream_error",
		}})
	}

	sawVisibleText := false
	var bridgeBuffer strings.Builder

	var res ChatResult
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		attempts = attempt
		acct, err := selectAccount(attempt)
		if err != nil {
			lastErr = err
			log.Printf("chat %s attempt=%d/%d account=none error=%v", id, attempt, maxAttempts, err)
			break
		}
		usedAccount = acct
		sess := SessionFromAccount(acct, s.cfg)
		if err := sess.validate(); err != nil {
			s.accounts.ReportFailure(ctx, acct.ID, err)
			lastErr = err
			log.Printf("chat %s attempt=%d/%d account=%s invalid_session error=%v",
				id, attempt, maxAttempts, accountLogLabel(acct), err)
			continue
		}
		log.Printf("chat %s attempt=%d/%d account=%s", id, attempt, maxAttempts, accountLogLabel(acct))
		res, err = s.upstream.Chat(ctx, sess, in, func(ev StreamEvent) error {
			if ev.Type == "text" && ev.Text != "" {
				if bridgeActive {
					bridgeBuffer.WriteString(ev.Text)
					return nil
				}
				if hasVisibleText(ev.Text) {
					sawVisibleText = true
				}
				writeChunk(map[string]any{"content": ev.Text}, nil)
			}
			return nil
		})
		if err == nil {
			lastErr = nil
			break
		}
		lastErr = err
		if !errors.Is(err, ErrUserMessageTooLarge) && ctx.Err() == nil {
			s.accounts.ReportFailure(ctx, acct.ID, err)
		}
		log.Printf("chat %s attempt=%d/%d account=%s failed error=%v",
			id, attempt, maxAttempts, accountLogLabel(acct), err)
		if errors.Is(err, ErrUserMessageTooLarge) {
			break
		}
		if resume && !started {
			conv.reset()
			resume = false
			in.ThreadID = ""
			in.CreateSession = true
			in.Prompt = prompt
			in.Messages = req.Messages
			sawVisibleText = false
			bridgeBuffer.Reset()
			log.Printf("chat %s session resume failed; retrying with full transcript", id)
			continue
		}
		if started || ctx.Err() != nil {
			if started {
				log.Printf("chat %s stream already started; not retrying", id)
			}
			break
		}
		// Nothing reached the client yet; reset per-attempt state and try
		// the next account in rotation.
		sawVisibleText = false
		bridgeBuffer.Reset()
	}
	if lastErr != nil {
		conv.reset()
		if !started {
			status, code := http.StatusBadGateway, "upstream_error"
			switch {
			case errors.Is(lastErr, accounts.ErrNoAccounts):
				status, code = http.StatusUnauthorized, "upstream_session_unavailable"
			case errors.Is(lastErr, ErrUserMessageTooLarge):
				status, code = http.StatusRequestEntityTooLarge, "context_length_exceeded"
			}
			logChatDone(id, startedAt, in, usedAccount, attempts, "failed", "", res.Usage, 0, lastErr)
			writeOpenAIError(w, status, code, lastErr.Error())
			return
		}
		logChatDone(id, startedAt, in, usedAccount, attempts, "failed", "", res.Usage, 0, lastErr)
		writeStreamError(lastErr.Error())
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}
	if s.cfg.SessionReuse && res.ThreadID != "" {
		conv.threadID = res.ThreadID
		conv.accountID = usedAccount.ID
		conv.messages = append([]Message(nil), req.Messages...)
		conv.systemPrompt = systemPromptText
	}
	var outText string
	var outTools []ToolCall
	finish := "stop"
	if bridgeActive {
		// Assign to the outer variables: a `:=` here would shadow outText and
		// drop the buffered answer, streaming a stop with usage but no content.
		var replies []ToolCallReply
		outText, replies = parseClientToolCalls(bridgeBuffer.String(), allowedTools)
		if len(replies) == 0 {
			replies = nativeToolReplies(res.ToolCalls, allowedTools)
		}
		replies = limitToolReplies(replies, choiceMode, choiceName, parallelTools)
		outTools = replyToolCalls(replies)
		if len(outTools) > 0 {
			finish = "tool_calls"
		} else if !hasVisibleText(outText) {
			errEmpty := fmt.Errorf("upstream returned no visible text or tool calls (completion_tokens=%d)", res.Usage.CompletionTokens)
			logChatDone(id, startedAt, in, usedAccount, attempts, "empty", "", res.Usage, 0, errEmpty)
			writeStreamError(errEmpty.Error())
			_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}
	} else if !sawVisibleText {
		errEmpty := fmt.Errorf("upstream returned no visible text (completion_tokens=%d)", res.Usage.CompletionTokens)
		logChatDone(id, startedAt, in, usedAccount, attempts, "empty", "", res.Usage, 0, errEmpty)
		writeStreamError(errEmpty.Error())
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}
	if hasVisibleText(outText) {
		writeChunk(map[string]any{"content": outText}, nil)
	}
	for i, tc := range outTools {
		writeChunk(map[string]any{"tool_calls": []map[string]any{{
			"index": i,
			"id":    tc.ID,
			"type":  "function",
			"function": map[string]any{
				"name":      tc.Function.Name,
				"arguments": tc.Function.Arguments,
			},
		}}}, nil)
	}
	writeChunk(map[string]any{}, finish)
	logChatDone(id, startedAt, in, usedAccount, attempts, "ok", finish, res.Usage, len(outTools), nil)
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		writePayload(map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []any{},
			"usage":   usageMap(res.Usage),
		})
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// accountLogLabel identifies an account in request logs without exposing
// tokens. Format: id:provider:email.
func accountLogLabel(a *store.Account) string {
	if a == nil {
		return "none"
	}
	name := strings.TrimSpace(a.Email)
	if name == "" {
		name = strings.TrimSpace(a.DisplayName)
	}
	if name == "" {
		name = "-"
	}
	return fmt.Sprintf("%d:%s:%s", a.ID, a.Provider, name)
}

// logChatDone emits one completion line per chat request with the outcome,
// selected account, mapped model, token usage, finish reason, and error.
func logChatDone(id string, startedAt time.Time, in ChatInput, acct *store.Account, attempts int, status, finish string, u Usage, toolCalls int, err error) {
	if finish == "" {
		finish = "-"
	}
	line := fmt.Sprintf("chat %s done status=%s account=%s model=%s attempts=%d elapsed=%s tokens{prompt=%d completion=%d total=%d cache_read=%d cache_write=%d} finish=%s tool_calls=%d",
		id, status, accountLogLabel(acct), in.Model, attempts, time.Since(startedAt).Round(time.Millisecond),
		u.PromptTokens, u.CompletionTokens, u.TotalTokens, u.CacheReadTokens, u.CacheWriteTokens, finish, toolCalls)
	if err != nil {
		log.Printf("%s error=%v", line, err)
		return
	}
	log.Print(line)
}

func (s *Server) maxAttempts(ctx context.Context) int {
	n, err := s.accounts.EnabledCount(ctx)
	if err != nil || n < 1 {
		return 1
	}
	max := s.cfg.MaxAttempts
	if max < 1 {
		max = 1
	}
	if n < max {
		return n
	}
	return max
}

func (s *Server) mapModel(model string) string {
	m := strings.ToLower(strings.TrimSpace(model))
	m = strings.TrimPrefix(m, "quickwork/")
	m = strings.TrimPrefix(m, "qw-")
	switch m {
	case "smart", "fast", "balanced", "auto":
		return m
	case "":
		return s.cfg.DefaultModel
	default:
		log.Printf("unknown model %q, using %s", model, s.cfg.DefaultModel)
		return s.cfg.DefaultModel
	}
}

func usageMap(u Usage) map[string]any {
	completion := u.CompletionTokens
	total := u.TotalTokens
	if total == 0 {
		total = u.PromptTokens + u.CacheReadTokens + u.CacheWriteTokens + completion
	}
	prompt := total - completion
	if prompt < 0 {
		prompt = u.PromptTokens + u.CacheReadTokens + u.CacheWriteTokens
	}
	m := map[string]any{
		"prompt_tokens":     prompt,
		"completion_tokens": completion,
		"total_tokens":      total,
	}
	if u.CacheReadTokens > 0 {
		m["prompt_tokens_details"] = map[string]any{"cached_tokens": u.CacheReadTokens}
	}
	return m
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeOpenAIError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    code,
			"code":    code,
		},
	})
}
