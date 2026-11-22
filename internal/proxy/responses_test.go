package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"quickwork-openai/internal/config"
)

func TestParseResponsesMessagesStringInput(t *testing.T) {
	msgs, err := parseResponsesMessages("Be concise.", "Hello QuickWork")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].Role != "system" || msgs[0].Content != "Be concise." {
		t.Fatalf("unexpected system message: %#v", msgs[0])
	}
	if msgs[1].Role != "user" || msgs[1].Content != "Hello QuickWork" {
		t.Fatalf("unexpected user message: %#v", msgs[1])
	}
}

func TestParseResponsesMessagesArrayWithToolLoop(t *testing.T) {
	input := []any{
		map[string]any{
			"type": "message",
			"role": "user",
			"content": []any{
				map[string]any{"type": "input_text", "text": "Check weather in Taipei."},
			},
		},
		map[string]any{
			"type":      "function_call",
			"call_id":   "call_123",
			"name":      "get_weather",
			"arguments": `{"city":"Taipei"}`,
		},
		map[string]any{
			"type":    "function_call_output",
			"call_id": "call_123",
			"output":  `{"temp":"24C"}`,
		},
	}

	msgs, err := parseResponsesMessages("", input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(msgs))
	}
	if msgs[1].Role != "assistant" || len(msgs[1].ToolCalls) != 1 || msgs[1].ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("unexpected assistant tool call message: %#v", msgs[1])
	}
	if msgs[2].Role != "tool" || msgs[2].ToolCallID != "call_123" || msgs[2].Content != `{"temp":"24C"}` {
		t.Fatalf("unexpected tool output message: %#v", msgs[2])
	}

	prompt, err := formatPrompt(msgs)
	if err != nil {
		t.Fatalf("formatPrompt error: %v", err)
	}
	if !strings.Contains(prompt, "get_weather") || !strings.Contains(prompt, "call_id=call_123") {
		t.Fatalf("formatted prompt missing tool history:\n%s", prompt)
	}
}

func TestToChatToolsSupportsFlatAndNested(t *testing.T) {
	tools := toChatTools([]responsesTool{
		{
			Type:        "function",
			Name:        "read_file",
			Description: "Read a file",
			Parameters:  map[string]any{"type": "object"},
		},
		{
			Type: "function",
			Function: &ToolFunction{
				Name: "write_file",
			},
		},
		{
			Type: "web_search",
		},
	})
	if len(tools) != 2 {
		t.Fatalf("expected 2 function tools, got %d", len(tools))
	}
	if tools[0].Function.Name != "read_file" || tools[1].Function.Name != "write_file" {
		t.Fatalf("unexpected converted tools: %#v", tools)
	}
}

func TestResponsesRoutesAndPathNormalization(t *testing.T) {
	cfg := config.Default()
	srv := NewServer(&cfg, nil)
	h := srv.Handler()

	// 確認 /v1/responses、/v1/v1/responses、/responses 不會回傳 404（GET 應回 405 Method Not Allowed）。
	for _, path := range []string{"/v1/responses", "/v1/v1/responses", "/responses", "/v1/responses/"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusMethodNotAllowed {
			t.Fatalf("path %s: expected 405, got %d", path, rec.Code)
		}
	}

	// 確認無 body 的 POST /v1/responses 會進入 handler 回傳 400，而非 404。
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("POST /v1/responses empty body: expected 400, got %d (%s)", rec.Code, rec.Body.String())
	}
}
