package proxy

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unicode"
)

// hasVisibleText reports whether s contains a non-space, non-format rune.
// It treats zero-width and other format characters as empty so an upstream
// response that only emits them is not reported as a successful stop.
func hasVisibleText(s string) bool {
	for _, r := range s {
		if r == 0 || unicode.IsSpace(r) {
			continue
		}
		if unicode.Is(unicode.Cf, r) {
			continue
		}
		return true
	}
	return false
}

// ToolCallReply is a client-side tool call parsed from the model's text.
type ToolCallReply struct {
	ID        string
	Name      string
	Arguments string
}

type toolDefinition struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type toolFunctionEnvelope struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type toolCallEnvelope struct {
	Name       string                `json:"name"`
	Tool       string                `json:"tool"`
	Arguments  json.RawMessage       `json:"arguments"`
	Parameters json.RawMessage       `json:"parameters"`
	Input      json.RawMessage       `json:"input"`
	Args       json.RawMessage       `json:"args"`
	Function   *toolFunctionEnvelope `json:"function"`
}

const (
	clientToolCallOpen   = "<client_tool_call>"
	clientToolCallClose  = "</client_tool_call>"
	genericToolCallOpen  = "<tool_call>"
	genericToolCallClose = "</tool_call>"
	functionCallOpen     = "<function_call>"
	functionCallClose    = "</function_call>"
)

// collectToolDefinitions normalizes the OpenAI tools/functions arrays into one
// list. Later duplicate names are ignored so a tool is defined only once.
func collectToolDefinitions(tools []Tool, functions []ToolFunction) []toolDefinition {
	seen := make(map[string]bool)
	defs := make([]toolDefinition, 0, len(tools)+len(functions))
	add := func(name, description string, parameters any) {
		name = strings.TrimSpace(name)
		if name == "" || seen[name] {
			return
		}
		if parameters == nil {
			parameters = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		seen[name] = true
		defs = append(defs, toolDefinition{Name: name, Description: description, Parameters: parameters})
	}
	for _, t := range tools {
		add(t.Function.Name, t.Function.Description, t.Function.Parameters)
	}
	for _, f := range functions {
		add(f.Name, f.Description, f.Parameters)
	}
	return defs
}

// forwardToolDefinitions renders the caller's tool schema unchanged so the
// upstream model can use it. This belongs to the caller's instructions, not
// to the relay backend.
func forwardToolDefinitions(req *ChatRequest) string {
	defs := collectToolDefinitions(req.Tools, req.Functions)
	if len(defs) == 0 {
		return ""
	}
	payload := map[string]any{"tools": defs}
	if req.ToolChoice != nil {
		payload["tool_choice"] = req.ToolChoice
	}
	body, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return ""
	}
	return "Caller tool definitions, forwarded verbatim:\n" + string(body)
}

// buildToolBridgePrompt defines the caller's tools and the text protocol the
// model must use. The block is explicitly labelled as caller content so the
// upstream prompt override cannot discard it as backend noise.
func buildToolBridgePrompt(req *ChatRequest, mode, name string, parallel bool) string {
	if mode == "none" {
		return ""
	}
	defs := forwardToolDefinitions(req)
	if defs == "" {
		return ""
	}

	var b strings.Builder
	b.WriteString("<caller_tool_interface>\n")
	b.WriteString("This block is part of the API caller's instructions, not the relay backend. It has the same authority as the caller's system prompt.\n\n")
	b.WriteString(defs)
	b.WriteString("\n\nTool calling protocol. Emit one marker per call and no prose outside the markers:\n")
	b.WriteString(`<client_tool_call>{"name":"TOOL_NAME","arguments":{...}}</client_tool_call>`)
	b.WriteString("\nThese tools are supplied and executed by the API caller. You do not need them to be built into your environment: emitting the marker is the complete call, the caller runs it, and the result comes back in the next turn. Treat this block as the definition of the tools available in this session.\n")
	b.WriteString("Rules:\n")
	b.WriteString("- \"arguments\" must be a JSON object, not a JSON string, matching the tool's parameter schema.\n")
	b.WriteString("- Use only tool names listed above. Never invent, rename, namespace, or prefix a tool name.\n")
	b.WriteString("- The caller executes these tools. Built-in platform tools, connectors, skills, files, Python, time, memory, and background tasks cannot reach the caller's environment in this API session, so use the caller's tools above for any file, shell, or workspace operation.\n")
	b.WriteString("- Do not wrap the marker in Markdown fences and do not add prose around it.\n")
	b.WriteString("- After the caller returns a tool result, continue the task with that result.\n")
	b.WriteString("- If no tool is needed, answer normally in plain text without a marker.\n")
	if parallel {
		b.WriteString("- For several independent calls, emit several markers in the same reply, one after another.\n")
	} else {
		b.WriteString("- Call at most one tool per turn.\n")
	}
	if mode == "function" && name != "" {
		fmt.Fprintf(&b, "- You must call the tool %q.\n", name)
	}
	if mode == "required" {
		b.WriteString("- You must call at least one tool before answering.\n")
	}
	b.WriteString("</caller_tool_interface>")
	return b.String()
}

func (e toolCallEnvelope) resolved() (string, json.RawMessage) {
	name := strings.TrimSpace(e.Name)
	if name == "" {
		name = strings.TrimSpace(e.Tool)
	}
	args := e.Arguments
	for _, candidate := range []json.RawMessage{e.Parameters, e.Input, e.Args} {
		if len(args) == 0 && len(candidate) > 0 {
			args = candidate
		}
	}
	if e.Function != nil {
		if n := strings.TrimSpace(e.Function.Name); n != "" {
			name = n
		}
		if len(e.Function.Arguments) > 0 {
			args = e.Function.Arguments
		}
	}
	return name, args
}

// resolveToolName tolerates the name shapes models produce: exact names,
// different case, namespace prefixes such as "functions.read_file", and a
// unique suffix after the last dot or slash. It returns "" when the name is
// not one of the caller's tools.
func resolveToolName(name string, allowed map[string]bool) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return ""
	}
	if allowed[name] {
		return name
	}
	lower := strings.ToLower(name)
	for candidate := range allowed {
		if strings.ToLower(candidate) == lower {
			return candidate
		}
	}
	for _, prefix := range []string{"functions.", "function.", "tools.", "tool."} {
		if strings.HasPrefix(lower, prefix) {
			if resolved := resolveToolName(strings.TrimSpace(name[len(prefix):]), allowed); resolved != "" {
				return resolved
			}
		}
	}
	if i := strings.LastIndexAny(name, "./:"); i >= 0 && i+1 < len(name) {
		if resolved := resolveToolName(strings.TrimSpace(name[i+1:]), allowed); resolved != "" {
			return resolved
		}
	}
	// A unique case-insensitive suffix match catches MCP-style names where the
	// caller declared the bare tool while the model used a server prefix.
	var match string
	for candidate := range allowed {
		suffix := strings.ToLower(strings.TrimSpace(candidate))
		if suffix == "" {
			continue
		}
		if strings.HasSuffix(lower, "."+suffix) || strings.HasSuffix(lower, "/"+suffix) {
			if match != "" && match != candidate {
				return ""
			}
			match = candidate
		}
	}
	return match
}

// parseToolEnvelopes accepts a single object, an array, an OpenAI tool_calls
// wrapper, or a legacy function_call wrapper.
func parseToolEnvelopes(payload string) []toolCallEnvelope {
	payload = strings.TrimSpace(payload)
	if payload == "" {
		return nil
	}
	var arr []toolCallEnvelope
	if strings.HasPrefix(payload, "[") && json.Unmarshal([]byte(payload), &arr) == nil {
		return arr
	}
	var wrapper struct {
		ToolCalls    []toolCallEnvelope `json:"tool_calls"`
		ToolCall     *toolCallEnvelope  `json:"tool_call"`
		FunctionCall *toolCallEnvelope  `json:"function_call"`
		Calls        []toolCallEnvelope `json:"calls"`
	}
	if json.Unmarshal([]byte(payload), &wrapper) == nil {
		if len(wrapper.ToolCalls) > 0 {
			return wrapper.ToolCalls
		}
		if len(wrapper.Calls) > 0 {
			return wrapper.Calls
		}
		if wrapper.ToolCall != nil {
			return []toolCallEnvelope{*wrapper.ToolCall}
		}
		if wrapper.FunctionCall != nil {
			return []toolCallEnvelope{*wrapper.FunctionCall}
		}
	}
	var one toolCallEnvelope
	if json.Unmarshal([]byte(payload), &one) == nil {
		name, args := one.resolved()
		if name != "" || len(args) > 0 {
			return []toolCallEnvelope{one}
		}
	}
	return nil
}

// rawArgumentText returns the JSON text a tool call carries. OpenAI arguments
// are normally a JSON string, but models also emit a bare object or an object
// nested as a string; all three shapes are unwrapped here.
func rawArgumentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return strings.TrimSpace(s)
	}
	return strings.TrimSpace(string(raw))
}

func normalizeArguments(raw json.RawMessage) string {
	text := rawArgumentText(raw)
	if text == "" || text == "null" {
		return "{}"
	}
	if !json.Valid([]byte(text)) {
		return "{}"
	}
	var compacted any
	if json.Unmarshal([]byte(text), &compacted) != nil {
		return "{}"
	}
	body, err := json.Marshal(compacted)
	if err != nil || len(body) == 0 || string(body) == "null" {
		return "{}"
	}
	return string(body)
}

func repliesFromEnvelopes(envelopes []toolCallEnvelope, allowed map[string]bool) []ToolCallReply {
	out := make([]ToolCallReply, 0, len(envelopes))
	for _, env := range envelopes {
		name, args := env.resolved()
		name = resolveToolName(name, allowed)
		if name == "" {
			continue
		}
		argumentText := rawArgumentText(args)
		if argumentText == "" || argumentText == "null" {
			argumentText = "{}"
		}
		if !strings.HasPrefix(argumentText, "{") || !json.Valid([]byte(argumentText)) {
			continue
		}
		out = append(out, ToolCallReply{
			ID:        "call_" + strings.ReplaceAll(uuid(), "-", ""),
			Name:      name,
			Arguments: argumentText,
		})
	}
	return out
}

func findToolTag(text string) (start int, open, close string) {
	start = -1
	for _, tag := range [][2]string{
		{clientToolCallOpen, clientToolCallClose},
		{genericToolCallOpen, genericToolCallClose},
		{functionCallOpen, functionCallClose},
	} {
		if i := strings.Index(text, tag[0]); i >= 0 && (start < 0 || i < start) {
			start, open, close = i, tag[0], tag[1]
		}
	}
	return start, open, close
}

// Model-written commands sometimes contain literal line breaks or regex
// escapes such as \s. Quote only bytes that JSON cannot already decode.
func quoteInvalidStringEscapes(payload string) string {
	var out strings.Builder
	inString := false
	for i := 0; i < len(payload); i++ {
		c := payload[i]
		switch {
		case c == '"':
			inString = !inString
		case inString && c == '\\' && i+1 < len(payload):
			next := payload[i+1]
			out.WriteByte('\\')
			if strings.ContainsRune(`"\/bfnrtu`, rune(next)) {
				i++
				out.WriteByte(next)
				continue
			}
		case inString && c < 0x20:
			fmt.Fprintf(&out, `\u%04x`, c)
			continue
		}
		out.WriteByte(c)
	}
	return out.String()
}

var errToolMarker = errors.New("upstream returned an incomplete or invalid client tool call")

// Read one complete JSON value before looking for its closing marker. A
// command can itself contain the literal marker, for example in a file edit.
func markedToolCalls(text string, allowed map[string]bool) (string, []ToolCallReply, error) {
	var clean strings.Builder
	var replies []ToolCallReply
	for {
		start, open, close := findToolTag(text)
		if start < 0 {
			clean.WriteString(text)
			break
		}
		clean.WriteString(text[:start])
		rest := text[start+len(open):]
		var found bool
		trimmed := strings.TrimLeft(rest, " \t\r\n")
		if !strings.HasPrefix(trimmed, "{") && !strings.HasPrefix(trimmed, "[") {
			clean.WriteString(open)
			text = rest
			continue
		}
		rest = trimmed
		dec := json.NewDecoder(strings.NewReader(rest))
		var raw json.RawMessage
		var payload, after string
		if err := dec.Decode(&raw); err == nil {
			payload = string(raw)
			tail := strings.TrimLeft(rest[dec.InputOffset():], " \t\r\n")
			if tail == "" {
				// The JSON value is complete and only its final tag is missing.
				after, found = "", true
			} else {
				after, found = strings.CutPrefix(tail, close)
			}
		} else {
			payload, after, found = strings.Cut(rest, close)
			payload = quoteInvalidStringEscapes(payload)
		}
		if !found {
			return "", nil, fmt.Errorf("%w: missing closing marker", errToolMarker)
		}
		envelopes := parseToolEnvelopes(payload)
		if len(envelopes) == 0 {
			return "", nil, fmt.Errorf("%w: could not decode call JSON", errToolMarker)
		}
		calls := repliesFromEnvelopes(envelopes, allowed)
		if len(calls) == 0 || len(calls) != len(envelopes) {
			return "", nil, fmt.Errorf("%w: tool name or arguments do not match the client protocol", errToolMarker)
		}
		replies = append(replies, calls...)
		text = after
	}
	if len(replies) == 0 {
		return clean.String(), nil, nil
	}
	return strings.TrimSpace(clean.String()), replies, nil
}

// dropEmptyFences removes a Markdown fence that only wrapped a tool marker.
// Fences around real content are preserved.
func dropEmptyFences(s string) string {
	t := strings.TrimSpace(s)
	if !strings.HasPrefix(t, "```") || !strings.HasSuffix(t, "```") || len(t) < 6 {
		return t
	}
	inner := t[3 : len(t)-3]
	if i := strings.IndexByte(inner, '\n'); i >= 0 {
		inner = inner[i+1:]
	}
	if hasVisibleText(inner) {
		return t
	}
	return ""
}

// extractFencedToolCalls accepts a tool call inside a Markdown fence anywhere
// in the message. Fences that do not contain one of the caller's tools are
// left untouched.
func extractFencedToolCalls(text string, allowed map[string]bool) (string, []ToolCallReply) {
	var clean strings.Builder
	var replies []ToolCallReply
	rest := text
	for {
		start := strings.Index(rest, "```")
		if start < 0 {
			clean.WriteString(rest)
			break
		}
		afterOpen := start + 3
		nl := strings.IndexByte(rest[afterOpen:], '\n')
		if nl < 0 {
			clean.WriteString(rest)
			break
		}
		bodyStart := afterOpen + nl + 1
		relEnd := strings.Index(rest[bodyStart:], "```")
		if relEnd < 0 {
			clean.WriteString(rest)
			break
		}
		bodyEnd := bodyStart + relEnd
		parsed := repliesFromEnvelopes(parseToolEnvelopes(strings.TrimSpace(rest[bodyStart:bodyEnd])), allowed)
		if len(parsed) == 0 {
			clean.WriteString(rest[:bodyEnd+3])
			rest = rest[bodyEnd+3:]
			continue
		}
		clean.WriteString(rest[:start])
		replies = append(replies, parsed...)
		rest = rest[bodyEnd+3:]
	}
	if len(replies) == 0 {
		return text, nil
	}
	return dropEmptyFences(strings.TrimSpace(clean.String())), replies
}

// extractBareToolCall accepts a whole-message JSON tool call, optionally
// wrapped in a Markdown fence.
func extractBareToolCall(text string, allowed map[string]bool) (string, []ToolCallReply) {
	trimmed := strings.TrimSpace(text)
	if strings.HasPrefix(trimmed, "```") {
		if i := strings.Index(trimmed, "\n"); i >= 0 {
			trimmed = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(trimmed[i+1:]), "```"))
		}
	}
	if !strings.HasPrefix(trimmed, "{") && !strings.HasPrefix(trimmed, "[") {
		return text, nil
	}
	replies := repliesFromEnvelopes(parseToolEnvelopes(trimmed), allowed)
	if len(replies) == 0 {
		return text, nil
	}
	return "", replies
}

// limitToolReplies enforces tool_choice/parallel_tool_calls limits.
func limitToolReplies(replies []ToolCallReply, mode, name string, parallel bool) []ToolCallReply {
	out := make([]ToolCallReply, 0, len(replies))
	for _, reply := range replies {
		if mode == "function" && name != "" && reply.Name != name {
			continue
		}
		out = append(out, reply)
		if !parallel {
			break
		}
	}
	return out
}

func replyToolCalls(replies []ToolCallReply) []ToolCall {
	out := make([]ToolCall, 0, len(replies))
	for _, r := range replies {
		out = append(out, ToolCall{
			ID:   r.ID,
			Type: "function",
			Function: FunctionCall{
				Name:      r.Name,
				Arguments: r.Arguments,
			},
		})
	}
	return out
}

// nativeToolReplies maps upstream-native tool events to client tool calls.
// Only names the caller declared are exposed.
func nativeToolReplies(calls []ToolCallState, allowed map[string]bool) []ToolCallReply {
	out := make([]ToolCallReply, 0, len(calls))
	for _, call := range calls {
		name := resolveToolName(call.Name, allowed)
		if name == "" {
			continue
		}
		id := strings.TrimSpace(call.ID)
		if id == "" {
			id = "call_" + strings.ReplaceAll(uuid(), "-", "")
		}
		args := strings.TrimSpace(call.Arguments)
		if args == "" || !json.Valid([]byte(args)) {
			args = "{}"
		}
		out = append(out, ToolCallReply{ID: id, Name: name, Arguments: args})
	}
	return out
}

func requestedToolNames(req *ChatRequest) map[string]bool {
	names := make(map[string]bool)
	for _, t := range req.Tools {
		if name := strings.TrimSpace(t.Function.Name); name != "" {
			names[name] = true
		}
	}
	for _, f := range req.Functions {
		if name := strings.TrimSpace(f.Name); name != "" {
			names[name] = true
		}
	}
	return names
}

func parseToolChoice(v any) (mode, name string) {
	switch x := v.(type) {
	case string:
		return strings.ToLower(strings.TrimSpace(x)), ""
	case map[string]any:
		if fn, ok := x["function"].(map[string]any); ok {
			if n, ok := fn["name"].(string); ok {
				return "function", strings.TrimSpace(n)
			}
		}
		if n, ok := x["name"].(string); ok {
			return "function", strings.TrimSpace(n)
		}
		if t, ok := x["type"].(string); ok {
			return strings.ToLower(strings.TrimSpace(t)), ""
		}
	}
	return "auto", ""
}
func resolveToolCalls(res ChatResult, allowed map[string]bool, mode, name string, parallel bool) (string, []ToolCall, error) {
	if mode == "none" || len(allowed) == 0 {
		return res.Text, nil, nil
	}
	text, replies, err := markedToolCalls(res.Text, allowed)
	if err != nil {
		return "", nil, err
	}
	if len(replies) == 0 {
		text, replies = extractFencedToolCalls(text, allowed)
	}
	if len(replies) == 0 {
		text, replies = extractBareToolCall(text, allowed)
	}
	if len(replies) == 0 {
		replies = nativeToolReplies(res.ToolCalls, allowed)
	}
	if len(replies) > 0 {
		text = dropEmptyFences(text)
	}
	return text, replyToolCalls(limitToolReplies(replies, mode, name, parallel)), nil
}

// A format-repair response can put one long string argument after a JSON
// header. Commands and file contents then need no additional escaping.
const rawArgumentRepair = `For a single call containing a command or file contents, use this exact alternative:
CLIENT_TOOL_TEXT
{"name":"actual tool name","text_argument":"actual string parameter name","arguments":{"other_parameter":"other value"}}
literal text parameter value, with original quotes and line breaks
END_CLIENT_TOOL_TEXT
The header occupies one line. Put all remaining parameters in arguments. Output nothing else. For multiple calls, use the normal protocol.`

func repairedToolCalls(res ChatResult, allowed map[string]bool, mode, name string, parallel bool) (string, []ToolCall, error) {
	body, raw := strings.CutPrefix(strings.TrimSpace(res.Text), "CLIENT_TOOL_TEXT\n")
	if !raw {
		return resolveToolCalls(res, allowed, mode, name, parallel)
	}
	header, body, ok := strings.Cut(body, "\n")
	value, closed := strings.CutSuffix(body, "\nEND_CLIENT_TOOL_TEXT")
	var call struct {
		Name  string                     `json:"name"`
		Field string                     `json:"text_argument"`
		Args  map[string]json.RawMessage `json:"arguments"`
	}
	if !ok || !closed || json.Unmarshal([]byte(header), &call) != nil || !allowed[call.Name] || call.Field == "" || call.Args == nil || call.Args[call.Field] != nil {
		return "", nil, errToolMarker
	}
	call.Args[call.Field], _ = json.Marshal(value)
	args, err := json.Marshal(call.Args)
	if err != nil {
		return "", nil, err
	}
	replies := repliesFromEnvelopes([]toolCallEnvelope{{Name: call.Name, Arguments: args}}, allowed)
	return "", replyToolCalls(replies), nil
}
