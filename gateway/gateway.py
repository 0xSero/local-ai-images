#!/usr/bin/env python3
"""Omarchy Local AI gateway: one endpoint, three dialects, one engine.

Listens on GATEWAY_PORT (default 12434) and forwards to the engine at
UPSTREAM (default http://engine:8000), which speaks OpenAI chat completions.

  /v1/models, /v1/chat/completions   pass through unchanged (streaming too)
  /v1/messages                       Anthropic Messages  -> chat completions
  /v1/messages/count_tokens          rough estimate, Anthropic shape
  /v1/responses                      OpenAI Responses    -> chat completions
  /health                            200 when the engine answers /v1/models

Auth: when GATEWAY_KEY_FILE names a non-empty file, every request must carry
its value as `Authorization: Bearer <key>` or `x-api-key: <key>`. The file is
read on each request, so the plugin can rotate the key without a restart.

Standard library only. No configuration beyond the environment above.
"""

import http.client
import json
import os
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("UPSTREAM", "http://engine:8000").rstrip("/")
PORT = int(os.environ.get("GATEWAY_PORT", "12434"))
KEY_FILE = os.environ.get("GATEWAY_KEY_FILE", "")
MODEL_ALIAS = os.environ.get("MODEL", "")  # served model id; client model names are mapped onto it
TIMEOUT = int(os.environ.get("UPSTREAM_TIMEOUT", "600"))


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, file=sys.stderr, flush=True)


def required_key():
    if not KEY_FILE:
        return ""
    try:
        with open(KEY_FILE) as handle:
            return handle.read().strip()
    except OSError:
        return ""


# ----------------------------------------------------------------------------- upstream
class Upstream:
    """One HTTP request to the engine. `stream()` yields SSE data payloads as parsed JSON."""

    def __init__(self, method, path, body=None):
        parsed = urllib.parse.urlparse(UPSTREAM)
        self.conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=TIMEOUT)
        headers = {"Content-Type": "application/json", "Accept": "*/*"}
        data = json.dumps(body).encode() if body is not None else None
        self.conn.request(method, path, body=data, headers=headers)
        self.response = self.conn.getresponse()
        self.status = self.response.status

    def json(self):
        raw = self.response.read()
        self.conn.close()
        return json.loads(raw) if raw else {}

    def raw(self):
        raw = self.response.read()
        self.conn.close()
        return raw

    def stream(self):
        try:
            buffer = b""
            while True:
                chunk = self.response.read1(65536) if hasattr(self.response, "read1") else self.response.read(65536)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        return
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        continue
        finally:
            self.conn.close()


def upstream_model(requested):
    return MODEL_ALIAS or requested


def scrub_schema(schema):
    """Tool parameter schemas as the engine can take them. llama.cpp compiles them to a grammar and rejects
    regex escapes it does not know (Claude Code ships a pattern with \\- ); `pattern` and `format` are
    validation hints the model does not need to call the tool, so they are dropped at every depth."""
    if isinstance(schema, dict):
        return {k: scrub_schema(v) for k, v in schema.items() if k not in ("pattern", "format")}
    if isinstance(schema, list):
        return [scrub_schema(v) for v in schema]
    return schema


def hoist_system(messages):
    """One leading system message. Chat templates such as Qwen's refuse a system message anywhere
    else, and clients send them mid-conversation (Codex developer items, Claude's per-turn system)."""
    system = [m.get("content") or "" for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if not system:
        return rest
    return [{"role": "system", "content": "\n\n".join(c for c in system if c)}] + rest


# ----------------------------------------------------------------------------- Anthropic Messages -> chat
def anthropic_to_chat(req):
    messages = []
    system = req.get("system")
    if isinstance(system, list):
        system = "\n".join(block.get("text", "") for block in system if isinstance(block, dict))
    if system:
        messages.append({"role": "system", "content": system})
    for message in req.get("messages", []):
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        text_parts, tool_calls, tool_results = [], [], []
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                text_parts.append(block.get("text", ""))
            elif kind == "tool_use":
                tool_calls.append({"id": block.get("id"), "type": "function",
                                   "function": {"name": block.get("name"), "arguments": json.dumps(block.get("input") or {})}})
            elif kind == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    inner = "\n".join(b.get("text", "") for b in inner if isinstance(b, dict) and b.get("type") == "text")
                tool_results.append({"role": "tool", "tool_call_id": block.get("tool_use_id"), "content": inner if isinstance(inner, str) else json.dumps(inner)})
            elif kind == "image":
                text_parts.append("[image omitted]")
        if role == "assistant":
            entry = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
        else:
            if tool_results:
                messages.extend(tool_results)
            if text_parts:
                messages.append({"role": "user", "content": "\n".join(text_parts)})
    out = {"model": upstream_model(req.get("model")), "messages": hoist_system(messages), "stream": bool(req.get("stream"))}
    if req.get("max_tokens"):
        out["max_tokens"] = req["max_tokens"]
    for key in ("temperature", "top_p"):
        if key in req:
            out[key] = req[key]
    if req.get("stop_sequences"):
        out["stop"] = req["stop_sequences"]
    tools = req.get("tools") or []
    if tools:
        out["tools"] = [{"type": "function", "function": {"name": t.get("name"), "description": t.get("description", ""),
                                                          "parameters": scrub_schema(t.get("input_schema") or {"type": "object", "properties": {}})}}
                        for t in tools if t.get("name")]
        choice = req.get("tool_choice") or {}
        kind = choice.get("type")
        if kind == "any":
            out["tool_choice"] = "required"
        elif kind == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": choice.get("name")}}
        elif kind == "none":
            out["tool_choice"] = "none"
        else:
            out["tool_choice"] = "auto"
    return out


def anthropic_stop(finish_reason, had_tools):
    if had_tools or finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def chat_to_anthropic(resp, model):
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = []
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    tool_calls = message.get("tool_calls") or []
    for call in tool_calls:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {"_raw": function.get("arguments")}
        content.append({"type": "tool_use", "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": function.get("name"), "input": arguments})
    usage = resp.get("usage") or {}
    return {
        "id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant", "model": model,
        "content": content, "stop_reason": anthropic_stop(choice.get("finish_reason"), bool(tool_calls)), "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
    }


def anthropic_stream(handler, upstream, model):
    """Translate chat SSE deltas into the Anthropic event stream."""
    def event(name, data):
        handler.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
        handler.wfile.flush()

    message_id = "msg_" + uuid.uuid4().hex[:24]
    event("message_start", {"type": "message_start", "message": {
        "id": message_id, "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})
    index = -1
    text_open = False
    tools = {}  # chat tool_call index -> anthropic block index
    finish = None
    usage = {}
    for chunk in upstream.stream():
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                if not text_open:
                    index += 1
                    text_open = True
                    event("content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}})
                event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": delta["content"]}})
            for call in delta.get("tool_calls") or []:
                position = call.get("index", 0)
                if position not in tools:
                    if text_open:
                        event("content_block_stop", {"type": "content_block_stop", "index": index})
                        text_open = False
                    index += 1
                    tools[position] = index
                    function = call.get("function") or {}
                    event("content_block_start", {"type": "content_block_start", "index": index, "content_block": {
                        "type": "tool_use", "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}", "name": function.get("name") or "", "input": {}}})
                arguments = (call.get("function") or {}).get("arguments")
                if arguments:
                    event("content_block_delta", {"type": "content_block_delta", "index": tools[position], "delta": {"type": "input_json_delta", "partial_json": arguments}})
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    if text_open:
        event("content_block_stop", {"type": "content_block_stop", "index": index})
    for block_index in tools.values():
        event("content_block_stop", {"type": "content_block_stop", "index": block_index})
    event("message_delta", {"type": "message_delta", "delta": {"stop_reason": anthropic_stop(finish, bool(tools)), "stop_sequence": None},
                            "usage": {"output_tokens": usage.get("completion_tokens", 0)}})
    event("message_stop", {"type": "message_stop"})


# ----------------------------------------------------------------------------- OpenAI Responses -> chat
def responses_to_chat(req):
    messages = []
    if req.get("instructions"):
        messages.append({"role": "system", "content": req["instructions"]})
    items = req.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    for item in items or []:
        kind = item.get("type") or ("message" if "role" in item else "")
        if kind == "message":
            role = item.get("role", "user")
            content = item.get("content")
            if isinstance(content, list):
                content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text"))
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": content or ""})
        elif kind == "function_call":
            call = {"id": item.get("call_id") or item.get("id"), "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"}}
            if messages and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                messages[-1]["tool_calls"].append(call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind == "function_call_output":
            output = item.get("output")
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": output if isinstance(output, str) else json.dumps(output)})
        elif kind == "reasoning":
            continue
    out = {"model": upstream_model(req.get("model")), "messages": hoist_system(messages), "stream": bool(req.get("stream"))}
    if req.get("max_output_tokens"):
        out["max_tokens"] = req["max_output_tokens"]
    for key in ("temperature", "top_p"):
        if key in req and req[key] is not None:
            out[key] = req[key]
    tools = [t for t in req.get("tools") or [] if t.get("type") == "function"]
    if tools:
        out["tools"] = [{"type": "function", "function": {"name": t.get("name"), "description": t.get("description") or "",
                                                          "parameters": scrub_schema(t.get("parameters") or {"type": "object", "properties": {}})}} for t in tools]
        choice = req.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") == "function":
            out["tool_choice"] = {"type": "function", "function": {"name": choice.get("name")}}
        elif choice in ("required", "none", "auto"):
            out["tool_choice"] = choice
        else:
            out["tool_choice"] = "auto"
    if req.get("parallel_tool_calls") is False:
        out["parallel_tool_calls"] = False
    return out


def responses_envelope(model, req):
    return {"id": "resp_" + uuid.uuid4().hex[:24], "object": "response", "created_at": int(time.time()), "model": model,
            "status": "in_progress", "output": [], "error": None, "incomplete_details": None,
            "instructions": req.get("instructions"), "parallel_tool_calls": req.get("parallel_tool_calls", True),
            "tool_choice": req.get("tool_choice", "auto"), "tools": req.get("tools") or [], "usage": None, "metadata": req.get("metadata") or {}}


def chat_to_responses(resp, model, req):
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output = []
    if message.get("content"):
        output.append({"type": "message", "id": "msg_" + uuid.uuid4().hex[:24], "status": "completed", "role": "assistant",
                       "content": [{"type": "output_text", "text": message["content"], "annotations": []}]})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        output.append({"type": "function_call", "id": "fc_" + uuid.uuid4().hex[:24], "call_id": call.get("id") or "call_" + uuid.uuid4().hex[:24],
                       "name": function.get("name"), "arguments": function.get("arguments") or "{}", "status": "completed"})
    usage = resp.get("usage") or {}
    envelope = responses_envelope(model, req)
    envelope.update({"status": "completed", "output": output,
                     "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
                               "total_tokens": usage.get("total_tokens", 0), "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}})
    return envelope


def responses_stream(handler, upstream, model, req):
    """Translate chat SSE deltas into the Responses event stream Codex consumes."""
    sequence = [0]

    def event(name, data):
        sequence[0] += 1
        data = dict(data, type=name, sequence_number=sequence[0])
        handler.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
        handler.wfile.flush()

    envelope = responses_envelope(model, req)
    event("response.created", {"response": envelope})
    event("response.in_progress", {"response": envelope})
    output = []
    output_index = -1
    text_item = None  # (index, id, accumulated text)
    calls = {}  # chat index -> {"index", "id", "call_id", "name", "arguments"}
    usage = {}
    for chunk in upstream.stream():
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                if text_item is None:
                    output_index += 1
                    text_item = [output_index, "msg_" + uuid.uuid4().hex[:24], ""]
                    item = {"type": "message", "id": text_item[1], "status": "in_progress", "role": "assistant", "content": []}
                    event("response.output_item.added", {"output_index": output_index, "item": item})
                    event("response.content_part.added", {"item_id": text_item[1], "output_index": output_index, "content_index": 0,
                                                          "part": {"type": "output_text", "text": "", "annotations": []}})
                text_item[2] += delta["content"]
                event("response.output_text.delta", {"item_id": text_item[1], "output_index": text_item[0], "content_index": 0, "delta": delta["content"]})
            for call in delta.get("tool_calls") or []:
                position = call.get("index", 0)
                if position not in calls:
                    if text_item is not None:
                        finish_text(event, text_item, output)
                        text_item = None
                    output_index += 1
                    function = call.get("function") or {}
                    calls[position] = {"index": output_index, "id": "fc_" + uuid.uuid4().hex[:24],
                                       "call_id": call.get("id") or "call_" + uuid.uuid4().hex[:24], "name": function.get("name") or "", "arguments": ""}
                    item = {"type": "function_call", "id": calls[position]["id"], "call_id": calls[position]["call_id"],
                            "name": calls[position]["name"], "arguments": "", "status": "in_progress"}
                    event("response.output_item.added", {"output_index": output_index, "item": item})
                function = call.get("function") or {}
                if function.get("name") and not calls[position]["name"]:
                    calls[position]["name"] = function["name"]
                if function.get("arguments"):
                    calls[position]["arguments"] += function["arguments"]
                    event("response.function_call_arguments.delta", {"item_id": calls[position]["id"], "output_index": calls[position]["index"], "delta": function["arguments"]})
    if text_item is not None:
        finish_text(event, text_item, output)
    for call in calls.values():
        event("response.function_call_arguments.done", {"item_id": call["id"], "output_index": call["index"], "arguments": call["arguments"]})
        item = {"type": "function_call", "id": call["id"], "call_id": call["call_id"], "name": call["name"], "arguments": call["arguments"], "status": "completed"}
        output.append((call["index"], item))
        event("response.output_item.done", {"output_index": call["index"], "item": item})
    envelope.update({"status": "completed", "output": [item for _, item in sorted(output, key=lambda pair: pair[0])],
                     "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
                               "total_tokens": usage.get("total_tokens", 0), "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}})
    event("response.completed", {"response": envelope})


def finish_text(event, text_item, output):
    index, item_id, text = text_item
    event("response.output_text.done", {"item_id": item_id, "output_index": index, "content_index": 0, "text": text})
    part = {"type": "output_text", "text": text, "annotations": []}
    event("response.content_part.done", {"item_id": item_id, "output_index": index, "content_index": 0, "part": part})
    item = {"type": "message", "id": item_id, "status": "completed", "role": "assistant", "content": [part]}
    output.append((index, item))
    event("response.output_item.done", {"output_index": index, "item": item})


# ----------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "omarchy-local-ai-gateway/1"

    def log_message(self, fmt, *args):  # quiet by default; errors go through log()
        pass

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def authorized(self):
        key = required_key()
        if not key:
            return True
        auth = self.headers.get("Authorization", "")
        given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "").strip()
        return given == key

    def read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw) if raw else {}

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/health":
            try:
                up = Upstream("GET", "/v1/models")
                up.raw()
                self.send_json(200 if up.status == 200 else 503, {"status": "ok" if up.status == 200 else "engine unavailable"})
            except Exception as error:
                self.send_json(503, {"status": f"engine unavailable: {error}"})
            return
        if not self.authorized():
            return self.send_json(401, {"error": {"type": "authentication_error", "message": "invalid or missing API key"}})
        if path in ("/v1/models", "/models"):
            return self.passthrough("GET", "/v1/models", None)
        self.send_json(404, {"error": {"type": "not_found", "message": path}})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if not self.authorized():
            return self.send_json(401, {"error": {"type": "authentication_error", "message": "invalid or missing API key"}})
        try:
            body = self.read_body()
        except json.JSONDecodeError:
            return self.send_json(400, {"error": {"type": "invalid_request_error", "message": "body is not JSON"}})
        try:
            if path in ("/v1/chat/completions", "/chat/completions"):
                if isinstance(body.get("messages"), list):
                    body["messages"] = hoist_system(body["messages"])
                if isinstance(body.get("tools"), list):
                    body["tools"] = scrub_schema(body["tools"])
                if MODEL_ALIAS:
                    body["model"] = MODEL_ALIAS
                return self.passthrough("POST", "/v1/chat/completions", body)
            if path in ("/v1/messages", "/messages"):
                return self.handle_messages(body)
            if path in ("/v1/messages/count_tokens", "/messages/count_tokens"):
                text = json.dumps(body.get("messages", [])) + json.dumps(body.get("system", ""))
                return self.send_json(200, {"input_tokens": max(1, len(text) // 4)})
            if path in ("/v1/responses", "/responses"):
                return self.handle_responses(body)
            if path in ("/v1/completions", "/completions"):
                return self.passthrough("POST", "/v1/completions", body)
        except (ConnectionError, OSError, http.client.HTTPException) as error:
            log(f"upstream error on {path}: {error}")
            return self.send_json(502, {"error": {"type": "api_error", "message": f"engine unavailable: {error}"}})
        self.send_json(404, {"error": {"type": "not_found", "message": path}})

    def passthrough(self, method, path, body):
        up = Upstream(method, path, body)
        if body and body.get("stream"):
            self.start_sse()
            try:
                # copy the engine's SSE bytes verbatim
                while True:
                    chunk = up.response.read1(65536) if hasattr(up.response, "read1") else up.response.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            finally:
                up.conn.close()
            return
        raw = up.raw()
        self.send_response(up.status)
        self.send_header("Content-Type", up.response.getheader("Content-Type") or "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def handle_messages(self, body):
        model = body.get("model") or MODEL_ALIAS
        chat = anthropic_to_chat(body)
        if chat.get("stream"):
            chat["stream_options"] = {"include_usage": True}
            up = Upstream("POST", "/v1/chat/completions", chat)
            if up.status != 200:
                return self.send_json(up.status, {"error": {"type": "api_error", "message": up.raw().decode(errors="replace")[:500]}})
            self.start_sse()
            anthropic_stream(self, up, model)
            return
        up = Upstream("POST", "/v1/chat/completions", chat)
        if up.status != 200:
            return self.send_json(up.status, {"error": {"type": "api_error", "message": up.raw().decode(errors="replace")[:500]}})
        self.send_json(200, chat_to_anthropic(up.json(), model))

    def handle_responses(self, body):
        model = body.get("model") or MODEL_ALIAS
        chat = responses_to_chat(body)
        if chat.get("stream"):
            chat["stream_options"] = {"include_usage": True}
            up = Upstream("POST", "/v1/chat/completions", chat)
            if up.status != 200:
                return self.send_json(up.status, {"error": {"type": "server_error", "message": up.raw().decode(errors="replace")[:500]}})
            self.start_sse()
            responses_stream(self, up, model, body)
            return
        up = Upstream("POST", "/v1/chat/completions", chat)
        if up.status != 200:
            return self.send_json(up.status, {"error": {"type": "server_error", "message": up.raw().decode(errors="replace")[:500]}})
        self.send_json(200, chat_to_responses(up.json(), model, body))


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    log(f"gateway listening on :{PORT}, engine {UPSTREAM}, key {'required' if required_key() else 'not set'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
