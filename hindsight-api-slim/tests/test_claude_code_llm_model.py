"""Regression test: the Claude Code provider must forward its configured model.

With ``HINDSIGHT_API_*_LLM_PROVIDER=claude-code``, the model resolved by config
(global, per-scope, or the provider default) is stored on the provider as
``self.model`` and reported in metrics/traces — but it was never placed on the
``ClaudeAgentOptions`` handed to the Claude Agent SDK, so the spawned ``claude``
CLI silently ran its own default model instead (issue #2881).

``ClaudeAgentOptions.model`` is the only field the SDK transport turns into the
CLI's ``--model`` flag. These tests mock the SDK to capture the options passed
at call time and assert the provider's configured model rides along on BOTH code
paths (``call`` and ``call_with_tools``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

# A model that is deliberately NOT the claude-code provider default
# (``claude-sonnet-4-5``), so a hardcoded/leaked default cannot make the
# assertion pass by accident.
TEST_MODEL = "claude-haiku-4-5"


@dataclass
class _FakeOptions:
    """Stand-in for ClaudeAgentOptions; captures kwargs without importing SDK."""

    model: str | None = None
    system_prompt: str | None = None
    max_turns: int | None = None
    allowed_tools: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    mcp_servers: dict[str, Any] = field(default_factory=dict)


class _FakeAssistantMessage:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


def _instantiate_provider(model: str = TEST_MODEL):
    from hindsight_api.engine.providers.claude_code_llm import ClaudeCodeLLM

    return ClaudeCodeLLM(
        provider="claude-code",
        api_key="",
        base_url="",
        model=model,
        reasoning_effort="low",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        TEST_MODEL,  # undated full name
        "sonnet",  # short alias
        "claude-sonnet-4-5",  # the new provider default
    ],
)
async def test_call_passes_configured_model_to_sdk_options(monkeypatch, model):
    """call() must put the provider's configured model on ClaudeAgentOptions."""
    import claude_agent_sdk

    captured: dict[str, _FakeOptions] = {}

    async def fake_query(prompt: str, options: _FakeOptions):
        captured["options"] = options
        yield _FakeAssistantMessage(content=[_FakeTextBlock(text="ok")])

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    provider = _instantiate_provider(model=model)
    result = await provider.call(
        messages=[{"role": "user", "content": "hi"}],
        max_retries=0,
        scope="test",
    )

    assert result == "ok"
    assert "options" in captured, "fake query was not called"
    assert captured["options"].model == model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        TEST_MODEL,
        "sonnet",
        "claude-sonnet-4-5",
    ],
)
async def test_call_with_tools_passes_configured_model_to_sdk_options(monkeypatch, model):
    """call_with_tools() must put the provider's configured model on ClaudeAgentOptions."""
    import claude_agent_sdk

    captured: dict[str, _FakeOptions] = {}

    class _FakeClient:
        def __init__(self, options: _FakeOptions) -> None:
            captured["options"] = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self):
            yield _FakeAssistantMessage(content=[_FakeTextBlock(text="ok")])

    @dataclass
    class _FakeSdkMcpTool:
        name: str
        description: str
        input_schema: dict[str, Any]
        handler: Any

    def fake_create_sdk_mcp_server(name: str, version: str, tools=None):
        return {"name": name, "version": version, "tools": tools}

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "ToolUseBlock", type("ToolUseBlock", (), {}))
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _FakeClient)
    monkeypatch.setattr(claude_agent_sdk, "SdkMcpTool", _FakeSdkMcpTool)
    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", fake_create_sdk_mcp_server)

    provider = _instantiate_provider(model=model)
    result = await provider.call_with_tools(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "function": {
                    "name": "noop",
                    "description": "no-op",
                    "parameters": {"type": "object", "properties": {}},
                }
            }
        ],
        max_retries=0,
        scope="test",
    )

    assert result.content == "ok"
    assert "options" in captured, "fake client was not constructed"
    assert captured["options"].model == model
