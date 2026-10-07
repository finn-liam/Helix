import pytest
from pydantic import TypeAdapter, ValidationError

from helix.types import (
    AssistantMessage,
    Message,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
)

message_adapter = TypeAdapter(Message)

def test_assistant_interleaved_content_roundtrip():
    msg = AssistantMessage(
        content=[
            TextContent(text="我先看看目录里有什么"),
            ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"}),
            ToolCall(id="call_2", name="run_shell", arguments={"command": "python -V"}),
            TextContent(text="同时确认一下解释器版本"),
        ],
        model="deepseek-chat",
        usage=Usage(input=120, output=35, total=155),
        stop_reason=StopReason.TOOL_USE,
    )
    restored = AssistantMessage.model_validate_json(msg.model_dump_json())
    assert restored == msg

def test_union_dispatches_on_role():
    raw = {
        "role": "tool_result",
        "tool_call_id": "call_1",
        "tool_name": "read_file",
        "content": "file body here",
        "is_error": False,
    }
    msg = message_adapter.validate_python(raw)
    assert isinstance(msg, ToolResultMessage)
    assert msg.tool_call_id == "call_1"


def test_union_rejects_unknown_role():
    with pytest.raises(ValidationError):
        message_adapter.validate_python({"role": "wizard", "content": "hi"})


def test_tool_call_requires_id():
    with pytest.raises(ValidationError):
        ToolCall(name="read_file", arguments={})


def test_usage_total_auto_and_validate():
    auto = Usage(input=10, output=5)
    assert auto.total == 15
    with pytest.raises(ValidationError):
        Usage(input=10, output=5, total=999)


def test_stop_reason_full_set():
    for reason in StopReason:
        msg = AssistantMessage(model="deepseek-chat", stop_reason=reason)
        assert msg.stop_reason is reason