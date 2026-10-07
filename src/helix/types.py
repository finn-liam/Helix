from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


class StopReason(str,Enum):
    PENDING = "pending"   
    STOP = "stop"          
    TOOL_USE = "tool_use"  
    LENGTH = "length"    
    ERROR = "error"
    ABORTED = "aborted"

# 总的输入和输出
class Usage(BaseModel):
    input:int=0
    output: int=0
    total: int=0

    @model_validator(mode="after")
    def _fill_or_check_total(self) -> "Usage":
        if self.total == 0:
            self.total = self.input + self.output
        elif self.total != self.input+self.output:
            raise ValueError(
                f"usage total {self.total} != input {self.input} + output {self.output}"
            )
        return self
class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str,Any] = Field(default_factory=dict) #这个字段是一个字典，创建时如果没有，自动创建一个空字典

ContentBlock = Annotated[TextContent | ToolCall, Field(discriminator ="type")]

class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str
    timestamp: float | None = None

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str
    timestamp: float | None = None

class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock] = Field(default_factory=list)
    model: str
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = StopReason.PENDING
    timestamp: float | None = None

class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"  
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    timestamp: float | None = None

Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator ="role")
]