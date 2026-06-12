"""
LangGraph Agent 状态定义
所有节点共享的状态结构，贯穿整个 StateGraph 执行流程
"""
from typing import TypedDict, Annotated, Optional
from operator import add

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Agent 状态图的状态结构"""

    # ---- 输入 ----
    messages: list[BaseMessage]         # 完整对话：system prompt + 历史消息 + 当前用户消息
    user_id: str                        # 用户标识
    session_id: str                     # 会话标识

    # ---- 规划阶段 ----
    llm_message: Optional[BaseMessage]  # LLM 最新返回的 AIMessage（可能含 tool_calls）

    # ---- 工具执行阶段 ----
    tool_results: Annotated[list[str], add]   # 累积的工具执行结果
    tools_used: Annotated[list[str], add]     # 累积使用的工具名
    tool_round: int                           # 当前工具调用轮次（安全限制）

    # ---- 人机确认 ----
    pending_confirmation: Optional[dict]      # {"pending_id": str, "preview": str}

    # ---- 反思阶段 ----
    reflection_decision: str                  # "respond" | "retry"

    # ---- 最终输出 ----
    final_response: str                       # 最终回复文本
