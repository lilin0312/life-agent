"""
LangGraph StateGraph 构建 — Agent 编排核心

流程：
  START → planning → (有工具?) ─YES→ tool_execution → planning (循环)
                     └──NO──→ reflection ──→ (重试?) ─YES→ planning
                                                   └──NO──→ response → END
"""
import logging
from functools import partial
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.agent_state import AgentState
from app.agent_nodes import (
    planning_node,
    tool_execution_node,
    reflection_node,
    response_node,
    MAX_TOOL_ROUNDS,
)
from app.tool_schemas import TOOL_DEFINITIONS
from app.tool_service import ToolService

logger = logging.getLogger(__name__)


class AgentGraph:
    """LangGraph Agent 编排器"""

    def __init__(self, llm: Any, tool_service: ToolService, zhipu_llm: Any = None):
        """
        Args:
            llm: 主 ChatOpenAI 实例（千问 qwen-plus，用于规划 + 回答）
            tool_service: ToolService 实例
            zhipu_llm: 智谱 ChatOpenAI 实例（glm-4-flash，用于反思）；为 None 时回退到主模型
        """
        self.llm = llm
        self.zhipu_llm = zhipu_llm or llm  # 未配置智谱时回退到千问
        self.llm_with_tools = llm.bind_tools(TOOL_DEFINITIONS) if llm else None
        self.tool_service = tool_service
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建 StateGraph"""
        builder = StateGraph(AgentState)

        # ---- 添加节点 ----
        # 使用 partial 注入依赖（llm 和 tool_service）
        builder.add_node(
            "planning",
            partial(planning_node, llm_with_tools=self.llm_with_tools),
        )
        builder.add_node(
            "tool_execution",
            partial(tool_execution_node, tool_service=self.tool_service),
        )
        builder.add_node(
            "reflection",
            partial(reflection_node, llm=self.zhipu_llm),  # 智谱 GLM 负责反思
        )
        builder.add_node(
            "response",
            partial(response_node, llm=self.llm),
        )

        # ---- 添加边 ----
        builder.add_edge(START, "planning")

        # planning 后：有工具调用 → tool_execution，无 → reflection
        builder.add_conditional_edges("planning", self._route_after_planning, {
            "tool_execution": "tool_execution",
            "reflection": "reflection",
            "response": "response",
        })

        # tool_execution 后：简单工具→直接回答，否则按轮次决定
        builder.add_conditional_edges("tool_execution", self._route_after_tools, {
            "planning": "planning",
            "reflection": "reflection",
            "response": "response",
        })

        # reflection 后：重试 → planning，否则 → response
        builder.add_conditional_edges("reflection", self._route_after_reflection, {
            "planning": "planning",
            "response": "response",
        })

        builder.add_edge("response", END)

        # 使用内存级 checkpointer
        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)

    # ---- 路由函数 ----

    # 全局工具调用硬上限
    MAX_TOTAL_TOOLS = 4

    def _route_after_planning(self, state: AgentState) -> str:
        """规划后路由：无工具直接回答，有工具去执行"""
        llm_message = state.get("llm_message")
        if isinstance(llm_message, AIMessage):
            tool_calls = getattr(llm_message, "tool_calls", None)
            if tool_calls:
                # 全局硬上限检查
                total_used = len(state.get("tools_used", []))
                if total_used >= self.MAX_TOTAL_TOOLS:
                    logger.warning(f"[路由] 全局工具上限({self.MAX_TOTAL_TOOLS})已达，强制回答")
                    return "response"
                return "tool_execution"
        # 无工具调用，直接回答（跳过反思）
        return "response"

    # 简单工具：结果清晰，不需要反思质检
    SIMPLE_TOOLS = {
        "generate_image", "get_weather", "get_current_time",
        "calculator", "translate", "web_search",
    }

    def _route_after_tools(self, state: AgentState) -> str:
        """工具执行后路由：简单工具直接回答，复杂工具走反思"""
        tool_round = state.get("tool_round", 0)
        tools_used = state.get("tools_used", [])

        # 简单工具（图片生成、天气、时间等）结果清晰，跳过反思直接回答
        if tools_used and all(t in self.SIMPLE_TOOLS for t in tools_used):
            logger.info(f"[路由] 简单工具 {tools_used}，跳过反思直接回答")
            return "response"

        if tool_round < MAX_TOOL_ROUNDS:
            return "planning"  # 回到 planning 让 LLM 决定
        return "reflection"

    def _route_after_reflection(self, state: AgentState) -> str:
        """反思后路由：重试还是直接回答"""
        decision = state.get("reflection_decision", "respond")
        if decision == "retry":
            return "planning"
        return "response"

    # ---- 对外接口 ----

    async def run(
        self,
        messages: list[dict],
        user_id: str,
        session_id: str,
    ) -> dict:
        """
        运行 Agent 状态图

        Args:
            messages: 完整消息列表（已包含 system prompt + history）
            user_id: 用户 ID
            session_id: 会话 ID

        Returns:
            包含 final_response, tools_used, pending_confirmation 的字典
        """
        if not self.llm_with_tools:
            return {
                "final_response": "系统尚未配置 API Key，请联系管理员。",
                "tools_used": [],
                "tool_results": [],
                "pending_confirmation": None,
            }

        # 将 dict 消息转为 LangChain Message 对象
        from langchain_core.messages import HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        initial_state = {
            "messages": lc_messages,
            "user_id": user_id,
            "session_id": session_id,
            "llm_message": None,
            "tool_results": [],
            "tools_used": [],
            "tool_round": 0,
            "pending_confirmation": None,
            "reflection_decision": "",
            "final_response": "",
        }

        try:
            result = await self.graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": session_id}},
            )
            return result
        except Exception as e:
            logger.error(f"[AgentGraph] 执行失败: {e}", exc_info=True)
            return {
                "final_response": f"抱歉，Agent 执行出错: {e}",
                "tools_used": [],
                "tool_results": [],
                "pending_confirmation": None,
            }
