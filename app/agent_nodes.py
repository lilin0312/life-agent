"""
LangGraph 节点函数 — 规划 / 工具执行 / 反思 / 回复
每个节点接收 AgentState，返回部分状态更新
"""
import json
import logging
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from app.agent_state import AgentState
from app.tool_schemas import TOOL_DEFINITIONS
from app.tool_service import ToolService

logger = logging.getLogger(__name__)

# 工具调用最大轮次
MAX_TOOL_ROUNDS = 2
# 反思重试最大次数
MAX_REFLECTION_RETRIES = 1


def _parse_dsml_tool_calls(text: str) -> list[dict]:
    """
    解析 DeepSeek DSML 格式的工具调用（兜底方案）
    当模型输出 <｜｜DSML｜｜tool_calls> 原始文本而非标准 tool_calls 时使用
    返回 [{"name": ..., "args": {...}, "id": ...}] 列表
    """
    tool_calls = []

    # 匹配 <｜｜DSML｜｜invoke name="xxx"> ... <｜｜DSML｜｜parameter name="xxx" string="true">value
    invoke_pattern = re.findall(
        r'<\|[^>]*DSML[^>]*\|invoke\s+name="([^"]+)">(.*?)</\|[^>]*DSML[^>]*\|invoke>',
        text,
        re.DOTALL,
    )

    for tool_name, body in invoke_pattern:
        args = {}
        # 提取所有参数
        param_pattern = re.findall(
            r'<\|[^>]*DSML[^>]*\|parameter\s+name="([^"]+)"[^>]*>(.*?)(?=<(?:\|[^>]*DSML|$))',
            body,
            re.DOTALL,
        )
        for param_name, param_value in param_pattern:
            param_value = param_value.strip()
            # 尝试解析为 JSON（数字/布尔/对象），失败则作为字符串
            try:
                args[param_name] = json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                args[param_name] = param_value

        tool_calls.append({
            "name": tool_name,
            "args": args,
            "id": f"dsml_{uuid.uuid4().hex[:8]}",
        })

    return tool_calls


async def planning_node(state: AgentState, *, llm_with_tools: Any) -> dict:
    """
    规划节点：调用 LLM 分析用户意图，决定是否需要工具
    使用 bind_tools() 让 LLM 通过原生 function calling 返回结构化工具调用
    """
    messages = state["messages"]

    try:
        ai_message = await llm_with_tools.ainvoke(messages)
    except Exception as e:
        logger.error(f"[planning] LLM 调用失败: {e}")
        return {
            "llm_message": AIMessage(content=f"抱歉，服务暂时不可用（{type(e).__name__}）"),
            "final_response": f"抱歉，服务暂时不可用（{type(e).__name__}）",
            "reflection_decision": "respond",
        }

    # 记录工具调用情况
    tool_calls = getattr(ai_message, "tool_calls", None)
    if tool_calls:
        logger.info(f"[planning] LLM 请求调用工具: {[tc['name'] for tc in tool_calls]}")
    else:
        # 兜底：检查是否为 DeepSeek DSML 格式的文本输出
        dsml_calls = _parse_dsml_tool_calls(ai_message.content or "")
        if dsml_calls:
            logger.info(f"[planning] 检测到 DSML 格式工具调用: {[tc['name'] for tc in dsml_calls]}")
            ai_message = AIMessage(
                content="",
                tool_calls=dsml_calls,
                id=ai_message.id,
            )
        else:
            logger.info("[planning] LLM 直接回复（无工具调用）")

    return {"llm_message": ai_message}


async def tool_execution_node(state: AgentState, *, tool_service: ToolService) -> dict:
    """
    工具执行节点：执行 LLM 返回的 tool_calls
    - 普通工具：直接执行
    - 危险工具：走人机确认流程（pending action）
    """
    llm_message = state.get("llm_message")
    if not llm_message:
        return {}

    tool_calls = getattr(llm_message, "tool_calls", [])
    if not tool_calls:
        return {}

    # 安全限制：单轮最多执行 3 个工具调用，防止 LLM 疯狂调用
    MAX_CALLS_PER_ROUND = 3
    if len(tool_calls) > MAX_CALLS_PER_ROUND:
        logger.warning(f"[tool_execution] 工具调用过多({len(tool_calls)}个)，截断为{MAX_CALLS_PER_ROUND}个")
        tool_calls = tool_calls[:MAX_CALLS_PER_ROUND]

    user_id = state["user_id"]
    tool_results = []
    tools_used = []
    pending_confirmation = None
    new_messages = [llm_message]  # 先把 AIMessage 加入

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc.get("args", {})
        tool_call_id = tc.get("id", tool_name)

        logger.info(f"[tool_execution] 执行工具: {tool_name}({tool_args})")

        try:
            result = await tool_service.execute(tool_name, tool_args, user_id)
            content = result["content"]
            tools_used.append(tool_name)

            # 检查是否需要人机确认
            if result.get("need_confirm"):
                pending_confirmation = {
                    "pending_id": result["pending_id"],
                    "preview": result["content"],
                }
                logger.info(f"[tool_execution] 危险操作等待确认: {tool_name}")
        except Exception as e:
            content = f"工具执行出错: {e}"
            logger.error(f"[tool_execution] 工具异常 [{tool_name}]: {e}")

        tool_results.append(content)

        # 构建 ToolMessage 追加到对话中
        new_messages.append(
            ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)
        )

    current_round = state.get("tool_round", 0) + 1
    logger.info(f"[tool_execution] 第{current_round}轮完成，已执行: {tools_used}")

    return {
        "messages": new_messages,
        "tool_results": tool_results,
        "tools_used": tools_used,
        "tool_round": current_round,
        "pending_confirmation": pending_confirmation,
    }


async def reflection_node(state: AgentState, *, llm: Any) -> dict:
    """
    反思节点：让 LLM 自查工具结果是否充分、数据是否一致
    如果没用过工具，直接跳过反思 → respond
    """
    tools_used = state.get("tools_used", [])
    tool_round = state.get("tool_round", 0)

    # 没用过工具，直接回答
    if not tools_used:
        return {"reflection_decision": "respond"}

    # 超过重试上限，强制回答
    if tool_round > MAX_REFLECTION_RETRIES:
        logger.info("[reflection] 超过重试上限，强制回答")
        return {"reflection_decision": "respond"}

    # 构建反思 prompt
    reflection_prompt = f"""请回顾以上对话和工具执行结果，回答以下问题：

1. 工具返回的数据是否充分回答了用户的问题？
2. 数据之间是否存在矛盾或不一致？
3. 是否需要调用额外的工具来补充信息？

请用以下 JSON 格式回答（不要输出其他内容）：
{{"decision": "respond", "reasoning": "你的分析"}}
或
{{"decision": "retry", "reasoning": "需要重试的原因和需要的工具"}}

只输出 JSON，不要其他文字。"""

    messages = state["messages"] + [{"role": "user", "content": reflection_prompt}]

    try:
        response = await llm.ainvoke(messages)
        text = response.content.strip()

        # 尝试解析 JSON
        import json
        # 提取 JSON 部分
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        text = text.strip("` \n")

        result = json.loads(text)
        decision = result.get("decision", "respond")
        reasoning = result.get("reasoning", "")

        if decision == "retry" and tool_round < MAX_REFLECTION_RETRIES:
            logger.info(f"[reflection] 决定重试: {reasoning}")
            # 追加重试上下文
            retry_msg = {
                "role": "user",
                "content": f"【反思结果】之前的工具结果可能不充分：{reasoning}\n请尝试用不同方式或额外工具来完善回答。",
            }
            return {
                "reflection_decision": "retry",
                "messages": [response, retry_msg],
            }
        else:
            logger.info(f"[reflection] 决定回答: {reasoning}")
            return {"reflection_decision": "respond"}

    except Exception as e:
        logger.warning(f"[reflection] 反思解析失败，默认回答: {e}")
        return {"reflection_decision": "respond"}


async def response_node(state: AgentState, *, llm: Any) -> dict:
    """
    回复节点：生成最终精炼回复
    基于全部对话和工具结果，生成用户友好的回答
    """
    messages = state["messages"]
    tools_used = state.get("tools_used", [])
    tool_results = state.get("tool_results", [])

    # 如果用过工具，追加验证指令
    if tools_used:
        verify_instruction = {
            "role": "user",
            "content": (
                "现在请你生成最终回复给用户。严格遵循以下规则：\n"
                "1. **只基于对话中已有的工具返回数据或上下文来回答**，绝对不要编造任何数据\n"
                "2. 如果工具返回的数据和你之前的认知矛盾，以工具数据为准\n"
                "3. 如果信息不足以完整回答，明确告知用户哪些部分不确定\n"
                "4. 语言自然、简洁、友好，避免冗余\n"
                "5. 关键数据（金额、时间、温度等）加粗展示\n"
                "6. **工具返回的数值必须原样引用，不得省略精度**（时间保留毫秒、计算结果保留完整小数）\n"
                "7. **如果工具结果中包含图片链接（以 ![图片] 或 图片链接: 开头），必须原样保留图片链接，不得丢弃**\n"
                "请直接输出最终回复。"
            ),
        }
        final_messages = messages + [verify_instruction]
    else:
        # 没用工具，llm_message 的内容就是回复
        llm_message = state.get("llm_message")
        if llm_message and llm_message.content:
            return {"final_response": _clean_response(llm_message.content)}
        final_messages = messages

    try:
        result = await llm.ainvoke(final_messages)
        final_text = _clean_response(result.content)
    except Exception as e:
        logger.error(f"[response] 生成回复失败: {e}")
        # 降级：使用最后的 assistant 内容
        last_content = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                last_content = msg.content
                break
        final_text = _clean_response(last_content) if last_content else "抱歉，回复生成失败，请重试。"

    # 兜底：如果用了 generate_image 但 LLM 回复中没有图片链接，从工具结果中补回
    if "generate_image" in tools_used and "![生成的图片]" not in final_text:
        for tr in tool_results:
            img_match = re.search(r'!\[生成的图片\]\([^)]+\)', tr)
            if img_match:
                final_text += f"\n\n{img_match.group()}"
                break
            url_match = re.search(r'图片链接:\s*(https?://[^\s]+)', tr)
            if url_match:
                final_text += f"\n\n![生成的图片]({url_match.group(1)})"
                break

    # 清理：本次未生成图片，但 LLM 复制了历史中的旧图片链接 → 移除
    if "generate_image" not in tools_used:
        final_text = re.sub(r'\n*!\[[^\]]*\]\(https?://[^\s)]+\)', '', final_text)
        final_text = re.sub(r'\n*图片链接:\s*https?://[^\s]+', '', final_text)

    return {"final_response": final_text}


def _clean_response(text: str) -> str:
    """清理回复中的工具标记残留（含 DeepSeek DSML 格式）"""
    if not text:
        return ""
    # 清理标准工具调用标记
    text = re.sub(r'\[TOOL_CALL\].*?\[/TOOL_CALL\]', '', text, flags=re.DOTALL)
    # 清理 DeepSeek DSML 格式（全系列 token）
    text = re.sub(r'<\|[^>]*DSML[^>]*\|[^>]*>', '', text)
    text = re.sub(r'<\|[^>]*\|>', '', text)
    # 清理连续空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
