from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
# from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from agent.tools.middleware import monitor_tool, log_before_model
from agent.tools.agent_tools import (
    rag_summarize,
    get_weather,
    # get_user_location,
    # get_user_id,
    # get_current_month,
    # fetch_external_data,
    # fill_context_for_report,
    save_user_memory,
    recall_user_memory,
)
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt
from utils.config_handler import memory_conf


class ReactAgent(object):
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=[
                rag_summarize,
                get_weather,
                # get_user_location,
                # get_user_id,
                # get_current_month,
                # fetch_external_data,
                # fill_context_for_report,
                save_user_memory,
                recall_user_memory,
            ],
            # middleware=[monitor_tool, log_before_model, report_prompt_switch],
            middleware=[monitor_tool, log_before_model],
        )
        self.max_history = int(memory_conf.get("max_history_messages", 20))

    @staticmethod
    def _content_to_text(content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    text = getattr(item, "text", None)
                    parts.append(text if isinstance(text, str) else str(item))
            return "".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _chunk_text_delta(content) -> str:
        """从流式 AIMessageChunk 取出增量文本（不做 strip，避免吃掉空格/换行）。"""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
            return "".join(parts)
        return ""

    def execute_stream(self, messages, user_id: str = "1001"):
        """按模型 token 增量 yield 文本，供前端打字机展示（无需 sleep）。"""
        # 只把 user/assistant 文本消息交给模型，并截断过长历史
        cleaned = []
        for item in messages:
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                cleaned.append({"role": role, "content": content})
        if self.max_history > 0:
            cleaned = cleaned[-self.max_history:]

        input_dict = {"messages": cleaned}

        # messages 模式：边生成边推送；工具调用阶段会自然停顿，无需人为 sleep
        for msg, _metadata in self.agent.stream(
            input_dict,
            stream_mode="messages",
            context={"report": False, "user_id": str(user_id)},
        ):
            if not isinstance(msg, AIMessageChunk):
                continue
            # 纯 tool_call 分片没有对用户可见的正文，跳过
            delta = self._chunk_text_delta(msg.content)
            if delta:
                yield delta


if __name__ == "__main__":
    agent = ReactAgent()
    demo_messages = [
        {"role": "user", "content": "上海，身高170体重65，推荐一件。"},
    ]
    for chunk in agent.execute_stream(demo_messages, user_id="default"):
        print(chunk, end="", flush=True)
