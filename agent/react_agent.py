from langchain.agents import create_agent
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from agent.tools.agent_tools import (
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
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
                get_user_location,
                get_user_id,
                get_current_month,
                fetch_external_data,
                fill_context_for_report,
                save_user_memory,
                recall_user_memory,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
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

    def execute_stream(self, messages, user_id: str = "1001"):
        """messages: [{"role": "user"|"assistant", "content": "..."}, ...]"""
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

        for chunk in self.agent.stream(
            input_dict,
            stream_mode="values",
            context={"report": False, "user_id": str(user_id)},
        ):
            latest_message = chunk["messages"][-1]
            text = self._content_to_text(latest_message.content)
            if text:
                yield text + "\n"


if __name__ == "__main__":
    agent = ReactAgent()
    demo_messages = [
        {"role": "user", "content": "我是用户1001，我在杭州，用的是扫拖一体X1，请记住。"},
    ]
    for chunk in agent.execute_stream(demo_messages, user_id="1001"):
        print(chunk, end="", flush=True)
