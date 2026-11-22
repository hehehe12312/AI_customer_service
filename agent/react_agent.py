from langchain.agents import create_agent
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

    def execute_stream(self, messages, user_id: str = "1001"):
        """Yield 模型 token 增量。messages: [{"role": "user"|"assistant", "content": "..."}, ...]"""
        # 只把 user/assistant 文本消息交给模型，并截断过长历史
        cleaned = []
        for item in messages:
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                cleaned.append({"role": role, "content": content})
        if self.max_history > 0:
            cleaned = cleaned[-self.max_history:]

        for token, metadata in self.agent.stream(
            {"messages": cleaned},
            stream_mode="messages",
            context={"report": False, "user_id": str(user_id)},
        ):
            # 工具节点里的 RAG 摘要不展示。流式片段的 type 是 AIMessageChunk，不是 ai。
            if metadata.get("langgraph_node") == "tools":
                continue
            if getattr(token, "type", None) not in ("ai", "AIMessageChunk"):
                continue
            text = token.text
            if text:
                yield text


if __name__ == "__main__":
    from model.factory import get_chat_model, get_embed_model
    from agent.tools.agent_tools import init_knowledge, sync_documents

    get_chat_model()
    get_embed_model()
    init_knowledge()
    sync_documents()
    agent = ReactAgent()
    demo_messages = [
        {"role": "user", "content": "上海，身高170体重65，推荐一件。"},
    ]
    for chunk in agent.execute_stream(demo_messages, user_id="default"):
        print(chunk, end="", flush=True)
