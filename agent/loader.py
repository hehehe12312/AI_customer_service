import threading

from utils.logger_handler import logger

_agent = None
_error: BaseException | None = None
_build: str | None = None
_lock = threading.Lock()
_ready = threading.Event()
_sync_started = False


def _load_named(label: str, fn) -> Exception | None:
    try:
        fn()
    except Exception as exc:
        logger.error(f"{label}失败: {exc}")
        return exc
    return None


def build_agent():
    """嵌入模型和聊天模型同时准备，再打开向量库索引，然后组装客服。"""
    from model.factory import get_chat_model, get_embed_model

    logger.info("开始加载嵌入模型和聊天模型")
    errors: list[Exception] = []
    workers = (
        ("聊天模型", get_chat_model),
        ("嵌入模型", get_embed_model),
    )
    threads = []
    for label, fn in workers:
        def run(label=label, fn=fn) -> None:
            err = _load_named(label, fn)
            if err is not None:
                errors.append(err)

        thread = threading.Thread(target=run, name=f"load-{label}", daemon=True)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    logger.info("嵌入模型和聊天模型已加载，开始打开向量库索引")

    from agent.tools.agent_tools import init_knowledge

    init_knowledge()
    logger.info("向量库索引已打开，开始组装客服")

    from agent.react_agent import ReactAgent

    loaded = ReactAgent()
    logger.info("基础客服已可回复")
    return loaded


def start_document_sync() -> None:
    """资料文件同步单独跑，不挡住已经可以回复的客服。"""
    global _sync_started
    with _lock:
        if _sync_started:
            return
        _sync_started = True

    def _run() -> None:
        try:
            from agent.tools.agent_tools import sync_documents

            sync_documents()
        except Exception as exc:
            logger.error(f"资料文件同步失败: {exc}")

    threading.Thread(target=_run, name="doc-sync", daemon=True).start()


def start_agent_load(build_id: str) -> None:
    """页面元素登记完后再启动。同一版本只加载一次。"""
    global _agent, _error, _build

    def _run(build: str) -> None:
        global _agent, _error
        try:
            loaded = build_agent()
        except Exception as exc:
            with _lock:
                if _build == build:
                    _error = exc
            _ready.set()
            return
        with _lock:
            if _build == build:
                _agent = loaded
        _ready.set()
        if _agent is loaded:
            start_document_sync()

    with _lock:
        if _build == build_id:
            return
        _build = build_id
        _agent = None
        _error = None
        _ready.clear()
        threading.Thread(target=_run, args=(build_id,), name="agent-load", daemon=True).start()


def is_agent_ready(build_id: str) -> bool:
    return _build == build_id and _agent is not None and _error is None


class _ReadyAgent:
    """基础客服就绪后，第一次回复前再打开这个用户的画像和记忆。"""

    def __init__(self, inner):
        self._inner = inner

    def execute_stream(self, messages, user_id: str = "1001"):
        from agent.tools.agent_tools import ensure_user_memory

        ensure_user_memory(user_id)
        return self._inner.execute_stream(messages, user_id=user_id)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def get_agent(build_id: str):
    """加载未完成时一直等到客服组装完。失败则抛出加载时的异常。"""
    if _build != build_id:
        start_agent_load(build_id)
    _ready.wait()
    if _error is not None:
        raise _error
    if _agent is None:
        raise RuntimeError("客服还没有组装完成")
    return _ReadyAgent(_agent)
