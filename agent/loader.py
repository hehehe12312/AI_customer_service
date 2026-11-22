import threading
import time

_agent = None
_error: BaseException | None = None
_build: str | None = None
_lock = threading.Lock()


def start_agent_load(build_id: str) -> None:
    """页面已经画完后再加载知识库和模型图。同一版本只加载一次。"""
    global _agent, _error, _build

    def _run(build: str) -> None:
        global _agent, _error
        # 先让出主线程，等页面刷新到浏览器，再加载知识库和模型图。
        time.sleep(0.5)
        try:
            from agent.react_agent import ReactAgent

            loaded = ReactAgent()
        except Exception as exc:
            with _lock:
                if _build == build:
                    _error = exc
            return
        with _lock:
            if _build == build:
                _agent = loaded

    with _lock:
        if _build == build_id:
            return
        _build = build_id
        _agent = None
        _error = None
        threading.Thread(target=_run, args=(build_id,), name="agent-load", daemon=True).start()


def get_agent(build_id: str):
    if _build != build_id:
        start_agent_load(build_id)
    while _agent is None and _error is None:
        time.sleep(0.05)
    if _error is not None:
        raise _error
    return _agent
