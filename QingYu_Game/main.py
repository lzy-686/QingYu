# -*- coding: utf-8 -*-
"""
main.py
-------
功能：复用 qingyu_chat.py 的 RAG 逻辑（Chroma 检索 + 墨清影人设 Prompt + Ollama），
      封装为 FastAPI HTTP 接口。

接口：
    POST /chat
    请求体：{"question": "用户提问内容"}
    返回体：{"answer": "墨清影的回答"}

启动：
    uvicorn main:app --host 127.0.0.1 --port 8000

前提：已运行 build_vector.py 生成 ./chroma_db，且本地 Ollama 服务与模型可用。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 直接复用命令行版本的检索器、生成链与上下文格式化逻辑，
# 保证 HTTP 接口与命令行对话的人设、检索参数完全一致。
import qingyu_chat as qc

# ---------------------------- 运行时全局状态 ----------------------------
# 检索器与生成链在应用启动时初始化一次，避免每个请求重复加载嵌入模型。
retriever = None
chain = None
init_error: str | None = None   # 初始化失败原因，供接口返回友好提示


class ChatRequest(BaseModel):
    """POST /chat 请求体。"""

    question: str = Field(default="", description="弟子的提问内容")


class ChatResponse(BaseModel):
    """POST /chat 成功时的返回体。"""

    answer: str = Field(description="墨清影的回答")


def init_rag() -> None:
    """初始化检索器与生成链，失败时记录原因而不终止服务进程。

    qingyu_chat.build_retriever() 在向量库缺失时会调用 sys.exit()，
    因此这里需要额外捕获 SystemExit，保证 FastAPI 进程不被直接退出。
    """
    global retriever, chain, init_error

    try:
        print("[INFO] 正在加载向量库、嵌入模型与对话链 ...")
        retriever = qc.build_retriever()
        chain = qc.build_chain()
        print(f"[INFO] 初始化完成，对话模型：{qc.OLLAMA_MODEL}")
    except SystemExit:
        # 向量库目录不存在或集合为空
        init_error = "向量库未就绪"
        print("[ERROR] 向量库未就绪，请先运行 python build_vector.py")
    except Exception as exc:
        init_error = str(exc)
        print(f"[ERROR] RAG 初始化失败：{exc}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """FastAPI 生命周期：启动时初始化 RAG，关闭时无需额外清理。"""
    init_rag()
    yield


app = FastAPI(
    title="清雨宗 · 雨司堂",
    description="墨清影角色问答接口（RAG）",
    lifespan=lifespan,
)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """接收弟子提问，检索典籍后由墨清影作答。

    Args:
        request: 包含 question 字段的 JSON 请求体。

    Returns:
        ChatResponse: {"answer": "..."}

    Raises:
        HTTPException: 400 空提问；503 检索/生成失败或服务未就绪。
    """
    # 去除首尾空白，纯空格同样视为空提问
    question = (request.question or "").strip()
    if not question:
        # 400：请求内容不合法，提示语保持人设口吻
        raise HTTPException(
            status_code=400,
            detail="弟子还未开口呢，请先写下想问的事。",
        )

    # 服务未初始化成功时，直接返回友好提示，避免抛出 500
    if retriever is None or chain is None:
        raise HTTPException(
            status_code=503,
            detail=f"雨司堂的典籍尚未整理妥当（{init_error or '服务未就绪'}），请稍后再来。",
        )

    # 终端日志：记录用户提问，便于排查问题
    print(f"[CHAT] 弟子提问：{question}")

    # 1) 向量检索：与命令行版本一致，k=3 相似度检索
    try:
        documents = retriever.invoke(question)
    except Exception as exc:
        print(f"[ERROR] 检索失败：{exc}")
        raise HTTPException(
            status_code=503,
            detail="典籍一时翻检不出，此事容我稍后再查。",
        ) from exc

    # 2) 生成回答：注入检索上下文；HTTP 接口无会话状态，历史为空
    try:
        answer = chain.invoke(
            {
                "context": qc.format_context(documents),
                "history": [],
                "question": question,
            }
        )
    except Exception as exc:
        # 常见原因：Ollama 未启动、模型未拉取、请求超时
        print(f"[ERROR] 生成回答失败：{exc}")
        raise HTTPException(
            status_code=503,
            detail="墨清影眼下有些不便，请稍后再试。",
        ) from exc

    print(f"[CHAT] 墨清影回答：{answer}")
    return ChatResponse(answer=answer)
