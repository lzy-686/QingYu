# -*- coding: utf-8 -*-
"""
qingyu_chat.py
--------------
功能：加载本地 Chroma 向量库，构建 k=3 相似度检索器，
      结合墨清影专属人设 Prompt 与本地 Ollama 大模型，实现命令行循环对话。

运行前提：
    1. 已安装并启动 Ollama（默认服务地址 http://localhost:11434）；
    2. 已拉取对话模型：ollama pull qwen2.5:7b
    若使用其他模型，可通过环境变量 OLLAMA_MODEL 指定。

依赖：langchain-chroma、sentence-transformers、langchain-ollama
"""

import os

# 必须在导入 huggingface / sentence-transformers 之前设置：
#   HF_ENDPOINT —— 使用镜像站加载嵌入模型（与 build_vector.py 保持一致）；
#   ANONYMIZED_TELEMETRY —— 关闭 Chroma 匿名遥测。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama

# Windows 控制台默认编码可能是 GBK，统一为 UTF-8，避免中文输出报错
sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------- 路径与参数配置 ----------------------------
BASE_DIR = Path(__file__).resolve().parent            # 项目根目录
CHROMA_DIR = BASE_DIR / "chroma_db"                  # 向量库目录
COLLECTION_NAME = "world_lore"                       # 与 build_vector.py 中的集合名一致
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"           # 必须在入库/检索时保持一致

RETRIEVAL_K = 3                                      # 检索返回的片段数量
MAX_HISTORY_MESSAGES = 10                            # 保留的最近对话轮次消息数

# Ollama 本地服务配置，可用环境变量覆盖
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# ---------------------------- 墨清影人设 System Prompt ----------------------------
SYSTEM_PROMPT = """你是清雨宗雨司堂常驻执事墨清影，金丹中期修士，面对提问者是清雨宗新晋外门弟子。

说话温柔平缓、耐心，合适时机自然使用口头禅：“不急，慢慢说。”“这事我记下了。”“不太合规矩呢……”

遇到弟子想要浪费灵材、欺压低阶修士、言语刻薄，温柔规劝，说出“不太合规矩呢”。

知识库不存在的内容，统一回复「此事我不甚知晓」，绝对不编造设定。

不能自称AI，不能出现现代词汇，全程保持修仙世界对话。

参考检索到的上下文回答，回答简短自然，不要长篇大论。"""


def build_retriever() -> BaseRetriever:
    """加载本地 Chroma 向量库，并返回 k=3 的相似度检索器。"""
    if not CHROMA_DIR.exists():
        print(f"[ERROR] 未找到向量库目录：{CHROMA_DIR}")
        print("[ERROR] 请先运行 python build_vector.py 构建向量库。")
        sys.exit(1)

    # 嵌入模型必须与建库时完全相同，否则向量空间不一致，检索结果会失真
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    if vector_store._collection.count() == 0:
        print(f"[ERROR] 集合 {COLLECTION_NAME} 为空，请重新运行 python build_vector.py。")
        sys.exit(1)

    # 相似度检索：search_type="similarity"，返回前 k 个最相近片段
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )


def format_context(documents: list) -> str:
    """把检索到的片段拼接为提示词上下文，并标注来源页码。"""
    if not documents:
        # 未检索到内容时明确告知模型，触发「此事我不甚知晓」规则
        return "（未检索到相关记载）"

    blocks = []
    for index, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page", "未知")
        source = doc.metadata.get("source", "未知")
        blocks.append(f"{index}. 来源《{source}》第 {page} 页：\n{doc.page_content}")

    return "\n\n".join(blocks)


def build_chain():
    """构建 提示词 -> 大模型 -> 文本输出 的处理链。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            # 系统人设 + 本轮检索到的上下文
            ("system", SYSTEM_PROMPT + "\n\n【可参考的宗门典籍片段】\n{context}"),
            # 历史对话，用于维持多轮语境
            MessagesPlaceholder(variable_name="history"),
            # 弟子本轮提问
            ("human", "{question}"),
        ]
    )

    llm = ChatOllama(
        model=OLLAMA_MODEL,        # 本地模型名称，可由环境变量 OLLAMA_MODEL 覆盖
        base_url=OLLAMA_BASE_URL,  # Ollama 服务地址
        temperature=0.7,           # 略带变化，让语气更自然
        num_predict=512,           # 限制输出长度，避免长篇大论
        timeout=95
    )

    return prompt | llm | StrOutputParser()


def check_ollama() -> None:
    """检查 Ollama 服务是否运行、目标模型是否已下载。

    提前失败可以避免加载嵌入模型后才发现本地服务不可用。
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] 无法连接 Ollama 服务 {OLLAMA_BASE_URL}：{exc}")
        print("[HINT] 请确认已安装 Ollama 并启动服务（命令行执行 ollama serve）。")
        sys.exit(1)

    # 服务返回的模型名可能带 :latest 标签，匹配时忽略标签部分
    installed = {
        model.get("name", "").split(":")[0]
        for model in data.get("models", [])
    }
    if OLLAMA_MODEL.split(":")[0] not in installed:
        print(f"[ERROR] 本地未找到模型 {OLLAMA_MODEL}。")
        print(f"[HINT] 请先执行：ollama pull {OLLAMA_MODEL}")
        sys.exit(1)


def main() -> None:
    """主流程：读取输入 -> 检索 -> 生成回复 -> 循环。"""
    # 启动前先确认本地 Ollama 服务与模型可用
    check_ollama()

    print(f"[INFO] Ollama 服务就绪，对话模型：{OLLAMA_MODEL}")
    print("[INFO] 正在加载向量库与嵌入模型 ...")
    retriever = build_retriever()
    chain = build_chain()

    print("=" * 60)
    print("清雨宗 · 雨司堂")
    print("墨清影：不急，慢慢说。有什么想问的，坐下讲便是。")
    print("（按空格键后回车，即可离开对话）")
    print("=" * 60)

    history: list[BaseMessage] = []   # 保存历史对话消息

    while True:
        try:
            raw_input = input("\n弟子：")
        except (EOFError, KeyboardInterrupt):
            # 用户按 Ctrl+C 或输入流结束（Ctrl+Z）时优雅退出
            print("\n墨清影：路上仔细些，得空再来坐坐。")
            break

        # 只输入空格（一个或多个）后回车：离开对话
        if raw_input and not raw_input.strip():
            print("墨清影：路上仔细些，得空再来坐坐。")
            break

        # 直接回车、未输入任何字符：视为空输入，提示后继续对话
        if raw_input == "":
            print("[提示] 输入为空，请说出想问的事；若要离开，按空格键后回车。")
            continue

        # 去除首尾空白后作为本轮提问
        user_input = raw_input.strip()

        try:
            # 1) 向量检索：取 k=3 相似片段
            documents = retriever.invoke(user_input)

            # 2) 生成回答：注入上下文、历史对话与本轮提问
            answer = chain.invoke(
                {
                    "context": format_context(documents),
                    "history": history[-MAX_HISTORY_MESSAGES:],
                    "question": user_input,
                }
            )
        except Exception as exc:
            # 网络、鉴权、限流等运行时错误不应中断整个对话循环
            print(f"[ERROR] 本轮回复失败：{exc}")
            continue

        print(f"墨清影：{answer}")

        # 3) 记录本轮对话，供后续轮次参考
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=answer))


if __name__ == "__main__":
    main()
