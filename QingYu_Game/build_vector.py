# -*- coding: utf-8 -*-
"""
build_vector.py
---------------
功能：读取 world_data 目录下的全部 PDF，加载并切分文本，
      使用 sentence-transformers 生成向量嵌入，持久化写入 Chroma 向量数据库（./chroma_db）。

特性：
    1. 若 ./chroma_db 已存在，先整体删除后重建（覆盖式入库）；
    2. 检索配置为 k=3 的相似度检索；
    3. 捕获 PDF 读取异常，单个文件失败不影响其余文件；
    4. 结束时打印成功入库的文档块数量。

依赖：pypdf、langchain-text-splitters、langchain-chroma、sentence-transformers
"""

import os

# ---------------------------------------------------------------------------
# 环境变量必须在导入 huggingface / sentence-transformers 之前设置：
#   HF_ENDPOINT      —— 国内无法直连 huggingface.co，默认改用镜像站下载模型；
#   ANONYMIZED_TELEMETRY —— 关闭 Chroma 匿名遥测，避免多余的联网请求。
# 使用 setdefault，用户若已在系统环境变量中配置，则以用户的配置优先。
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import shutil
import sys
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Windows 控制台默认编码可能为 GBK，统一输出为 UTF-8，避免生僻字打印报错
sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------- 路径与参数配置 ----------------------------
BASE_DIR = Path(__file__).resolve().parent          # 项目根目录（脚本所在目录）
WORLD_DIR = BASE_DIR / "world_data"                # PDF 素材目录
CHROMA_DIR = BASE_DIR / "chroma_db"                # 向量库持久化目录
COLLECTION_NAME = "world_lore"                     # Chroma 集合名称

CHUNK_SIZE = 450                                   # 切分片段最大字符数
CHUNK_OVERLAP = 60                                 # 相邻片段重叠字符数
RETRIEVAL_K = 3                                    # 检索返回的相似片段数量
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"         # 中文语义嵌入模型（维度 512）


def load_pdf_documents(pdf_path: Path) -> list[Document]:
    """加载单个 PDF，并转换为按页组织的 Document 列表。

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        list[Document]: 每页一个 Document，metadata 记录来源文件名和页码。

    Raises:
        FileNotFoundError: 文件不存在。
        PdfReadError: PDF 结构损坏或无法解析。
        OSError: 文件读取失败（权限、占用等）。
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"文件不存在：{pdf_path}")

    reader = PdfReader(str(pdf_path))
    documents: list[Document] = []

    # 页码从 1 开始，与阅读器显示保持一致
    for page_index, page in enumerate(reader.pages, start=1):
        # 纯图片页可能提取不到文字，extract_text() 返回 None
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,   # 来源 PDF 文件名
                    "page": page_index,        # 来源页码（1 起）
                },
            )
        )

    return documents


def collect_documents() -> list[Document]:
    """遍历 world_data 下的所有 PDF，汇总为 Document 列表。

    单个文件读取失败时打印警告并跳过，不影响其他文件入库。
    """
    if not WORLD_DIR.exists():
        print(f"[WARN] 素材目录不存在：{WORLD_DIR}")
        return []

    pdf_files = sorted(WORLD_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] 目录中没有 PDF 文件：{WORLD_DIR}")
        return []

    all_documents: list[Document] = []
    for pdf_path in pdf_files:
        print(f"[INFO] 读取 PDF：{pdf_path.name}")
        try:
            documents = load_pdf_documents(pdf_path)
        except FileNotFoundError as exc:
            print(f"[WARN] 文件不存在，已跳过：{exc}")
            continue
        except PdfReadError as exc:
            print(f"[WARN] PDF 解析失败，已跳过 {pdf_path.name}：{exc}")
            continue
        except OSError as exc:
            # 例如文件被其他程序占用、无读取权限
            print(f"[WARN] 文件读取失败，已跳过 {pdf_path.name}：{exc}")
            continue

        print(f"[INFO]   有效页数：{len(documents)}")
        all_documents.extend(documents)

    return all_documents


def split_documents(documents: list[Document]) -> list[Document]:
    """使用 RecursiveCharacterTextSplitter 切分文档。

    切分器优先在段落、换行、句号等自然边界处断开，
    并把原 Document 的 metadata（含页码）复制到每个片段上。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,   # 按字符数计算长度
    )
    return splitter.split_documents(documents)


def rebuild_vector_store(chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> Chroma:
    """覆盖式重建 Chroma 向量库。

    先删除已存在的 ./chroma_db 目录（含旧集合），再把全部片段重新嵌入写入。
    """
    if CHROMA_DIR.exists():
        print(f"[INFO] 检测到已有向量库 {CHROMA_DIR}，正在删除并重建 ...")
        shutil.rmtree(CHROMA_DIR)

    print("[INFO] 正在生成向量并写入 Chroma（首次运行需下载嵌入模型，请稍候）...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),   # 持久化目录
    )
    return vector_store


def demo_search(vector_store: Chroma, query: str) -> None:
    """按 k=3 做相似度检索，并打印命中的片段用于验证入库结果。"""
    print(f"\n[INFO] 相似度检索演示（k={RETRIEVAL_K}）：{query}")
    results = vector_store.similarity_search(query, k=RETRIEVAL_K)

    for index, doc in enumerate(results, start=1):
        page = doc.metadata.get("page", "未知")
        source = doc.metadata.get("source", "未知")
        print("-" * 70)
        print(f"命中 #{index} | 来源: {source} | 页码: {page}")
        # 片段较长时只展示前 120 个字符，保持终端输出简洁
        preview = doc.page_content.replace("\n", " ")
        print(preview[:120] + ("..." if len(preview) > 120 else ""))


def main() -> None:
    """主流程：加载 PDF -> 切分 -> 嵌入入库 -> 检索验证。"""
    print(f"[STEP 1] 扫描素材目录：{WORLD_DIR}")
    documents = collect_documents()
    if not documents:
        print("[ERROR] 没有可处理的有效文本，向量库未构建。")
        sys.exit(1)

    print(f"\n[STEP 2] 开始切分（chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}）...")
    chunks = split_documents(documents)
    if not chunks:
        print("[ERROR] 切分结果为空，向量库未构建。")
        sys.exit(1)
    print(f"[INFO] 得到 {len(chunks)} 个文本片段。")

    print(f"\n[STEP 3] 加载嵌入模型：{EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        # 归一化后向量长度为 1，配合 L2 距离等价于余弦相似度
        encode_kwargs={"normalize_embeddings": True},
    )

    try:
        vector_store = rebuild_vector_store(chunks, embeddings)
    except Exception as exc:
        # 嵌入模型下载失败、磁盘写入失败等属于致命错误，直接终止
        print(f"[ERROR] 向量库构建失败：{exc}")
        sys.exit(1)

    # 以集合中实际存储的向量条数作为入库结果，比 len(chunks) 更可信
    stored_count = vector_store._collection.count()
    print(f"\n[DONE] 向量库已保存到：{CHROMA_DIR}")
    print(f"[DONE] 成功入库的文档块数量：{stored_count}")
    print(f"[DONE] 检索配置：k={RETRIEVAL_K}，相似度检索（similarity_search）")

    demo_search(vector_store, "清雨宗的门规有哪些？")


if __name__ == "__main__":
    main()
