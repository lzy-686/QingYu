# -*- coding: utf-8 -*-
"""
load_world.py
-------------
功能：读取 world_data/九洲灵寰_墨清影设定集.pdf，按页提取文本，
      使用 RecursiveCharacterTextSplitter 切分，并打印每个文本片段及其序号、来源页码。

依赖：pypdf、langchain-text-splitters（langchain_core.documents.Document）
"""

import sys
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Windows 控制台默认编码可能是 GBK，强制标准输出为 UTF-8，
# 避免打印设定集中的生僻字/特殊符号时出现 UnicodeEncodeError。
sys.stdout.reconfigure(encoding="utf-8")

# PDF 文件路径：使用 pathlib 以脚本所在目录为基准，避免受运行时工作目录影响
PDF_PATH = Path(__file__).resolve().parent / "world_data" / "九洲灵寰_墨清影设定集.pdf"


def load_pdf_documents(pdf_path: Path) -> list[Document]:
    """读取 PDF，并转换为按页组织的 langchain Document 列表。

    Args:
        pdf_path: PDF 文件的绝对路径。

    Returns:
        list[Document]: 每个元素对应 PDF 的一页，metadata 中记录 source 和 page。

    Raises:
        FileNotFoundError: PDF 文件不存在时抛出，由调用方统一处理。
    """
    # 显式抛出 FileNotFoundError，便于入口处给出清晰的中文提示
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    reader = PdfReader(str(pdf_path))
    documents: list[Document] = []

    # 逐页遍历，页码从 1 开始（与阅读器显示的页码一致）
    for page_index, page in enumerate(reader.pages, start=1):
        # extract_text() 可能返回 None（如纯图片页），统一转换为空字符串
        text = page.extract_text() or ""
        text = text.strip()

        # 跳过没有可提取文本的页面（例如扫描件、纯图页）
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,  # 来源文件名
                    "page": page_index,       # 来源页码（1 起）
                },
            )
        )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """使用 RecursiveCharacterTextSplitter 对文档进行切分。

    切分器会优先按段落、换行、句号等自然边界递归切分，
    并把原 Document 的 metadata（含页码）复制到每个片段上。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,      # 单个片段最大字符数
        chunk_overlap=60,    # 相邻片段重叠字符数，保留上下文连续性
        length_function=len,  # 以字符数计算长度
    )
    return splitter.split_documents(documents)


def main() -> None:
    """主流程：加载 PDF -> 切分 -> 打印片段。"""
    try:
        print(f"[INFO] 正在加载 PDF：{PDF_PATH}")
        documents = load_pdf_documents(PDF_PATH)
    except FileNotFoundError as exc:
        # 捕获文件不存在异常，打印提示后以非 0 状态码退出
        print(f"[ERROR] {exc}")
        print("[ERROR] 请确认 world_data 目录下存在《九洲灵寰_墨清影设定集.pdf》。")
        sys.exit(1)

    if not documents:
        print("[WARN] PDF 中没有提取到任何文本，可能是扫描版或纯图片 PDF。")
        return

    print(f"[INFO] 共读取 {len(documents)} 页有效文本，开始切分 ...")
    chunks = split_documents(documents)
    print(f"[INFO] 切分完成，共得到 {len(chunks)} 个文本片段。\n")

    # 逐个打印片段：序号、来源页码、字符数以及正文
    for index, chunk in enumerate(chunks, start=1):
        page = chunk.metadata.get("page", "未知")
        source = chunk.metadata.get("source", "未知")
        print("=" * 70)
        print(f"片段 #{index}  |  来源: {source}  |  页码: {page}  |  字符数: {len(chunk.page_content)}")
        print("-" * 70)
        print(chunk.page_content)
        print()

    print("=" * 70)
    print(f"[DONE] 处理结束：{len(documents)} 页 -> {len(chunks)} 个片段。")


if __name__ == "__main__":
    main()
