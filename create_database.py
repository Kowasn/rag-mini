import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = Path("data/books")
CHROMA_PATH = "chroma"
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")


def load_documents() -> list[Document]:
    """把目录下所有 .md 读成 LangChain 的 Document 对象。"""
    docs = []
    for path in sorted(DATA_PATH.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
        print(f"  载入 {path.name}（{len(text)} 字符）")
    return docs


def split_text(documents: list[Document]) -> list[Document]:
    """切分：每块 300 字符、重叠 100，并记录在原文件里的起始位置。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"\n{len(documents)} 个文档 -> {len(chunks)} 个 chunk")
    if chunks:
        sample = chunks[0]                       # 不再写死下标，取第一块
        print("\n--- 样例 chunk ---")
        print(sample.page_content[:120])
        print("metadata:", sample.metadata)
    return chunks                               


def save_to_chroma(chunks: list[Document], batch_size: int = 32) -> None:
    """向量化并写入 Chroma（分批，避免单次请求过大被服务端掐断）。"""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"\n已删除旧库：{CHROMA_PATH}/")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},   # 余弦相似度
    )

    total = len(chunks)
    print(f"\n开始向量化：共 {total} 个 chunk，每批 {batch_size} 条")

    for start in range(0, total, batch_size):
        batch = chunks[start:start + batch_size]
        db.add_documents(batch)
        done = min(start + batch_size, total)
        print(f"  已写入 {done}/{total}")

    print(f"\n完成：{total} 个 chunk -> {CHROMA_PATH}/")


def main() -> None:
    print(f"开始建库（embedding 模型：{EMBED_MODEL}）\n")
    documents = load_documents()
    if not documents:
        raise SystemExit(f"没有在 {DATA_PATH} 找到 .md 文件")
    chunks = split_text(documents)
    save_to_chroma(chunks)
    print("\n完成。下一步：python query_data.py \"Who is Alice?\"")


if __name__ == "__main__":
    main()