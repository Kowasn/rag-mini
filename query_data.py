import argparse
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import OllamaEmbeddings

load_dotenv()

CHROMA_PATH = "chroma"
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.30"))  # 低于此分数就拒答

PROMPT_TEMPLATE = """你是一个严谨的文档问答助手。
规则：
1. 只能依据下面的【资料】回答；
2. 资料里没有的信息，直接回答"资料中未提及"，不要编造；
3. 回答用中文，控制在 200 字以内。

【资料】
{context}

【问题】
{question}"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="你的问题")
    parser.add_argument("--k", type=int, default=3, help="检索几条")
    parser.add_argument("--show", action="store_true", help="打印所有片段和分数")
    args = parser.parse_args()

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    results = db.similarity_search_with_relevance_scores(args.query_text, k=args.k)
    if not results:
        print("库里没有内容，请先运行 create_database.py")
        return

    print("\n--- 检索结果 ---")
    for doc, score in results:
        print(f"[{score:.3f}] {doc.metadata.get('source')}  start_index={doc.metadata.get('start_index')}")
        if args.show:
            print("   ", doc.page_content[:150].replace("\n", " "))

    if results[0][1] < SCORE_THRESHOLD:
        print(f"\n最高分 {results[0][1]:.3f} < 阈值 {SCORE_THRESHOLD} -> 拒答（避免编造）")
        return

    context_text = "\n\n---\n\n".join(doc.page_content for doc, _ in results)
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=args.query_text)

    model = ChatDeepSeek(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=os.getenv("DEEPSEEK_BASE_URL"),
    )
    response = model.invoke(prompt)

    print("\n=== 回答 ===")
    print(response.content)
    print("\n=== 引用来源 ===")
    for doc, score in results:
        print(f"- {doc.metadata.get('source')}  相关度 {score:.3f}")


if __name__ == "__main__":
    main()