from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    lang="en",
    top_k_results=2
)

docs = retriever.invoke("India")

for i, doc in enumerate(docs, start=1):
    print(f"\n--- Result {i} ---")
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Content:\n{doc.page_content}")
    print("-" * 80)