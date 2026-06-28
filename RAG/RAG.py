from langchain_core.runnables import RunnableParallel , RunnableSequence , RunnablePassthrough ,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings , HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

video_id = "LPZh9BOjkQs"

try:
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id, languages=["en"])

    transcript = " ".join(snippet.text for snippet in fetched_transcript)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")


splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 100)
chunks = splitter.create_documents([transcript])

# print(len(chunks))

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm = llm)

vector_store = FAISS.from_documents(chunks , embedding_model)

print(vector_store.index_to_docstore_id)

retriever = vector_store.as_retriever(search_type = "similarity" , search_kwargs = {"k" : 3})


prompt = PromptTemplate(
    template="""You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question:
{question}

Answer:""",
    input_variables=["context", "question"]
)

question = input("enter your question.... \n ")

# retrieved_docs = retriever.invoke(question)

# context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# print(context_text)

# final_prompt = prompt.invoke({"context": context_text, "question": question})


# ans = model.invoke(final_prompt)
# print(ans.content)


def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
   "context" : retriever | RunnableLambda(format_docs),
   "question" : RunnablePassthrough()
})

# parallel_chain.invoke(question)


parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser

print(main_chain.invoke(question))












