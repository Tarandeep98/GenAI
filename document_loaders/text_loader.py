from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm = llm)

loader = TextLoader('cricket.txt' , encoding='utf-8')
docs = loader.load()

prompt = PromptTemplate(template="write a summary for following content \n {content}" , input_variables=["content"])
parser = StrOutputParser()
chain = prompt | model | parser

result = chain.invoke({"content" : docs[0].page_content})

print(result)

# print(type(docs))

# print(docs[0])
# print(docs[0].page_content)
# print(docs[0].metadata)


