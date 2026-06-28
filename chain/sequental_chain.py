from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    task="text-generation",
    temperature=0.5
)

prompt1 = PromptTemplate(template= "generate detailed report on {topic}" , input_variables=["topic"])


prompt2 = PromptTemplate(template= "generate fove pointer summary from following text \n {text}" ,
                          input_variables=["text"])

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic" : "unemployment in india"})

print(result)
