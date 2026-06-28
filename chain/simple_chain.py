from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#json parsing does not enforce any schema 


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    task="text-generation",
    temperature=0.5
)

parser = StrOutputParser()

model = ChatHuggingFace(llm = llm)

template = PromptTemplate(template = "generate five interesting facts about {topic} \n" , input_variables=["topic"])

chain = template | model | parser

result = chain.invoke({"topic" : "cricket"})
print(result)

chain.get_graph().print_ascii()