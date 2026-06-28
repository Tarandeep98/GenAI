from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnablePassthrough , RunnableLambda , RunnableBranch
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.output_parsers import StructuredOutputParser

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt = PromptTemplate(template="write a detailed report on topic {topic}" , input_variables=["topic"])
prompt2 = PromptTemplate(template="summarize the report {topic}" , input_variables=["topic"])

report_gen = RunnableSequence(prompt , model , parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 500 , RunnableSequence(prompt2 , model , parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen , branch_chain)

result = final_chain.invoke({"topic" : "thapar university"})
print(result)