from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate   
from langchain_classic.output_parsers import StructuredOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel ,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

prompt1 = PromptTemplate(template="generate a joke about {topic}" , input_variables=['topic'])
prompt2 = PromptTemplate(template="generate a linkdin post about {topic}" , input_variables=['topic'])

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm = llm)
passthrough = RunnablePassthrough()

joke_gen = RunnableSequence(prompt1 , model , parser)

parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough() ,
    "explanation" : RunnableSequence(prompt2 , model , parser)
})

final_chain = RunnableSequence(joke_gen , parallel_chain)

result = final_chain.invoke({'topic' : 'cricket'})

print(result)