#used for converting python function to runnable


from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate   
from langchain_classic.output_parsers import StructuredOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel ,RunnableLambda , RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

prompt1 = PromptTemplate(template="generate a joke about {topic}" , input_variables=['topic'])


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm = llm)
passthrough = RunnablePassthrough()

joke_gen = RunnableSequence(prompt1 , model , parser)

def word_count(text):
    return len(text.split())


parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough() ,
    "word_count": RunnableLambda(word_count),
})

final_chain = RunnableSequence(joke_gen , parallel_chain)

result = final_chain.invoke({'topic' : 'cricket'})

print(result)