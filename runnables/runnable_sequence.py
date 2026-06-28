from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate   
from langchain_classic.output_parsers import StructuredOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

prompt1 = PromptTemplate(template="write a joke about {topic}" , input_variables=['topic'])
prompt2 = PromptTemplate(template="explain followig joke {text}" , input_variables=['text'])

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm = llm)

chain = RunnableSequence(prompt1 , model , parser , prompt2 , model, parser )

result = chain.invoke({'topic' : 'cricket'})

print(result)