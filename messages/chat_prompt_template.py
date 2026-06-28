from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage , AIMessage , HumanMessage

llm = ChatOllama(model="qwen3:4b")

chat_template = ChatPromptTemplate([
    ("system","you are helful {domain} expert"),
    ("human" , "explain in simple terms , what is {topic}")
    ])


prompt = chat_template.invoke({'domain' :"cricket" ,"topic" : "rules of wicket" })

print(prompt)
