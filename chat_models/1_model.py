from langchain_ollama import ChatOllama

llm = ChatOllama(model= " qwen3:4b" , temperature= 0.5)

response = llm.invoke("what is capital of india")
print(response.content)