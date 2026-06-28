from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage , AIMessage , HumanMessage

llm = ChatOllama(model="qwen3:4b")

messages = [SystemMessage(content="YOU ARE HELPFUL ASSISTANT"),
            HumanMessage(content="tell me about langchain")]

result = llm.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)