from langchain_text_splitters import RecursiveCharacterTextSplitter , Language
from langchain_community.document_loaders import PyPDFLoader



text = """from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage , SystemMessage ,AIMessage

llm = ChatOllama(model="qwen3:4b" , temperature=0.3)

chat_history = [SystemMessage(content="you are helpful assistant ")]


while True:
    user_input = input("YOU :")
    chat_history.append(HumanMessage(content=user_input  ))
    if user_input.lower() == 'exit':
        break
    result = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f"AI : {result.content}")

print(chat_history)
"""


splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 100 , 
    chunk_overlap = 0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[1])