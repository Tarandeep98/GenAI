from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate , load_prompt

llm = ChatOllama(model="qwen3:4b" , temperature=0.5)

paper_input = input("enter paper title :")
length_input = input("how bigger the response u want : ")
style_input = input("in which style u want ur response : ")


template = load_prompt('template.json')


prompt = template.invoke({"paper_input" : paper_input,"style_input" : style_input , "length_input" : length_input})

print(llm.invoke(prompt).content)
