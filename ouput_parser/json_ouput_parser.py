from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

#json parsing does not enforce any schema 


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm = llm)
parser = JsonOutputParser()


template = PromptTemplate(template = "give me name , age and city of fictional person \n {format_instruction}" ,
                           input_variables= [] , partial_variables={'format_instruction' : parser.get_format_instructions()})

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
# print(final_result)
# print(type(final_result))


chain = template | model | parser 
result = chain.invoke({})

print(result)

