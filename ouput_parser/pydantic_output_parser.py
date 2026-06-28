from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel , Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm = llm)


class Person(BaseModel):
    name : str = Field(description="name of person ")
    age : int = Field(gt = 18 , description="Age of person ")
    city : str = Field(description="city of person ")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(template="generate name , age , city of a fiction {place} person \n {format_instruction} " 
                          , input_variables=["place"],
                          partial_variables={"format_instruction" : parser.get_format_instructions()})

# prompt = template.invoke({"place" : "indian"})


# print(prompt)

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
# print(final_result)

chain = template | model | parser
result = chain.invoke({"place" : "indian"})
print(result)