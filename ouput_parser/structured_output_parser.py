from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

#json parsing does not enforce any schema 


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name="fact 1" , description = "fact 1 about the topic"),
    ResponseSchema(name="fact 2" , description = "fact 2 about the topic"),
    ResponseSchema(name="fact 3" , description = "fact 3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(template="give 3 fact about {topic} \n {format_instruction}" , input_variables=["topic"] , partial_variables={'format_instruction' : parser.get_format_instructions()})
    
# prompt = template.invoke({'topic' : "Black hole"})
# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
# print(final_result)

chain = template | model | parser

print(chain.invoke({"topic" : "black hole"}))