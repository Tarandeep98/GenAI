from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)


## by .content


# # First prompt
# template1 = PromptTemplate(
#     template="""
# You are an expert science writer.

# Write a detailed report on {topic}.

# Instructions:
# - Use clear headings and subheadings.
# - Explain concepts in simple language.
# - Include important facts and examples.
# - Keep the report informative and well-structured.
# - Return only the report.
# """,
#     input_variables=["topic"]
# )

# # Second prompt
# template2 = PromptTemplate(
#     template="""
# You are an expert summarizer.

# Summarize the following text in exactly 5 lines.

# Instructions:
# - Capture only the most important points.
# - Do not add any new information.
# - Do not explain your reasoning.
# - Do not think step by step.
# - Return only the summary.

# Text:
# {text}
# """,
#     input_variables=["text"]
# )

# prompt1 = template1.invoke({"topic": "black hole"})
# result = model.invoke(prompt1)

# prompt2 = template2.invoke({"text": result.content})

# print(model.invoke(prompt2).content)





#by str output parser

from langchain_core.output_parsers import StrOutputParser
# First prompt
template1 = PromptTemplate(
    template="""
You are an expert science writer.

Write a detailed report on {topic}.

Instructions:
- Use clear headings and subheadings.
- Explain concepts in simple language.
- Include important facts and examples.
- Keep the report informative and well-structured.
- Return only the report.
""",
    input_variables=["topic"]
)

# Second prompt
template2 = PromptTemplate(
    template="""
You are an expert summarizer.

Summarize the following text in exactly 5 lines.

Instructions:
- Capture only the most important points.
- Do not add any new information.
- Do not explain your reasoning.
- Do not think step by step.
- Return only the summary.

Text:
{text}
""",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

print(chain.invoke({"topic" : "Black hole"}))
