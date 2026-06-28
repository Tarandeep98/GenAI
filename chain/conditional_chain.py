from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    task="text-generation",
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback"
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template="""
Classify the sentiment of the following feedback as either positive or negative.

Feedback:
{text}

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": parser2.get_format_instructions()
    }
)

classifier_chain = prompt | model | parser2

positive_prompt = PromptTemplate(
    template="""
Write an appropriate response for this positive feedback:

{feedback}
""",
    input_variables=["feedback"]
)

negative_prompt = PromptTemplate(
    template="""
Write an appropriate response for this negative feedback:

{feedback}
""",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        positive_prompt | model | parser1
    ),
    (
        lambda x: x.sentiment == "negative",
        negative_prompt | model | parser1
    ),
    RunnableLambda(lambda x: "Could not determine sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({"text": "this is terrible smartphone, waste of money"}))