from langchain_core.tools import StructuredTool
from pydantic import BaseModel , Field

class MultiplyInput(BaseModel):
    a : int = Field(required = True , description="the first no to multiply ")
    b : int = Field(required = True , description="the second no to multiply ")

def multiply(a:int , b:int) -> int:
    """mutiply two numbers"""
    return a*b

multiply_tool = StructuredTool.from_function(
    func = multiply ,
    name = "multiply " ,description="multiply two numbers " , args_schema=MultiplyInput
)

result = multiply_tool.invoke({"a" : 75 , "b" : 9})
print(result)