from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def multiply(a:int , b: int) ->int:
    """Given 2 numbers a and b this tool returns their product"""
    return a*b

load_dotenv()

product = multiply.invoke({"a" : 42 , "b" : 5})
# print(product)

# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

# tool binding 

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)



llm_with_tools = llm.bind_tools([multiply])

# print(llm_with_tools.invoke("can u multiply 45 with 3")).tool_calls

# # llm does not run the tool or print output it suggest tool and input argument actual execution handles by
# # LANGCHAIN OR US   



user_query = input("enter your query ... ")
query = HumanMessage(content=user_query)
message = [query]

result = llm_with_tools.invoke(message)

message.append(result)


# print(result.tool_calls[0]["args"])
input1 = result.tool_calls[0]["args"]  # this thing passed to tool


tool_output = multiply.invoke(input1)
tool_msg = ToolMessage(
    content=str(tool_output),
    tool_call_id=result.tool_calls[0]["id"]
)

message.append(tool_msg)

print(tool_msg)  # tool message (correct output)

# print(message)

final_result = llm_with_tools.invoke(message)
print(final_result.content)