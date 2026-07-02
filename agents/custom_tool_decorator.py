from langchain_core.tools import tool

#assume llm dont know how to multiply we create tool


# #create function

# def multiply(a,b):
#     """mutiply two numbers"""
#     return a*b

# #add two hint type
# def multiply(a:int , b:int) ->int :
#     """mutiply two numbers"""
#     return a*b


# add tool decorator

@tool
def multiply(a:int , b:int) -> int:
    """mutiply two numbers"""
    return a*b

result = multiply.invoke({"a" : 75 , "b" : 3})
print(result)


print(multiply.name)
print(multiply.description)
print(multiply.args)

# what llm exactly see

print(multiply.args_schema.model_json_schema())

