from typing import TypedDict , Annotated , Optional , Literal
   
# class Person(TypedDict):
#     name:str
#     age:int

# new_person : Person = {'name' : 'Taran' , 'age' : 20}

# print(new_person)


## llm with typed_dict

from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:4b" , temperature= 0.5)

# review = """
#         The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the
# looks outdated compared to other brands. Hoping for a software update to fix this.

# I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3
# processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily
# lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

# The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me
# away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x
# actually works well for distant objects, but anything beyond 30x loses quality.

# However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with
# bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard
# pill to swallow.

# Pros:
# Insanely powerful processor (great for gaming and productivity)
# Stunning 200MP camera with incredible zoom capabilities
# Long battery life with fast charging
# S-Pen support is unique and useful

# Cons:
# Bulky and heavy-not great for one-handed use
# Bloatware still exists in One UI
# Expensive compared to competitors

# """

# class Review(TypedDict):
#     # summary : str
#     summary : Annotated[str,"a brief summary of review " ]

#     # sentiment : str
#     sentiment : Annotated[str,"return sentiment of review either negative ,positive or neutral "]

# structured_model = llm.with_structured_output(Review)

# result = structured_model.invoke(review)

# # print(result)
# print(result['sentiment'])

review = """
        The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI
looks outdated compared to other brands. Hoping for a software update to fix this.

I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3
processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily
lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me
away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x
actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with
bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard
pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors I

"""


# by annoatation we can tell in brief what we mean
# we can add as many key pair we want
#with optional we can make it optional like may be in some prompt optional args present
#with literal we just limit it like ans only from those 


#there is no guarantee in typed_dict


class Review(TypedDict):

    key_themes : Annotated[list[str],"write down all key theme discussed in review in list "]
    # summary : str
    summary : Annotated[str,"a brief summary of review " ]

    # sentiment : str
    sentiment : Annotated[Literal["positive","negative"],"return sentiment of review either negative ,positive or neutral "]

    pros:Annotated[Optional[list[str]] , "write down all pros of list "]
    cons:Annotated[Optional[list[str]] , "write down all cons of list "]

    name : Annotated[Optional[str],"write name of reviewer"]

structured_model = llm.with_structured_output(Review)

result = structured_model.invoke(review) 

# print(result)
print(result["sentiment"])
print(result.keys())
