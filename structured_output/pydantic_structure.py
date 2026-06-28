from pydantic import BaseModel , EmailStr , Field 
# from typing import Optional

# class Student(BaseModel):
#     # name : str
# # new_student = {"name" : "Taran"}

#     name : str = "nitish"
#     age : Optional[int] = None
#     email : EmailStr
#     cgpa : float = Field(gt=0 , le=10)

# new_student = {'age' : 32 , "email" : "abc@gmail.com" , 
#                'cgpa' : 11 }
# # new_student = {} 
# student = Student(**new_student)

# print(student)
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

from typing import TypedDict , Annotated , Optional , Literal
   
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:4b" , temperature= 0.5)
class Review(BaseModel):

    
    key_themes: list[str] = Field(
        description="Write down all key themes discussed in the review."
    )

    summary: str = Field(
        description="A brief summary of the review."
    )

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Return the sentiment of the review."
    )

    pros: Optional[list[str]] = Field(
        default=None,
        description="List all positive aspects mentioned in the review."
    )

    cons: Optional[list[str]] = Field(
        default=None,
        description="List all negative aspects mentioned in the review."
    )

structured_model = llm.with_structured_output(Review)

result = structured_model.invoke(review) 

# print(result)
print(result.sentiment)
print(result.model_dump().keys())
