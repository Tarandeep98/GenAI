from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage , ToolMessage
from langchain_core.tools import tool , InjectedToolArg
from typing import Annotated
from dotenv import load_dotenv
import requests

load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)



@tool
def get_conversionF(base_currency : str , target_currency : str) -> float:
    """this function fetches the currency conversion factor between base and target currency"""

    url = f"https://v6.exchangerate-api.com/v6/10f89e1908ff6e29ff4cb390/pair/{base_currency}/{target_currency}"

    response = requests.get(url)
    return response.json()


@tool
def convert(base_currency_value : float , conversion_factor : Annotated[float , InjectedToolArg]) -> float:
    """given currency conversion rate this function calculate targeted currency from base currency """

    return base_currency_value * conversion_factor


f = get_conversionF.invoke({"base_currency" : "USD" , "target_currency" : "INR"})
print(f)

print(convert.invoke({"base_currency_value" : 10 , "conversion_factor" : f["conversion_rate"]}))


llm_with_tool = llm.bind_tools([get_conversionF , convert])

messages = [HumanMessage("what is conversion factor between usd and inr and based on that calculate 10 usd to inr")
            ]

ai_msg = llm_with_tool.invoke(messages)

messages.append(ai_msg)

import json

conversion_rate = None

for tool_call in ai_msg.tool_calls:
  # execute the 1st tool and get the value of conversion rate
  if tool_call['name'] == 'get_conversionF':
    tool_message1 = get_conversionF.invoke(tool_call['args'])

    # fetch this conversion rate
    conversion_rate = tool_message1["conversion_rate"]

    # append this tool message to messages list
    messages.append(tool_message1)

  # execute the 2nd tool using the conversion rate from tool 1
  if tool_call['name'] == 'convert':
    # fetch the current arg
    tool_call['args']['conversion_factor'] = conversion_rate

    tool_message2 = convert.invoke(tool_call['args'])
    messages.append(tool_message2)


final_response = llm_with_tool.invoke(messages)

print(final_response.content)