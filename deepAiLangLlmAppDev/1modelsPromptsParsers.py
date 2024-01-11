import openai
import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

client = openai.OpenAI()

current_date = datetime.datetime.now().date()
# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)
# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

# print(get_completion("What is 1+1?"))

# ## Email in "pirate tone"
# customer_email = """
# Arrr, I be fuming that me blender lid \
# flew off and splattered me kitchen walls \
# with smoothie! And to make matters worse,\
# the warranty don't cover the cost of \
# cleaning up me kitchen. I need yer help \
# right now, matey!
# """
# style = """American English \
# in a calm and respectful tone
# """
# prompt = f"""Translate the text \
# that is delimited by triple backticks 
# into a style that is {style}.
# text: ```{customer_email}```
# """

# print(prompt)
# response = get_completion(prompt)
# print(response)



### Same as ^^^ using Langchain
### from langchain.chat_models import ChatOpenAI = creates chatGPT object
### from langchain_core.prompts import ChatPromptTemplate

# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
# chat = ChatOpenAI(temperature=0.0, model=llm_model)
# # print(chat)

# template_string = """Translate the text that is delimited by triple backticks \
# into a style that is {style}. \
# text: ```{text}```
# """

# prompt_template = ChatPromptTemplate.from_template(template_string)
# # print(prompt_template.messages[0].prompt)
# # print(prompt_template.messages[0].input_variables)

# customer_style = """American English in a calm and respectful tone"""
# customer_email = """
# Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls \
# with smoothie! And to make matters worse, the warranty don't cover the cost of \
# cleaning up me kitchen. I need yer help right now, matey!
# """

# customer_messages = prompt_template.format_messages(style=customer_style,
#                                                     text=customer_email)

# # print(type(customer_messages)) ## >>> Class List
# # print(type(customer_messages[0])) ## >>> <class 'langchain_core.messages.human.HumanMessage'>
# # print(customer_messages[0]) ## Basically the prompt you'd expect it to create

# # Call the LLM to translate to the style of the customer message
# customer_response = chat.invoke(customer_messages) ## Add invoke for upgrade

# # print(customer_response.content)
# #  >>> I am quite upset that my blender lid came off and caused my smoothie to splatter all over my kitchen walls. Additionally, the warranty does not cover the cost of cleaning up my kitchen. 
# # >>> I would greatly appreciate your assistance at this time. Thank you kindly.'''

# service_reply = """Hey there customer, the warranty does not cover \
# cleaning expenses for your kitchen because it's your fault that \
# you misused your blender by forgetting to put the lid on before \
# starting the blender. Tough luck! See ya!
# """

# service_style_pirate = """
# a polite tone that speaks in English Pirate
# """

# service_messages = prompt_template.format_messages(
#     style=service_style_pirate,
#     text=service_reply)

# # print(service_messages) ## >>> [HumanMessage(content="Translate the text that is delimited by triple backticks into a style that is \na polite tone that speaks in English Pirate\n. text: ```Hey there customer, the warranty does not cover cleaning expenses for your kitchen because it's your fault that you misused your blender by forgetting to put the lid on before starting the blender. Tough luck! See ya!\n```\n")]
# # print(service_messages[0].content) ## prints "style:, text:""

# service_response = chat.invoke(service_messages)
# # print(service_response.content) ## >> Ahoy there, me hearty customer! I be sorry to inform ye that the warranty be not coverin' the expenses o' cleaning yer kitchen. Ye see, 'tis yer own fault fer misusin' yer blender by forgettin' to put the lid on before startin' it. Tough luck, me heartie! Farewell and may the winds be in yer favor!



### React = Thought, Output, Observation -> Thought, Output, Observation
### Output Parsers - Define how we would like the LLM output to look like:

## {
##   "gift": False,
##   "delivery_days": 5,
##   "price_value": "pretty affordable!"
## }
## ^^^ How we'll want the output to be formatted

customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

## from langchain.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(review_template)
# print(prompt_template)

messages = prompt_template.format_messages(text=customer_review)
llm = ChatOpenAI(temperature=0.0, model=llm_model)
response = llm.invoke(messages)

# print(response.content) 
# print(type(response.content)) ## >> <class 'str'>

## You will get an error by running this line of code because'gift' is not a dictionary
## 'gift' is a string
## print(response.content.get('gift'))



### Parse the LLM output string into a Python dictionary
# from langchain.output_parsers import ResponseSchema
# from langchain.output_parsers import StructuredOutputParser
# print(help(ResponseSchema))

## Below for ResponseSchema to be put in list
gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
# print(format_instructions)

customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""
review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)

messages = prompt.format_messages(text=customer_review, 
                                  format_instructions=format_instructions)

# print(type(messages)) ## >>> <class 'list'>
# print(messages[0])
# print(type(messages[0].content)) ## >> <class 'string'>
response = llm.invoke(messages)
# print(type(response.content)) ## <class 'str'>
# print(response.content) ## String of '''json{'gift':true, 'delivery_days':'2',...}'''

output_dict = output_parser.parse(response.content)
# print(type(output_dict)) ## >>> <class 'dict'>
# print(output_dict)

print(output_dict.get('delivery_days')) ## >>> 2
print(output_dict.get('price_value'))

## Re-use PromTemplates... coupled with output parser to store in dict (or format)... to make easier for downstream processing
