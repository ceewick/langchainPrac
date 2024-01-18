import datetime
import pandas as pd
import pprint
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser


### Outline = 
###LLMChain, 
### Sequential: ChainsSimpleSequentialChain, SequentialChain
### Router Chain

# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

df = pd.read_csv('Data.csv')
# print(df.head())


### LLM Chains
# # from langchain.chat_models import ChatOpenAI
# # from langchain.prompts import ChatPromptTemplate
# # from langchain.chains import LLMChain

# llm = ChatOpenAI(temperature=0.9, model=llm_model)

# prompt = ChatPromptTemplate.from_template(
#     "What is the best name to describe \
#     a company that makes {product}?"
# )

# chain = LLMChain(llm=llm, prompt=prompt)
# product = "Queen Size Sheet Set"
# # print(chain.invoke(product))
# response = chain.invoke(product)
# print(response['text'])



### Sequential Chains - one after another
### SimpleSequentialChain = one IO SimpleSequentialChains = multiple IO

### SimpleSequentialChain - Multiple chains with 1 I/O
### from langchain.chains import SimpleSequentialChain

# llm = ChatOpenAI(temperature=0.9, model=llm_model)

# # prompt template 1
# first_prompt = ChatPromptTemplate.from_template(
#     "What is the best name to describe \
#     a company that makes {product}?"
# )
# # Chain 1
# chain_one = LLMChain(llm=llm, prompt=first_prompt)

# # prompt template 2
# second_prompt = ChatPromptTemplate.from_template(
#     "Write a 20 words description for the following \
#     company:{company_name}"
# )
# # chain 2
# chain_two = LLMChain(llm=llm, prompt=second_prompt)

# overall_simple_chain = SimpleSequentialChain(
#     chains=[chain_one, chain_two],
#     verbose=True
#     )

# product = "Queen Size Sheet Set"

# # print(overall_simple_chain.invoke(product))
# res_dict = overall_simple_chain.invoke(product)
# print(res_dict['output'])



### Sequential Chain - Multiple I/O
# ## from langchain.chains import SequentialChain
# llm = ChatOpenAI(temperature=0.9, model=llm_model)

# # prompt template 1: translate to english
# first_prompt = ChatPromptTemplate.from_template(
#     "Translate the following review to english:"
#     "\n\n{Review}"
# )
# # chain 1: input= Review and output= English_Review
# chain_one = LLMChain(llm=llm, prompt=first_prompt, 
#                      output_key="English_Review"
#                     )
# # prompt template 3: Summarize Review
# second_prompt = ChatPromptTemplate.from_template(
#     "Can you summarize the following review in 1 sentence:"
#     "\n\n{English_Review}"
# )
# # chain 2: input= English_Review and output= summary
# chain_two = LLMChain(llm=llm, prompt=second_prompt, 
#                      output_key="summary"
#                     )
# # prompt template 3: translate to english
# third_prompt = ChatPromptTemplate.from_template(
#     "What language is the following review:\n\n{Review}"
# )
# # chain 3: input= Review and output= language
# chain_three = LLMChain(llm=llm, prompt=third_prompt,
#                        output_key="language"
#                       )
# # prompt template 4: follow up message
# fourth_prompt = ChatPromptTemplate.from_template(
#     "Write a follow up response to the following "
#     "summary in the specified language:"
#     "\n\nSummary: {summary}\n\nLanguage: {language}"
# )
# # chain 4: input= summary, language and output= followup_message
# chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
#                       output_key="followup_message"
#                      )
# # overall_chain: input= Review 
# # and output= English_Review,summary, followup_message
# overall_chain = SequentialChain(
#     chains=[chain_one, chain_two, chain_three, chain_four],
#     input_variables=["Review"],
#     output_variables=["English_Review", "summary",
#                       "followup_message"],
#     verbose=True
#     )

# review = df.Review[5]
# print(review)
# print(df.Review[0])
# pprint.pprint(overall_chain.invoke(review))



### Router Chain - Route input to chain, pending what input is
### Multiple sub-chains depending on the input
# from langchain.chains.router import MultiPromptChain
# -- Chain used when routing multiple prompt templates
# from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
# LLMRouterChain - Uses language model to route subchains
# OutputParser - Output to dictionary to determine which chain to use
# from langchain.prompts import PromptTemplate

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""

math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""

computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "physics", 
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
    },
    {
        "name": "math", 
        "description": "Good for answering math questions", 
        "prompt_template": math_template
    },
    {
        "name": "History", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "computer science", 
        "description": "Good for answering computer science questions", 
        "prompt_template": computerscience_template
    }
]

llm = ChatOpenAI(temperature=0, model=llm_model)

#chains to be called by router chain
destination_chains = {} 
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain  
    
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

## Default Chains - called when router doesnt know which subchain
## i.e. chain nothing to do w physics, math, science
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

## Full Router template
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
## Prompt Template
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(), #OutputParser = help chain decide which subchains route between
)
## Router Chain
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

## Overall Chain
chain = MultiPromptChain(
    router_chain=router_chain, 
    destination_chains=destination_chains, 
    default_chain=default_chain, 
    verbose=True
)

print(chain.invoke("What is black body radiation?")['text'])
# print("---------------------")
# print(chain.invoke("what is 2 + 2")['text])
# print("---------------------")
# print(chain.invoke("Why does every cell in our body contain DNA?")['text'])