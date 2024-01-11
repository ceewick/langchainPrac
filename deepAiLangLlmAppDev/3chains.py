import datetime
import pandas as pd
import pprint
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

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

### SimpleSequentialChain
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


### Sequential Chain
## from langchain.chains import SequentialChain
llm = ChatOpenAI(temperature=0.9, model=llm_model)

# prompt template 1: translate to english
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)
# chain 1: input= Review and output= English_Review
chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                     output_key="English_Review"
                    )
# prompt template 3: Summarize Review
second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)
# chain 2: input= English_Review and output= summary
chain_two = LLMChain(llm=llm, prompt=second_prompt, 
                     output_key="summary"
                    )
# prompt template 3: translate to english
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)
# chain 3: input= Review and output= language
chain_three = LLMChain(llm=llm, prompt=third_prompt,
                       output_key="language"
                      )
# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
# chain 4: input= summary, language and output= followup_message
chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                      output_key="followup_message"
                     )
# overall_chain: input= Review 
# and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary",
                      "followup_message"],
    verbose=True
    )

review = df.Review[5]
pprint.pprint(overall_chain.invoke(review))