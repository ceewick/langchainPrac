import pandas as pd
from pprint import pprint
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import openai

from test import df, lifetimeStatsDict

# print(help(pd))

# df=pd.read_csv('lifetimeStats.csv')
# df = df.drop(df.columns=='Unnamed')

# df.index.name = 'Categories'
# pprint(lifetimeStatsDict)

llm = openai.OpenAI()
langLlm = ChatOpenAI()

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = llm.chat.completions.create(
        model=model,
        messages=messages,
        temperature=.5, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def get_completion_wild(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = llm.chat.completions.create(
        model=model,
        messages=messages,
        temperature=.9, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

# with open('lifetimeStats.csv','r') as lifetimeStatsFile:

# prompt_1 = f'''
# Your task is to summarize the below lifetimeStats dictionary. \
# This for a videogame Pubg.\
# The csv file is below, deliminates by triple "#".

# Except for the first column, each column are the players name. \ 
# The first column are the categories of different metrics/stats.

# Please compute and summarize the below stats for each player:
# 1. kills/roundsPlayed: <compare answer>
# 2. losses/wins: <compare answer>
# 3. damage dealt: <compare damage dealt>
# 4. revives/roundsPlayed: <compute and share answer>

# Finally, summarize the lifetime stats for each player, \
# which is each column except the first column (first column are metrics).

# The dictionary data is ###{lifetimeStatsDict}###
# '''

prompt_2 = f'''
Your task is to summarize the strengths and weaknesses of each player \
in the below lifetimeStats dictionary. This for a videogame Pubg.

The python dictionary is below, deliminates by triple "#".

The first column are the categories of different metrics/stats to compare.
Except for the first column, the other columns start with the players name, \
followed by their stats for each stat listed in first column.  

In a funny way, summarize each player by their stats, and share their \
strengths/weaknesses. 

The dictionary data is ###{lifetimeStatsDict}###
'''

response = get_completion(prompt_2)
# print(response)

prompt_3 = f'''
Use your previous response (deliminated by triple backticks) to summarize the pros \
and cons of each player in 1 bullet. Use no more than 50 words per bullet. 

Respond in the a funny way, the way you'd expect a professional esports gamer to respond. 

Your previous response was ```{response}```
'''

response_2 = get_completion(prompt_3)
# print(response_2)

responsesList = [response, response_2]

prompt_4 = f'''
Use the list of your previous 2 responses (deliminted by triple backticks) to compare players \
and determine which player is overall best and worst. 

Be funny in your response, pointing out obvious flaws. 

Provide your answer by stating the best and worst player, and then provide 2 sentences \
explaining your resoning. 

The list of your previous responses is ```{responsesList}```
'''

response_3 = get_completion(prompt_4)
# print(response_3)

# prompt_5 = f'''
# Using the overall stats dictionary (deliminated by triple "#"), and your previous summary of \
# strengths/weaknesses (deliminted by triple backticks), summarize and list the players best to \
# worst and list. 
# List your ranking of each player from 1-5, with 1 as best and 5 as worst. Example below:
# 1. <name> - best player
# 2.
# 3.
# 4.
# 5. <name> - worst player

# After, provide a very very funny summary of each player, and why you ranked them in the position \
# you did. No more than 3 sentences for the funny summary. 

# The lifetime stats are ###{lifetimeStatsDict}###, and your list repsonse was ```{response_2}```
# '''

# response_4 = get_completion_wild(prompt_5)
# print(response_4)

prompt_6 = f'''
Using the overall stats dictionary (deliminated by triple "#"), and your previous summary of \
strengths/weaknesses (deliminted by triple backticks), list the players best to worst. \ 
List your ranking of each player from 1-5, with 1 as best and 5 as worst. Example below:
1. <name> - best player
2.
3.
4.
5. <name> - worst player

After the list, provide a very very funny summary of your rankings... comparing the players \
and your reasoning for ranking them where you did. No more than 5 sentences for the funny summary. 

The lifetime stats are ###{lifetimeStatsDict}###, and your list repsonse was ```{response_2}```
'''

response_5 = get_completion_wild(prompt_6)
print(response_5)