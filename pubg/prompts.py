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

prompt_1 = f'''
Your task is to summarize the below lifetimeStats dictionary. \
This for a videogame Pubg.\
The dictionary is below, deliminates by triple "#".

Each column (except for the first column) starts with the \
players name.The first column are different stats.

Please compute and summarize the below metrics for each player.\
List the player and then the metrics. Example format below...

<Player Name>
1. kills/roundsPlayed: <compute metric>
2. losses/wins: <compute metric>
3. damage dealt: <compute metric>
4. revives/roundsPlayed: <compute metric>

Finally, after listing the metrics, use lifetimeStats dictionary \
data and create a summary of each player. Make each summary no \
more than 3 sentences.

The dictionary data is ###{lifetimeStatsDict}###
'''

response = get_completion(prompt_1)
# wild_response = get_completion_wild(prompt_1)
# print(f'''{response}\n**********\n{wild_response}''')
# print(response)

## prompt_2 = f'''
## Your task is to summarize the strengths and weaknesses of each player.

## Use data from both the lifetimeStats dictionary, and the metrics provided in previous \
## response. 

## The python lifetimeStats dictionary is below, deliminates by triple "#". \
## The previous response listing specific metrics is delimintated by triple backticks.

## In a funny way, summarize each player, and share their strengths/weaknesses. 

## For the summary, use information from both datasets. 

## The dictionary dataset is ###{lifetimeStatsDict}### . The metrics dataset \
## are ```{response}```.
## '''

prompt_2 = f'''
Your task is to summarize the strengths and weaknesses of each player \
from the below lifetimeStats dictionary. This for a videogame Pubg.

The python dictionary is below, deliminates by triple "#".

The first column are the categories of different stats to compare. \
Except for the first column, the other columns start with the players name, \
followed by their stats for each stat name listed in first column.  

In a funny way, summarize each player by their stats, and share their \
strengths/weaknesses. 

The dictionary data is ###{lifetimeStatsDict}###
'''

response_2 = get_completion(prompt_2)
# print(response_2)

prompt_3 = f'''
Use your previous response (deliminated by triple backticks) to summarize the pros \
and cons of each player in 1 bullet. Use no more than 50 words per bullet. 

List the players name, and then the pros/cons summary.  

Respond in the a funny way, the way you'd expect a professional esports gamer to respond. 

Your previous response was ```{response_2}```
'''

response_3 = get_completion(prompt_3)
# print(response_3)

responsesList = [response, response_3]

prompt_4 = f'''
Use data in the list of previous responses (deliminted by triple backticks) to compare \
players and determine which player is overall best and worst. 

Be funny in your response, pointing out obvious flaws. 

Provide your answer by stating the best and worst player, and then provide 2 sentences \
explaining your resoning. 

The list of your previous responses is ```{responsesList}```
'''

response_4 = get_completion(prompt_4)
# # print(response_4)

prompt_5 = f'''
Using the overall stats dictionary (deliminated by triple "#"), and your previous summary of \
strengths/weaknesses (deliminted by triple backticks), summarize and list the players best to \
worst and list. 
List your ranking of each player from 1-5, with 1 as best and 5 as worst. Example below:
1. <name> - best player - <funny summary>
2.
3.
4.
5. 
6.
7. <name> - worst player - <funny summary>

After the player name, provide a very very funny summary of each player, and why you ranked them \
in the position you did. No more than 3 sentences for the funny summary. 

The lifetime stats are ###{lifetimeStatsDict}###, and your list repsonse was ```{response_2}```
'''

response_5 = get_completion(prompt_5)
# print(response_5)

# # prompt_5 = f'''
# # Using the overall stats dictionary (deliminated by triple "#"), and your previous summary of \
# # strengths/weaknesses (deliminted by triple backticks), summarize and list the players best to \
# # worst and list. 
# # List your ranking of each player from 1-5, with 1 as best and 5 as worst. Example below:
# # 1. <name> - best player
# # 2.
# # 3.
# # 4.
# # 5. <name> - worst player

# # After, provide a very very funny summary of each player, and why you ranked them in the position \
# # you did. No more than 3 sentences for the funny summary. 

# # The lifetime stats are ###{lifetimeStatsDict}###, and your list repsonse was ```{response_2}```
# # '''


# prompt_6 = f'''
# Using the overall stats dictionary (deliminated by triple "#"), and your previous summary of \
# strengths/weaknesses (deliminted by triple backticks), list the players best to worst. \ 
# List your ranking of each player from 1-7, with 1 as best and 7 as worst. Example below:
# 1. <name> - best player
# 2.
# 3.
# 4.
# 5.
# 6.
# 7. <name> - worst player

# After the list, provide a very very funny summary of your rankings... comparing the players \
# and your reasoning for ranking them where you did. No more than 3 sentences for the funny summary. 

# The lifetime stats are ###{lifetimeStatsDict}###, and your pro/con response was ```{response_2}```
# '''

# response_6 = get_completion_wild(prompt_6)
# print(response_6)