import json
import openai
import pprint
import time

client = openai.OpenAI()

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

with open('snapGlassdoor2017.json') as jsonFile:
    jsonData = json.load(jsonFile)

primary = jsonData

total_reviews = {}
pros = []
cons = []
proConDic = {}

# for i in primary.values():
    # total_reviews.append(i)

# pprint.pprint(primary['1'])

for i in primary.values():
    pros.append(i['pro'])
    cons.append(i['con'])
    # print(i['pro'])

prosConsList = [pros,cons]

# proConDic['pro'] = pros
# proConDic['con'] = cons

# prompt = f"""
# Your task is to summarize positive reviews in this list. 
# Summarize common trends. 

# Summarize the review below, delimited by triple \
# backticks, in at most 50 words. 

# Reviews: ```{positive}```
# """

# prompt = f"""
# Your task is to summarize positive reviews in this list \
# and look for commons trends. 

# Summarize trends in the positive and negative reviews below, \
# delimited by triple backticks. 

# Positive Reviews: ```{pros[0:15]}```
# Negative Reviews: ```{cons[0:15]}```
# """

# response = get_completion(prompt)
# print(response)

### Response to last prompt
# '''
# >>>Positive trends in the reviews:
# - Good perks and benefits, including free meals and phone bill reimbursement
# - Positive company culture, with approachable and friendly coworkers
# - Exciting and everchanging work environment
# - Opportunities for growth and career development
# - Cool and innovative work environment
# - Diversity of employees
# - Good pay and benefits
# - Positive company vision and product
# - Fun and energetic atmosphere
# - Flexibility in work
# - Room for employees to grow
# - Great facilities and amenities
# - Famous and well-known company

# Negative trends in the reviews:
# - Bad hours and work-life balance
# - Lack of job security
# - Closed and political environment
# - Technical debt and poor architecture decisions
# - Lack of empowerment and trust for senior engineers
# - Working from home frowned upon
# - Lack of transparency
# - Feeling undervalued and mistreated
# - Lack of diversity in management
# - Confusion and overlapping roles and responsibilities
# - Long hours and demanding workload
# - Lack of internal tools and resources
# - Spread out offices requiring walking between them
# - Backloaded vesting schedule for equity package'''
    
prompt = f"""
Summarize common themes from the two lists below (deliminated\
by three backticks). 

For both the pros and cons lists, identify common list of emotions \  
that the writers of the following reviews are expressing. 
Include no more than five items in the list. 
Format your answer as two lists of \
lower-case words separated by commas.

# Positive Reviews: ```{prosConsList[0][:15]}```
# Negative Reviews: ```{prosConsList[1][:15]}```
"""

try:
    beg = 0
    count = 15
    for revs in range(len(prosConsList[0])):
        prompt = f"""
        Summarize common themes from the two lists below (deliminated\
        by three backticks). 
        
        For both the pros and cons lists, identify common list of emotions \  
        that the writers of the following reviews are expressing. 
        Include no more than five items in the list. 
        Format your answer as two lists of \
        lower-case words separated by commas.
        
        # Positive Reviews: ```{prosConsList[0][beg:count]}```
        # Negative Reviews: ```{prosConsList[1][beg:count]}```
        """
        response = get_completion(prompt)
        print(response)
        count += 15
        beg += 16
        time.sleep(22)
except:
    print('over')