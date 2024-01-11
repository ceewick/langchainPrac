import datetime
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationTokenBufferMemory, ConversationSummaryBufferMemory


### Outline - Outline = ConversationBufferMemory, ConversationBufferWindowMemory
### ConversationTokenBufferMemory, ConversationSummaryMemory

# Current Date
current_date = datetime.datetime.now().date()
# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

# llm = ChatOpenAI(temperature=0.0, model=llm_model)
# memory = ConversationBufferMemory()
# conversation = ConversationChain(
#     llm=llm, 
#     memory=memory,
#     verbose=False  ## To remove green text (what machine doing)
#     )
# conversation = ConversationChain(
#     llm=llm, 
#     memory=memory,
#     verbose=True  ## To remove green text (what machine doing)
#     )

# print(conversation.predict(input="Hi, my name is Andrew"))
# print(conversation.predict(input="What is 1+1?"))
# print(conversation.predict(input="What is my name?"))

# conversation.predict(input="Hi, my name is Andrew")
# conversation.predict(input="What is 1+1?")
# conversation.predict(input="What is my name?") ### Adds all above to conversation list

## print(type(memory.buffer)) ## <class 'str'>
# print(memory.buffer) ## Summary of chain
# print(type(memory.load_memory_variables({}))) ## <class 'dict'>
# print(memory.load_memory_variables({}))

# memory = ConversationBufferMemory()

# memory.save_context({"input": "Hi"}, 
#                     {"output": "What's up"})

# print(memory.buffer)

# memory.load_memory_variables({})
# memory_dic = memory.load_memory_variables({})
# print(memory_dic.keys()) ## >> dict_keys(['history'])
# print(memory_dic)

# memory.save_context({"input": "Not much, just hanging"}, 
#                     {"output": "Cool"})
# print(memory.load_memory_variables({}))


### LLMs are stateless. As convo grows, memory constraint grows
### Langchain provides models to store, so doesnt count as token
### ConversationBufferWindowMemory - only keeps set amount of memory 
## from langchain.memory import ConversationBufferWindowMemory

# memory = ConversationBufferWindowMemory(k=1) #stores 1 value IO

# memory.save_context({"input": "Hi"},
#                     {"output": "What's up"})
# memory.save_context({"input": "Not much, just hanging"},
#                     {"output": "Cool"})
# print(memory.load_memory_variables({})) ## only last memory

# llm = ChatOpenAI(temperature=0.0, model=llm_model)
# memory = ConversationBufferWindowMemory(k=1)
# conversation = ConversationChain(
#     llm=llm, 
#     memory = memory,
#     verbose=False
# )

## conversation.predict(input="Hi, my name is Andrew")
## conversation.predict(input="What is 1+1?")
## conversation.predict(input="What is my name?") ### Adds all above to conversation list
## ^^^ Wont remember name because k=1



#### ConversationTokenBufferMemory - Limits # Tokens
### from langchain.memory import ConversationTokenBufferMemory
### from langchain.llms import OpenAI
    
# llm = ChatOpenAI(temperature=0.0, model=llm_model)
# memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
# memory.save_context({"input": "AI is what?!"},
#                     {"output": "Amazing!"})
# memory.save_context({"input": "Backpropagation is what?"},
#                     {"output": "Beautiful!"})
# memory.save_context({"input": "Chatbots are what?"}, 
#                     {"output": "Charming!"})
# print(memory.load_memory_variables({}))
## ^^^ Increase token limit to get full convo in output



### ConversationSummaryMemory
### Write summ of convo so far and save that summ in memory
### from langchain.memory import ConversationSummaryBufferMemory

schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

llm = ChatOpenAI(temperature=0.0, model=llm_model)
## Less token makes summary, more token is full convo when load
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})

print(memory.load_memory_variables({}))


conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

print(conversation.predict(input="What would be a good demo to show?"))
print(memory.load_memory_variables({}))

### ^^^ For chat, but also good if have additional data coming in to store

### Additional Memory Types
## Vector DB