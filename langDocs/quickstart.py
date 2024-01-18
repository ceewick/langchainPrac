# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain, create_history_aware_retriever
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain.agents import create_openai_functions_agent, AgentExecutor
# from langchain import hub


# from pprint import pprint

# llm = ChatOpenAI()

# pprint(llm.invoke("how can langsmith help with testing?").content)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are world class technical documentation writer."),
#     ("user", "{input}")
# ])

# chain = prompt | llm 
# print("***************************************")
# pprint(chain.invoke({"input": "how can langsmith help with testing?"}).content)
# response = chain.invoke({"input": "how can langsmith help with testing?"})
# print(response).
# print(help(response))


## ChatModel Output is message, but easier to work with strings
# output_parser = StrOutputParser()
# chain = prompt | llm | output_parser
# response = chain.invoke({"input": "how can langsmith help with testing?"})
# print(response)


### RETRIEVAL CHAIN - Useful when too much data
### In this process, we will look up relevant documents from a Retriever and then pass them into the prompt. A Retriever can be backed by anything - a SQL table, the 
### internet, etc - but in this instance we will populate a vector store and use that as a retriever.

### Simple Vectorstore - pip install faiss-cpu, could be chroma

## from langchain_community.document_loaders import WebBaseLoader
## from langchain_community.vectorstores import FAISS
## from langchain.text_splitter import RecursiveCharacterTextSplitter
## from langchain.chains.combine_documents import create_stuff_documents_chain

# loader = WebBaseLoader("https://docs.smith.langchain.com/overview")

# docs = loader.load()
# embeddings = OpenAIEmbeddings()

# text_splitter = RecursiveCharacterTextSplitter()
# documents = text_splitter.split_documents(docs)
# vector = FAISS.from_documents(documents, embeddings)

# prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""")

# document_chain = create_stuff_documents_chain(llm, prompt)

## Could run ourselves w below, but would rather use retriever set up
## from langchain_core.documents import Document

## response = document_chain.invoke({
##     "input": "how can langsmith help with testing?",
##     "context": [Document(page_content="langsmith can let you visualize test results")]
## })
## print(response)

## from langchain.chains import create_retrieval_chain

# retriever = vector.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
# print(response["answer"])
## pprint(response)
## print(type(response)) # <class 'dict'>



### CONVERSATION RETRIEVAL CHAINS
## e can still use the create_retrieval_chain function, but we need to change two things:
## The retrieval method should now not just work on the most recent input, but rather should take the whole history into account.
## The final LLM chain should likewise take the whole history into account
## from langchain.chains import create_history_aware_retriever
## from langchain_core.prompts import MessagesPlaceholder
## from langchain_core.messages import HumanMessage, AIMessage

## First we need a prompt that we can pass into an LLM to generate this search query
# prompt = ChatPromptTemplate.from_messages([
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
# ])
# retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# chat_history = [
#     HumanMessage(content="Can LangSmith help test my LLM applications?"), 
#     AIMessage(content="Yes!")
# ]

# print(retriever_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# }))

## You should see that this returns documents about testing in LangSmith. This is because the LLM generated a new query, combining the chat history with the follow up question.

## Now that we have this new retriever, we can create a new chain to continue the conversation with these retrieved documents in mind.

# retriever_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# })

# print(retriever_chain)
# print(response['answer'])

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "Answer the user's questions based on the below context:\n\n{context}"),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
# ])

# document_chain = create_stuff_documents_chain(llm, prompt)

# retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"),
#                 AIMessage(content="Yes!")
#                 ]

# print(retrieval_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# }))

# response = retrieval_chain.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# })

# pprint(response)
# print(response['answer'])
# print(chat_history)
# print(response[.keys())] # >> dict_keys(['chat_history', 'input', 'context', 'answer'])
# print(response['chat_history']) # >> [HumanMessage(content='Can LangSmith help test my LLM applications?'), AIMessage(content='Yes!')]



### AGENTS - quickstart agents
##  We've so far create examples of chains - where each step is known ahead of time. The final thing we will create is an agent - where the LLM decides what steps to take.

## Note: for this example we will only show how to create an agent using OpenAI models, as local models are not reliable enough yet.

## One of the first things to do when building an agent is to decide what tools it should have access to. For this example, we will give the agent access two tools:

## The retriever we just created. This will let it easily answer questions about LangSmith
## A search tool. This will let it easily answer questions that require up to date information.

## from langchain.tools.retriever import create_retriever_tool

# retriever_tool = create_retriever_tool(
#     retriever,
#     "langsmith_search",
#     "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
# )

## from langchain_community.tools.tavily_search import TavilySearchResults
## Tavily API

## >>> pip install langchainHub
## from langchain_openai import ChatOpenAI
## from langchain import hub
## from langchain.agents import create_openai_functions_agent
## from langchain.agents import AgentExecutor

# search = TavilySearchResults()
## print(search.invoke("what is the weather in SF"))

# tools = [retriever_tool, search]

## Get the prompt to use - you can modify this!
# prompt = hub.pull("hwchase17/openai-functions-agent")
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# agent = create_openai_functions_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

## print(agent_executor.invoke({"input": "how can langsmith help with testing?"}))
# agent_executor.invoke({"input": "what is the weather in SF?"})
## agent_response = agent_executor.invoke({"input": "what is the weather in SF?"})

# print(agent_response['output'])

# chat_history = [
#     HumanMessage(content="Can LangSmith help test my LLM applications?"), 
#     AIMessage(content="Yes!")
#     ]

# agent_executor.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how"
# })



### LANGSERVE - Build REST API endpoints for client
#!/usr/bin/env python
from typing import List

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes
from pprint import pprint

## 1. Load Retriever
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

## for doc in documents:
##     for i in doc:
##         print(i)
## lSplit = '***********************'
## pprint(f'''
##     docstype:{type(docs)}\n{lSplit}\ndocsContens:{docs}\n{lSplit}\n \
##         retriever:{retriever}\n{lSplit}\ndocuments:{documents}
## ''') 

## 2. Create Tools
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
search = TavilySearchResults()
tools = [retriever_tool, search]

## 3. Create Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
## pprint(prompt)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

## 4. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

## 5. Adding chain route

# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas
class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )


class Output(BaseModel):
    output: str

add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
    ''' python serve.py we should see our chain being \
        served at localhost:8000.'''