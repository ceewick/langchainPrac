import json
import openai
import pprint
import time
import requests
from bs4 import BeautifulSoup
import openai
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from userAgents import user_agents, randomUserAgents

url = 'https://www.linkedin.com/in/clintwickham'
head = randomUserAgents()
# session = requests.Session()
# page = session.get(url, head)
# bs = BeautifulSoup(page.text)
# pprint.pprint(bs)

def soup(url,headers):
    session = requests.Session()
    req = session.get(url, headers=headers)
    bs = BeautifulSoup(req.text)
    return bs

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

client = openai.OpenAI()
bs = soup(url,head)
llm = ChatOpenAI()

loader = WebBaseLoader(url)
docs = loader.load()
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = DocArrayInMemorySearch.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

# document_chain.invoke({
#     "input": "how can langsmith help with testing?",
#     "context": [Document(page_content="langsmith can let you visualize test results")]
# })

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "what is a summary of the page?"})
print(response["answer"])

# prompt = f'''
# Give short summary of the linkedin page \
#     deliminated by three backticks.
    
#     Make the summary less than 50 words
    
#     ```{bs}```
# '''
# response = get_completion(prompt)
# print(response)


