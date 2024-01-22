import os
import openai
import sys
from langchain_community.document_loaders import PyPDFLoader,TextLoader,NotionDirectoryLoader
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import re
from pprint import pprint

### Load PDF and split
## loader = PyPDFLoader("docs/ClintWickhamResume.pdf")
## pages = loader.load()
## text_splitter = RecursiveCharacterTextSplitter(
##     chunk_size = 1500,
##     chunk_overlap = 0
## )

## splits = text_splitter.split_documents(pages)

embedding = OpenAIEmbeddings()

### Create vector db of splits
## persist_directory = 'docs/chroma/'
## vectordb = Chroma.from_documents(
##     documents=splits,
##     embedding=embedding,
##     persist_directory=persist_directory
## )
## vectordb.persist()

persist_directory = 'docs/chroma/'

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

# question = 'What is a summary of this persons experience? Include common skills, strengths, and weaknesses'
# result = qa_chain.invoke({'query':question})
# print(result['result'])
# ''' >>>
# Summary of Experience:
# This person has a diverse background in recruitment and management roles. They have experience in technical recruiting, operations management, \
# account management, and production management.They have a strong track record of building relationships, meeting recruitment targets, and \ 
# driving process improvement. They have a functional understanding of the technology industry and have recruited for various technical positions. \
# They have also demonstrated skills in account management, marketing, and database management. They have a Bachelor's degree in Health Services \
# Administration with a minor in Business Administration.

# Common Skills:
# - Recruitment and sourcing
# - Relationship building
# - Account management
# - Process improvement
# - Database management
# - Marketing
# - Team management and mentoring

# Strengths:
# - Strong work ethic and positive attitude
# - Ability to meet recruitment targets and place candidates quickly
# - Excellent communication and interpersonal skills
# - Ability to build and maintain relationships with clients and candidates
# - Strong organizational and time management skills
# - Technical industry knowledge and understanding

# Weaknesses:
# Based on the provided information, it is not possible to determine specific weaknesses of this person.
# '''

pprint(qa_chain)