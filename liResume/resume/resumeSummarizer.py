import os
import openai
import sys
from langchain_community.document_loaders import PyPDFLoader,TextLoader,NotionDirectoryLoader
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import re
from pprint import pprint

### Load PDF and split
# loader = PyPDFLoader("docs/ClintWickhamResume.pdf")
# pages = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1500,
#     chunk_overlap = 0
# )

# splits = text_splitter.split_documents(pages)

### Load PDF and split
# loader = PyPDFLoader("docs/ClintWickhamResume2.pdf")
# pages = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1500,
#     chunk_overlap = 0
# )

# splits = text_splitter.split_documents(pages)

# embedding = OpenAIEmbeddings()

# ## Create vector db of 1st resume
# persist_directory = 'docs/chroma/'
# vectordb = Chroma.from_documents(
#     documents=splits,
#     embedding=embedding,
#     persist_directory=persist_directory
# )
# vectordb.persist()

### Create vector db2 of splits
# persist_directory = 'docs/chroma2/'
# vectordb2 = Chroma.from_documents(
#     documents=splits,
#     embedding=embedding,
#     persist_directory=persist_directory
# )
# vectordb2.persist()

# ### Load all documents in vectorDB3
# docs = []
# loaders = [
#     PyPDFLoader("docs/ClintWickhamResume.pdf"),
#     PyPDFLoader("docs/ClintWickhamResume2.pdf"),
#     PyPDFLoader("docs/linkedinProfile.pdf")
# ]

# for loader in loaders:
#     docs.extend(loader.load())

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1500,
#     chunk_overlap = 0
# )

# splits = text_splitter.split_documents(docs)

# embedding = OpenAIEmbeddings()

# persist_directory = 'docs/chroma3/'

# vectordb3 = Chroma.from_documents(
#     documents=splits,
#     embedding=embedding,
#     persist_directory=persist_directory
# )

# vectordb3.persist()

### Retrieval
# persist_directory = 'docs/chroma2/'

# vectordb2 = Chroma(
#     persist_directory=persist_directory,
#     embedding_function=embedding,
# )

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectordb2.as_retriever()
# )

# question = 'What is a summary of this persons experience? Include common skills, strengths, and weaknesses'
# result = qa_chain.invoke({'query':question})
# # print(result['result'])
# # ''' >>>
# # Summary of Experience:
# # This person has a diverse background in recruitment and management roles. They have experience in technical recruiting, operations management, \
# # account management, and production management.They have a strong track record of building relationships, meeting recruitment targets, and \ 
# # driving process improvement. They have a functional understanding of the technology industry and have recruited for various technical positions. \
# # They have also demonstrated skills in account management, marketing, and database management. They have a Bachelor's degree in Health Services \
# # Administration with a minor in Business Administration.

# # Common Skills:
# # - Recruitment and sourcing
# # - Relationship building
# # - Account management
# # - Process improvement
# # - Database management
# # - Marketing
# # - Team management and mentoring

# # Strengths:
# # - Strong work ethic and positive attitude
# # - Ability to meet recruitment targets and place candidates quickly
# # - Excellent communication and interpersonal skills
# # - Ability to build and maintain relationships with clients and candidates
# # - Strong organizational and time management skills
# # - Technical industry knowledge and understanding

# # Weaknesses:
# # Based on the provided information, it is not possible to determine specific weaknesses of this person.
# # '''

### 2nd Resume
# question = 'What is a summary of this persons experience? Include common skills, strengths, and weaknesses'
# result = qa_chain.invoke({'query':question})
# print(result['result'])
'''
>>>
Summary of Clint Wickham's Experience:

Clint Wickham has a diverse range of experience in technical recruitment and operations management. He has a strong track record of successfully placing candidates and building relationships with clients. Clint is skilled in full-cycle recruitment, vendor management, and account management. He has a functional understanding of the technology industry and has recruited for various positions including software developers, systems engineers, network engineers, and project managers.

Clint's strengths include his sense of urgency, strong work ethic, positive attitude, coachability, and teamwork. He has a proven ability to meet deadlines and deliver results. He is also skilled in building and managing relationships, both with clients and employees. Clint has experience in process improvement initiatives and has developed tools to improve decision-making and visibility within operations.

In terms of weaknesses, specific information about Clint's weaknesses is not provided in the given context. It is important to note that weaknesses can vary from person to person and may not be explicitly mentioned in a professional summary.

Common Skills: Technical recruitment, full-cycle recruitment, vendor management, account management, relationship building, operations management, process improvement, decision-making, visibility, team management, customer coordination, marketing, database management.

Strengths: Sense of urgency, work ethic, positive attitude, coachability, teamwork, meeting deadlines, delivering results, building and managing relationships, process improvement, decision-making, visibility.

Weaknesses: Not provided in the given context.
'''

persist_directory = 'docs/chroma3/'
embedding = OpenAIEmbeddings()

vectordb3 = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

retriever=vectordb3.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

question = 'These are 2 resumes and a LinkedIn Profile. What is a summary of this \
    persons experience? Include common skills, strengths, and weaknesses?'
result = qa.invoke({"question": question})
# print(result['chat_history'][1].content)
'''>>>
Based on the provided information, the individual has experience in technical recruitment and operations management. They have worked in various roles such as Technical Sourcer, Technical Recruiter, and Operations Manager.

Common skills and strengths:
- Technical recruitment: The individual has a track record of successfully sourcing and hiring candidates for various technical positions, including distributed systems and applications development.
- Relationship building: They have demonstrated the ability to initiate, build, and manage relationships with candidates and clients, facilitating continued engagement and opportunities for growth.
- Mentoring and leadership: They have experience mentoring and onboarding new hires, helping them get up to speed and onboarded with best practices.
- Process improvement: They have driven process improvement initiatives, including developing database tools to improve decision-making and visibility in operations.
- Client management: They have experience gathering requirements, strategizing, and communicating with clients, serving as a trusted advisor for technical talent acquisition.

Weaknesses:
Based on the provided information, it is difficult to determine specific weaknesses as the focus is on the individual's strengths and accomplishments.'''

question = 'What tech skills does this person have'
result = qa.invoke({"question": question})
# print(result['answer'])
# print('*************************')
# print(result['chat_history'])
# print('*************************')

question1 = 'Looking at these documents, would this person be a good recruiter?'
question2 = 'Looking at these documents, would this person be a good technical support analyst?'
result1 = qa.invoke({"question": question1})
print(result1['answer'])
result2 = qa.invoke({'question': question2})
print('*************************')
print(result2['answer'])
print('*************************')
print(result2['chat_history'])
print('*************************')
print(qa.memory)