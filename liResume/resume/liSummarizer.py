
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


# ## Load PDF and split
# loader = PyPDFLoader("docs/linkedinProfile.pdf")
# pages = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1500,
#     chunk_overlap = 0
# )

# splits = text_splitter.split_documents(pages)

# embedding = OpenAIEmbeddings()

# ## Create vector db of 1st resume
# persist_directory = 'docs/chromaLi/'
# liVectorDb = Chroma.from_documents(
#     documents=splits,
#     embedding=embedding,
#     persist_directory=persist_directory
# )
# liVectorDb.persist()



### Retrieval
persist_directory = 'docs/chromaLi/'
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embedding = OpenAIEmbeddings()

liVectorDb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
)
retriever=liVectorDb.as_retriever()

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

question = "Looking at this LinkedIn profile pdf, please summarize. Summarize the persons \
    overall experisnce, and output the summary. Then provide a bulleted list with some of their \
    areas of professional strength (including skills, experience, or quality employers)"
result = qa.invoke({"question": question})

# print(result)
# print('****************************************')
print(result['answer'])
print('****************************************')

question = "Outside of recruiting, what are some potential new roles a person with this \
    experience could consider?"
result2 = qa.invoke({"question": question})
answer = result2['answer']

print(answer)
print('****************************************')
# print(qa.memory)

question = '''Considering the end of your previous response {result2}, what are some \
    of the additional skills or qualifications this person might want to learn or add to \
    resume to show the transferable skills?'''

result3 = qa.invoke({"question": question})
print(result3['answer'])
'''Based on the previous response, some additional skills or qualifications that this person might want to learn or add to their resume to show transferable skills are:
1. Project Management: Demonstrating proficiency in project management skills would showcase the ability to effectively manage and coordinate resources, set goals, and deliver results.
2. Database Management: Highlighting experience or knowledge in database management, particularly with tools like Microsoft Access and Excel, would showcase the ability to analyze and manipulate data to improve operations and visibility.
3. Account Management: Emphasizing experience in account management would demonstrate the ability to effectively communicate with customers, manage work schedules and sales timelines, and coordinate resources to meet customer needs.
4. Process Improvement: Showcasing a strong understanding of process improvement methodologies and experience in implementing process improvement initiatives would highlight the ability to identify inefficiencies and drive productivity.
5. Sales and Marketing: Demonstrating skills in sales and marketing, such as developing pricing strategies, managing inventory, forecasting, and customer communication, would showcase a well-rounded skill set and the ability to contribute to business growth.
6. Leadership and Team Management: Highlighting experience in hiring, training, and managing employees would showcase leadership and team management skills, which are valuable in various roles and industries.
7. Technical Recruitment: Emphasizing expertise in technical recruitment, including experience with applicant tracking systems (ATS) like Ultipro, MaxHire, Bullhorn, and iCIMS, would showcase the ability to identify and place candidates in various technical roles.
8. IT Knowledge: Demonstrating knowledge and familiarity with various programming languages (such as .NET, JavaScript, Java, Ruby, Python, PHP), operating systems (Linux, Windows), and IT roles (Software Developer, Mobile Developer, Data Engineer, Systems Engineer, Network Engineer, DevOps Engineer) would showcase technical expertise and the ability to understand and recruit for IT positions.
These additional skills and qualifications would help showcase the candidate's versatility and transferable skills across different industries and roles.
'''
print('****************************************')
print(qa.memory)