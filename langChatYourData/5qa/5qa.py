from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from pprint import pprint

persist_directory = '../3vectorsEmbed/docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
## print(vectordb._collection.count())
## 209

question = "What are major topics for this class?"
# docs = vectordb.similarity_search(question,k=3)
## print(len(docs))
## >>> 3

## from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


### Notes
'''
Retrieval QA Chain
-- Question applied to Vector Store as a query
-- Vector store provides K relevant documents
-- Docs and original question are sent to LLM for answer
-- Stuffs all documents into prompt - good bc only 1 call to LLM, but limitation if too many docs

Other Kinds QA Chain Retrival:
-- Usually if have lots of docs to pass through

1) Map-reduce
-- Each document sent to LLM by itself to get original answer... then all answers combined w final call to llm 
-- lot slower
2) Refine
3) Map_rerank

'''



### Retrieval QA Chain
### from langchain.chains import RetrievalQA
# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectordb.as_retriever()
# )

# result = qa_chain.invoke({'query':question})

# print(result)
## >>> {'query': 'What are major topics for this class?', 'result': 
## 'The major topics for this class are machine learning and its various subfields.'}
## print(result['result'])



### Prompt - takes in document and question and passes to LLM
## from langchain_core.prompts import PromptTemplate

# # Build prompt
template = """
Use the following pieces of context to answer the question at the end. \
If you don't know the answer, just say that you don't know, don't try to make up \
an answer. Use three sentences maximum. Keep the answer as concise as possible. \
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:
"""


QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# # Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# question = "Is probability a class topic?"
# result = qa_chain.invoke({'query':question})
# print(result)
# print(result.keys())
# >>> dict_keys(['query', 'result', 'source_documents'])
# print(result['result'])
# print('**********************************')
## >>> Yes, probability is a topic covered in this class. Thanks for asking!
# print(result["source_documents"][0])



### RetrievalQA chain types
'''Map_reduce - 4 call to llm... I/O for each doc. After, combo to final chain
All responses into doc chain... sysMessage is 4 summaries and then question
'''
# qa_chain_mr = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectordb.as_retriever(),
#     chain_type="map_reduce"
# )

# question = "Is probability a class topic?"
# result = qa_chain_mr.invoke({"query": question})
# print(result["result"])
# print('**********************************')
## >>> Based on the provided information, it is \ 
## not clear whether probability is a specific topic covered in the class.
## Answer each doc indiv... if spread across documents, it's hard to fit in context

'''
If you wish to experiment on the LangChain plus platform:

Go to langchain plus platform and sign up
-- https://smith.langchain.com/o/e70b6d42-870e-539f-8636-f31aa84c806f/
Create an API key from your account's settings
Use this API key in the code below
uncomment the code
Note, the endpoint in the video differs from the one below. Use the one below.

#import os
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
#os.environ["LANGCHAIN_API_KEY"] = "..." # replace dots with your api key

Langsmith = See under the hood
'''

## Refine
''''
invokes RetrivealQA -> refineDocumentsChain 
4 sequential calls to LLM chain... each new qa answer refined/combo w previous

'''
# qa_chain_mr = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectordb.as_retriever(),
#     chain_type="refine"
# )
# result = qa_chain_mr.invoke({"query": question})
# print(result["result"])
# print('**********************************')
'''
>>> Based on the additional context provided, it is still not explicitly mentioned whether probability 
is a specific class topic. The instructor mentions going over 
statistics and algebra as refreshers in the discussion sections, but it is not clear if probability 
is included in these refreshers. Additionally, the instructor mentions using the discussion 
sections to cover extensions for the material taught in the main lectures, but it is not specified 
if probability is one of these extensions. Therefore, the original answer remains valid.
'''


### RetrievalQA - Fails to preserve history
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

question = "Is probability a class topic?"
result = qa_chain.invoke({"query": question})
print(result["result"])

question = "why are those prerequesites needed?"
result = qa_chain.invoke({"query": question})
print(result["result"])