from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from pprint import pprint

persist_directory = '../3vectorsEmbed/docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
##print(vectordb._collection.count())
## >>> 209

question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
## print(len(docs))
## >>> 3

## from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

## print(llm.predict("Hello world!"))
## ^^^ predict deprecated, use invoke
## >>> Hello! How can I assist you today?

# print(llm.invoke('Hello World!'))
# ## >>> content='Hello! How can I assist you today?'
# print(llm.invoke('Hello World!').content)

# Build prompt
## from langchain_core.prompts import PromptTemplate
template = """Use the following pieces of context to answer the question at the end. \
    If you don't know the answer, just say that you don't know, don't try to make up an answer. \
    Use three sentences maximum. Keep the answer as concise as possible. \
    Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:
"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain
## from langchain.chains import RetrievalQA
question = "Is probability a class topic?"
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# result = qa_chain.invoke({"query": question})
# print(result["result"])
## >>> Yes, probability is a topic in this class. Thanks for asking!


### Memory
## from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
### conversationRetrievalChain 
## from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

question = "Is probability a class topic?"
result = qa.invoke({"question": question})
# print(result['answer'])
# ## >>>> Yes, probability is a topic that will be covered in this class. The instructor assumes familiarity with basic probability and statistics, so it is expected that students have prior knowledge of random variables, expectation, variance, and other related concepts.

question = "why are those prerequesites needed?"
result = qa.invoke({"question": question})
# print(result['answer'])

print(qa)