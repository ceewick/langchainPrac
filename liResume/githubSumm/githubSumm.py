from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from pprint import pprint

llm = ChatOpenAI()
loader = WebBaseLoader("https://github.com/ceewick")
docs = loader.load()
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = DocArrayInMemorySearch.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template(
    """
    You are a Technical Recruiter. You will look through this person's entire Github \
    page to understand their technical skills, strengths, weaknesses. 
    
    Answer the following question based on the provided context:
    
    <context>
    {context}
    </context>
    
    Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "what are some of my technical skills?"})
# print(response["answer"])
## >>> Based on the provided context, some of your technical skills include Python programming and web scraping. You have repositories related to OpenAI, Langchain tutorials, and personal projects.

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

chat_history = [
    HumanMessage(content="what are some of my technical skills?"), 
    AIMessage(content="Based on the provided context, some of your technical skills include Python \
              programming and web scraping. You have repositories related to OpenAI, Langchain tutorials, and personal projects.")
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "What should I learn next?"
})

print(response['answer'])