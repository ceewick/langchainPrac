from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

llm = ChatOpenAI()
loader = WebBaseLoader("https://pubglookup.com/account/scoreboards/E5G-39H")
docs = loader.load()
print(docs)

# embeddings = OpenAIEmbeddings()
# text_splitter = RecursiveCharacterTextSplitter()
# documents = text_splitter.split_documents(docs)
# vector = DocArrayInMemorySearch.from_documents(documents, embeddings)

# prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""")

# document_chain = create_stuff_documents_chain(llm, prompt)

# retriever = vector.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)


# response = retrieval_chain.invoke({"input": "what is summary of page?"})
# print(response["answer"])