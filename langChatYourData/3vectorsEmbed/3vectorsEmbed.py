import os
import openai
import sys
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import numpy as np

# # Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

# # print(len(splits))


# ### Embeddings
embedding = OpenAIEmbeddings()
# sentence1 = "i like dogs"
# sentence2 = "i like canines"
# sentence3 = "the weather is ugly outside"

# embedding1 = embedding.embed_query(sentence1)
# embedding2 = embedding.embed_query(sentence2)
# embedding3 = embedding.embed_query(sentence3)

# # print(np.dot(embedding1, embedding2))
# # print(np.dot(embedding1, embedding3))
# # print(np.dot(embedding2, embedding3))

# ## ^^^ If related, similar content have similar vectors

persist_directory = 'docs/chroma/'
# !rm -rf ./docs/chroma  # remove old database files if any

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

# # print(vectordb._collection.count())
## >>> 209

### Similarity Search

question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question,k=3)
# print(len(docs))
# print(docs[0].page_content)

vectordb.persist()

# question = "what did they say about matlab?"
# docs = vectordb.similarity_search(question,k=5)
# # print(docs[0])

question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(question,k=5)

for doc in docs:
    print(doc.metadata)

print(docs[4].page_content)