import os
import openai
import sys
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import SVMRetriever,TFIDFRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI, OpenAI
# from langchain.llms import OpenAI

from pprint import pprint
import shutil, os
from pathlib import Path


## >>> pip install lark

# sys.path.append('../..')

persist_directory = '../3vectorsEmbed/docs/chroma/'

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# print(vectordb._collection.count())
## >>> 209

# texts = [
#     """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
#     """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
#     """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
# ]

# smalldb = Chroma.from_texts(texts, embedding=embedding)

# question = "Tell me about all-white mushrooms with large fruiting bodies"

# sim_search_small = smalldb.similarity_search(question, k=2) ## k=2 -> return 2 most relavent docs
# max_relevance_small = smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3) ## return 2 docs but fetch 3
# # print(smalldb.similarity_search(question, k=2))
# # print(smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3))
# print(sim_search_small[0].page_content, sim_search_small[1].page_content)
# print('********************************')
# print(max_relevance_small[0].page_content, max_relevance_small[1].page_content)


### Max Marginal Relevance
'''Addressing Diversity: Maximum marginal relevance
Last class we introduced one problem: how to enforce diversity in the search results.

Maximum marginal relevance strives to achieve both relevance to the query and 
diversity among the results.'''

# question = "what did they say about matlab?"
# docs_ss = vectordb.similarity_search(question,k=3)
# print(docs_ss[0].page_content[:100])
# print('********************************')
# print(docs_ss[1].page_content[:100])
'''
Both return
>>> those homeworks will be done in either MATLA B or in Octave, which is sort of — I
know some people
'''

# question = "what did they say about matlab?"

# docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)
# print(docs_mmr[0].page_content[:100])
# print('********************************')
# print(docs_mmr[1].page_content[:100])
'''
>>> those homeworks will be done in either MATLA B or in Octave, which is sort of — I
know some people
********************************
algorithm then? So what’s different? How come  I was making all that noise earlier about
least squa
'''


### Specificity
'''
Addressing Specificity: working with metadata
In last lecture, we showed that a question about the third lecture 
can include results from other lectures as well.

To address this, many vectorstores support operations on metadata.

metadata provides context for each embedded chunk.
'''

# question = "what did they say about regression in the third lecture?"

# docs = vectordb.similarity_search(
#     question,
#     k=3,
#     filter={"source":"docs/MachineLearning-Lecture03.pdf"}
# )

# for d in docs:
#     print(d.metadata)

'''
Addressing Specificity: working with metadata using self-query retriever
But we have an interesting challenge: we often want to infer the metadata from the 
query itself.

To address this, we can use SelfQueryRetriever, which uses an LLM to extract:

The query string to use for vector search
A metadata filter to pass in as well
Most vector databases support metadata filters, so this doesn't require any 
new databases or indexe

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI, OpenAI
'''

# metadata_field_info = [
#     AttributeInfo(
#         name="source",
#         description="The lecture the chunk is from, should be one of \
#             `docs/MachineLearning-Lecture01.pdf`, `docs/MachineLearning-Lecture02.pdf`, \
#                 or `docs/MachineLearning-Lecture03.pdf`",
#         type="string",
#     ),
#     AttributeInfo(
#         name="page",
#         description="The page from the lecture",
#         type="integer",
#     ),
# ]

# document_content_description = "Lecture notes"

# # Initialize the LLM
# llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)

# # Set up self-query
# retriever = SelfQueryRetriever.from_llm(
#     llm,
#     vectordb,
#     document_content_description,
#     metadata_field_info,
#     verbose=True ## See what's going on under the hood
# )

# question = "what did they say about regression in the third lecture?"
# docs = retriever.get_relevant_documents(question)
# # docs = retriever.invoke(question)

# for d in docs:
#     print(d.metadata)



### Compresion
'''Additional tricks: compression
Another approach for improving the quality of retrieved docs is compression.

Information most relevant to a query may be buried in a document with a lot of irrelevant text.

Passing that full document through your application can lead to more expensive LLM calls and poorer responses.

Contextual compression is meant to fix this.

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
'''

# def pretty_print_docs(docs):
#     print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

# # Wrap our vectorstore
# llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
# compressor = LLMChainExtractor.from_llm(llm)

# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=vectordb.as_retriever()
# )

# question = "what did they say about matlab?"
# compressed_docs = compression_retriever.get_relevant_documents(question)
# pretty_print_docs(compressed_docs)

# ## 
# '''^^^ Get shorter summary of each doc, but some stuff repeated bc semantic search under 
# the hood. Combine w MMR to delete duplicates
# '''

# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=vectordb.as_retriever(search_type = "mmr")
# )

# question = "what did they say about matlab?"
# compressed_docs = compression_retriever.get_relevant_documents(question)
# pretty_print_docs(compressed_docs)


### Other Retrievers
'''
from langchain.retrievers import SVMRetriever
from langchain.retrievers import TFIDFRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
'''

# Load PDF
loader = PyPDFLoader("../3vectorsEmbed/docs/MachineLearning-Lecture01.pdf")
pages = loader.load()
all_page_text=[p.page_content for p in pages]
joined_page_text=" ".join(all_page_text)
# print(joined_page_text)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)
splits = text_splitter.split_text(joined_page_text)

# Retrieve
svm_retriever = SVMRetriever.from_texts(splits,embedding)
tfidf_retriever = TFIDFRetriever.from_texts(splits)

question = "What are major topics for this class?"
docs_svm=svm_retriever.get_relevant_documents(question)
print(docs_svm[0])

question = "what did they say about matlab?"
docs_tfidf=tfidf_retriever.get_relevant_documents(question)
print(docs_tfidf[0])