import os
import openai
import sys
from langchain_community.document_loaders import PyPDFLoader,TextLoader,NotionDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
import re

'''
langchain.text_splitter

CharacterTextSplitter() - Splitting text that looks at characters
MarkdownHeaderTextSplitter() - Split MD files based on specified headers
TokenTextSplitter() - Split text that looks at tokens
SentenceTransformersTokenTextSplitter() - Splitting text that looks at tokens
RecursiveCharacterTextSplitter() - Split text that looks at characters.. recursively tries to split by diff characters to find one that works
-- Recommended for generic text
Language() - for CPP, Python, Ruby, Markdown, etc
NLTKTextSplitter() - Split text that looks at sentences using NLTK
SpactTextSplittter() - Slit text looks at sentences using Spacy
'''

# chunk_size =26
# chunk_overlap = 4

# r_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=chunk_size,
#     chunk_overlap=chunk_overlap
# )
# c_splitter = CharacterTextSplitter(
#     chunk_size=chunk_size,
#     chunk_overlap=chunk_overlap
# )

# text1 = 'abcdefghijklmnopqrstuvwxyz'
# print(r_splitter.split_text(text1))

# text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
# print(r_splitter.split_text(text2))
# ## >>> ['abcdefghijklmnopqrstuvwxyz', 'wxyzabcdefg']

# text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
# print(r_splitter.split_text(text3))
# print(c_splitter.split_text(text3))
# ## >>> ['a b c d e f g h i j k l m', 'l m n o p q r s t u v w x', 'w x y z']
# ## ^^^ Spaces count as "chunk overlap"
# ## >>> ['a b c d e f g h i j k l m n o p q r s t u v w x y z']
# ## ^^^ CharacterTextSplitter automatically splits on \n

# c_splitter = CharacterTextSplitter(
#     chunk_size=chunk_size,
#     chunk_overlap=chunk_overlap,
#     separator = ' '
# )
# print(c_splitter.split_text(text3))
# ## >>> ['a b c d e f g h i j k l m', 'l m n o p q r s t u v w x', 'w x y z']

some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

# print(len(some_text))
## >>> 496

# c_splitter = CharacterTextSplitter(
#     chunk_size=450,
#     chunk_overlap=0,
#     separator = ' '
# )
# r_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=450,
#     chunk_overlap=0, 
#     separators=["\n\n", "\n", " ", ""] ## 1st try split \n\n, 2nd split \n, etc
# )

# print(c_splitter.split_text(some_text))
# ''' 
# >>> ['When writing documents, writers will use document structure to group content. This can 
# convey to the reader, which idea\'s are related. For example, closely related ideas are in 
# sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n Paragraphs are 
# often delimited with a carriage return or two carriage returns. Carriage returns are 
# the "backslash n" you see embedded in this string. Sentences have a period at the end, 
# but also,', 
# 'have a space.and words are separated by space.']'''

# print(r_splitter.split_text(some_text))
# '''
# >>> ["When writing documents, writers will use document structure to group content. This can convey 
# to the reader, which idea's are related. For example, closely related ideas are in sentances. 
# Similar ideas are in paragraphs. Paragraphs form a document.", 
# 'Paragraphs are often delimited with a carriage return or two carriage returns. Carriage returns 
# are the "backslash n" you see embedded in this string. Sentences have a period at the end, but 
# also, have a space.and words are separated by space.']
# '''

## Let's reduce the chunk size a bit and add a period to our separators:
# r_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=150,
#     chunk_overlap=0,
#     separators=["\n\n", "\n", "\. ", " ", ""]
# )
# print(r_splitter.split_text(some_text))
# '''
# >>> ["When writing documents, writers will use document structure to group content. This can convey to the reader, which idea's are related. For example,", 
#  'closely related ideas are in sentances. Similar ideas are in paragraphs. Paragraphs form a document.', 
#  'Paragraphs are often delimited with a carriage return or two carriage returns. Carriage returns are the "backslash n" you see embedded in this', 
#  'string. Sentences have a period at the end, but also, have a space.and words are separated by space.']
# '''

# r_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=150,
#     chunk_overlap=0,
#     separators=["\n\n", "\n", "(?<=\. )", " ", ""]
# )
    #separators=["\n\n", "\n", ". ", " ", ""] # "(?<=\. )", " ", ""])
## Should split by sentences, but isnt for some reason

# sentences = r_splitter.split_text(some_text)
# print(sentences)
'''
>>>["When writing documents, writers will use document structure to group content. This can convey to the reader, which idea's are related. For example,", 
'closely related ideas are in sentances. Similar ideas are in paragraphs. Paragraphs form a document.', 
'Paragraphs are often delimited with a carriage return or two carriage returns. Carriage returns are the "backslash n" you see embedded in this', 
'string. Sentences have a period at the end, but also, have a space.and words are separated by space.']
'''

loader = PyPDFLoader("docs/MachineLearning-Lecture01.pdf")
pages = loader.load()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

docs = text_splitter.split_documents(pages)
# print(len(docs))
# print(len(pages))

loader = NotionDirectoryLoader("docs/Notion_DB")
notion_db = loader.load()

docs = text_splitter.split_documents(notion_db)

print(len(notion_db))