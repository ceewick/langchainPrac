import os
import openai
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
# import ffmpeg
# import ffprobe

# loader = PyPDFLoader("docs/MachineLearning-Lecture01.pdf")
# pages = loader.load()

# page = pages[0]

# print(page.page_content[0:500])
# print(page.metadata)


###  YouTube
'''
# ! pip install yt_dlp
# ! pip install pydub
# ! pip install --upgrade --quiet  librosa

from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import (
    OpenAIWhisperParser,
    OpenAIWhisperParserLocal,
)

Saved ffmpeg to User Env Path Variable
'''

url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir="docs/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()


# print(len(docs))
print(docs[0].page_content[0:500])