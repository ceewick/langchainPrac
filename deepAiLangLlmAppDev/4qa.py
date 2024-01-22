import datetime
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from IPython.display import Markdown

# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"


file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file,encoding='utf-8')

# # print(loader.autodetect_encoding)
# # docs = docs.load()
# # print(docs[0])

# ## from langchain.indexes import VectorstoreIndexCreator

# # print(loader)

# index = VectorstoreIndexCreator(
#     vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])
#     # .from_loaders([loader])

# query ="Please list all your shirts with sun protection \
# in a table in markdown and summarize each one."

# llm_replacement_model = OpenAI(temperature=0, 
#                                model='gpt-3.5-turbo-instruct')

# response = index.query(query, 
#                        llm = llm_replacement_model)

# response = index.query(query)
# # print(display(Markdown(response)))

# # Markdown(response)
# # print(Markdown(response))

docs = loader.load()
# print(docs[0])

embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query("Hi my name is Harrison")
# print(embed[:5])

db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

query = "Please suggest a shirt with sunblocking"