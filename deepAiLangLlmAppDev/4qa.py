import datetime
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from IPython.display import display, Markdown


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
loader = CSVLoader(file_path=file)

print(loader.autodetect_encoding)
# docs = docs.load()

# print(docs[0])

# index = VectorstoreIndexCreator(
#     vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])
#     # .from_loaders([loader])

# query ="Please list all your shirts with sun protection \
# in a table in markdown and summarize each one."

# response = index.query(query)
# display(Markdown(response))