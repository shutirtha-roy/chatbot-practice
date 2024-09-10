# I have create a vector database which I have uploaded in the google drive
# You can go and download it using your gsuit id.
# link https://drive.google.com/drive/folders/1BtStFttBSCjCkWNqvePwuUxdqIzx-02w?usp=sharing


# or else you can use this method. Try to cleanup the urls_list and remove the broken urls first
# It will make your database size smaller and will give you a smoother run



from dotenv import load_dotenv
load_dotenv()


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from urls_list import urls

import os

os.environ["USER_AGENT"] = os.getenv('USER_AGENT')

splitter = RecursiveCharacterTextSplitter()
embedder = OpenAIEmbeddings(openai_api_key = os.getenv('OPENAI_API_KEY'))

splitted_docs = splitter.split_documents(WebBaseLoader(urls).load())

vector_store = FAISS.from_documents(splitted_docs, embedder)
vector_store.save_local("BOKU_BOT")


