import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

class PineconeService:
    def __init__(self, api_key, environment, openai_api_key):
        self.api_key = api_key
        self.environment = environment
        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        pinecone.init(api_key=self.api_key, environment=self.environment)
    
    def create_index_if_none(self, index_name, dimension=1536, metric="cosine", pods=1, pod_type="p1.x1"):
        active_indexes = pinecone.list_indexes()
        if not index_name in active_indexes:
            pinecone.create_index(index_name, dimension=dimension, metric=metric, pods=pods, pod_type=pod_type)

    def index_texts(self, texts, metadatas, index_name):
        vectorstore = Pinecone.from_texts([t.page_content for t in texts], self.embeddings, index_name=index_name, metadatas=metadatas)
        return vectorstore
