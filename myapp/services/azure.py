from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import AzureSearch
import myapp.utils.azure_utils as azure_utils

class AzureBlobService:
    def __init__(self, connection_string, container_name, url):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.parts = url.split('/')
    
    def upload_text_to_blob(self, name, text_data):
        blob_name = f"{self.parts[4]}/{self.parts[5]}/{self.parts[6]}/{self.parts[7]}/{self.parts[8]}/{name}.txt"
        blob_client = self.container_client.get_blob_client(blob_name)
        upload_blob_response = blob_client.upload_blob(data=text_data)
        return upload_blob_response


class AzureCognitiveService:
    def __init__(self,vector_store_address, vector_store_password, openai_api_key):
        self.vector_store_address = vector_store_address
        self.vector_store_password = vector_store_password
        self.index_client = SearchIndexClient(endpoint=vector_store_address, credential=AzureKeyCredential(vector_store_password))
        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
    
    def create_index_if_none(self, index_name):
        active_indexes = self.index_client.list_indexes()
        if not index_name in active_indexes:
            vector_store: AzureSearch = AzureSearch(
                azure_search_endpoint=self.vector_store_address,
                azure_search_key=self.vector_store_password,
                index_name=index_name,
                embedding_function=self.embeddings.embed_query,
                fields=azure_utils.FIELDS,
            )
            print("Index created", vector_store)
        else:
            print("Index already exists")
            vector_store = AzureSearch(index_name=index_name, search_type="hybrid", azure_search_endpoint=self.vector_store_address, azure_search_key=self.vector_store_password, embedding_function=self.embeddings.embed_query)
        return vector_store

    def index_texts(self, texts, metadatas, vector_store):
        vector_store.add_texts(
            [t.page_content for t in texts],
            metadatas,
        ) 
