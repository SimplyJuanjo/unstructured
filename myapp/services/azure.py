from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential

class AzureService:
    def __init__(self, connection_string, container_name, url):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.parts = url.split('/')
    
    def upload_text_to_blob(self, name, text_data):
        blob_name = f"{self.parts[4]}/{self.parts[5]}/{self.parts[6]}/{self.parts[7]}/{self.parts[8]}/{name}.txt"
        blob_client = self.container_client.get_blob_client(blob_name)
        upload_blob_response = blob_client.upload_blob(data=text_data)
        return upload_blob_response

