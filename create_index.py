from flask import Flask, request, jsonify
import logging
import json
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PDFMinerLoader, UnstructuredFileLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from pprint import pprint
import requests
import tempfile
import os
import pinecone
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from azure.messaging.webpubsubservice import WebPubSubServiceClient
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__)

pinecone.init(
    api_key="301911fa-2795-4620-9998-aa24ddab5f8e",
    environment="eu-west1-gcp",
)

# Initialize the OpenAIEmbeddings object with your credentials
os.environ["OPENAI_API_KEY"] = "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"

BLOB = {
    "KEY" : 'EoANszi/Uvi4PpbH3VExh9HxGvVJX9JNEcQkW5vk+QveguTA/3VcNQfS3Z1ZzAL720wuYUc8xF6R+AStO48GOA==',
    "NAMEBLOB": 'blobraitogpt2',
}
connection_string = f'DefaultEndpointsProtocol=https;AccountName={BLOB["NAMEBLOB"]};AccountKey={BLOB["KEY"]};EndpointSuffix=core.windows.net'
# connection_string = f"https://{account_name}core.windows.net/"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)   

@app.route('/triggerCreateBook', methods=['POST'])
def process_data():
    start_time = time.time()
    index_name = request.args.get('index')
    doc_id = request.args.get('doc_id')
    url = request.args.get('url')

    endpoint = "https://raito.webpubsub.azure.com"
    key = "OVFdvxYDiWSBbQv03eVvq7P/yu5u5/pl7TYJ0pFs3ac="
     # Configura el cliente de Web PubSub
    hub_name = "Hub"
     # Inicializar el cliente de Web PubSub
    service_client = WebPubSubServiceClient(endpoint=endpoint, credential=AzureKeyCredential(key), hub=hub_name)

    # Enviar notificación de estado
    user_id = request.args.get('userId')
    filename = request.args.get('filename')
    message = {"docId": doc_id, "status": "procesando", "filename": filename}
    service_client.send_to_group(
        user_id,
        json.dumps(message),
        content_type="application/json"
    )

    print(f'index: {index_name}, doc_id: {doc_id}, url: {url}')

    active_indexes = pinecone.list_indexes()
    if not index_name in active_indexes:
        print("Creating index")
        pinecone.create_index(index_name, dimension=1536, metric="cosine", pods=1, pod_type="p1.x1")

    message = {"docId": doc_id, "status": "index created", "filename": filename}
    service_client.send_to_group(
        user_id,
        json.dumps(message),
        content_type="application/json"
    )
    # Download the PDF from the URL and save it to a temporary file
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    # Load the PDF file
    loader_1 = PDFMinerLoader(temp_file_path)
    data = loader_1.load()
    data[0].metadata["doc_id"] = doc_id

    data_time_1 = time.time()
    print(f"Data 1 loaded in {data_time_1 - start_time} seconds")
    print(data)

    # Process the PDF data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(data)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    metadatas = []
    for text in texts:
        metadatas.append({
            "source": text.metadata["source"],
            "doc_id": text.metadata["doc_id"],
        })

    vStore = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name, metadatas=metadatas)

    unstructured_kwargs = {
        "strategy": "hi_res",
        "ocr_languages": "eng+spa",
    }
    loader_2 = UnstructuredFileLoader(temp_file_path, **unstructured_kwargs)
    data_2 = loader_2.load()

    data_time = time.time()
    print(f"Data 2 loaded in {data_time - start_time} seconds")
    print(data_2)

    # Send the data to Azure Blob Storage
    container_name = url.split("/")[3]
    container_client = blob_service_client.get_container_client(container_name)
    # Split the URL by '/'
    parts = url.split('/')
    # Concatenate the relevant parts
    result_name = f"{parts[4]}/{parts[5]}/{parts[6]}/{parts[7]}/{parts[8]}/extracted.txt"
    print(result_name)
    block_blob_client = container_client.get_blob_client(result_name)
    upload_blob_response = block_blob_client.upload_blob(data=data_2[0].page_content)
    print(upload_blob_response)
    message = {"docId": doc_id, "status": "extracted done", "filename": filename}
    service_client.send_to_group(
        user_id,
        json.dumps(message),
        content_type="application/json"
    )

    fast_result_name = f"{parts[4]}/{parts[5]}/{parts[6]}/{parts[7]}/{parts[8]}/fast_extracted.txt"
    print(fast_result_name)
    fast_block_blob_client = container_client.get_blob_client(fast_result_name)
    fast_upload_blob_response = fast_block_blob_client.upload_blob(data=data[0].page_content)
    print(fast_upload_blob_response)
    message = {"docId": doc_id, "status": "fast_extracted done", "filename": filename}
    service_client.send_to_group(
        user_id,
        json.dumps(message),
        content_type="application/json"
    )
    # for d in data:
    #     d.metadata["doc_id"] = doc_id

    # # Process the PDF data option 1
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # texts = text_splitter.split_documents(data)
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # metadatas = []
    # for text in texts:
    #     metadatas.append({
    #         "source": text.metadata["source"],
    #         "doc_id": text.metadata["doc_id"],
    #     })

    # vStore = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name, metadatas=metadatas)

    os.remove(temp_file_path)

    # Convertir el diccionario en una cadena JSON
    response = {
        "msg": "done",
        "status": 200,
    }
    json_response = json.dumps(response)

    finish_time = time.time()
    print("@@ TOTAL TIME: ", finish_time - start_time)

    return json_response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
