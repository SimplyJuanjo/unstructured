from flask import Flask, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import UnstructuredFileLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pinecone
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.messaging.webpubsubservice import WebPubSubServiceClient
from azure.core.credentials import AzureKeyCredential
import uuid
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import AgentType
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

from deepl_utils import get_deepl_code, translate

import openai
import json
import time
import requests
import tempfile

app = Flask(__name__)


openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
blobname = os.getenv('BLOBNAME')
blobkey = os.getenv('BLOBKEY')

pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_environment,
)

# Initialize the OpenAIEmbeddings object with your credentials

BLOB = {
    "KEY" : blobkey,
    "NAMEBLOB": blobname,
}
connection_string = f'DefaultEndpointsProtocol=https;AccountName={BLOB["NAMEBLOB"]};AccountKey={BLOB["KEY"]};EndpointSuffix=core.windows.net'
# connection_string = f"https://{account_name}core.windows.net/"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

subscription_key = '302b412fdec54f8595b3075c4e610754' # os.environ[key_var_name]
region = 'northeurope' # os.environ[region_var_name]

def call_analize_doc(container_name, url, document_id, filename, patient_id, user_id):
    serverurl = os.getenv('SERVER_URL')
    node_server_url = serverurl+'/api/analizeDoc'  # Replace with your node.js server URL
    headers = {'Content-Type': 'application/json'}
    payload = {
        'containerName': container_name,
        'url': url,
        'documentId': document_id,
        'filename': filename,
        'patientId': patient_id,
        'userId': user_id
    }
    # No need to handle the response
    response = requests.post(node_server_url, data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        print('Request to Node.js server was successful.')
    else:
        print('Failed to make the request, status code: ', response.status_code)

@app.route('/triggerCreateBook', methods=['POST'])
def process_data():
    try:
        start_time = time.time()
        index_name = request.args.get('index')
        doc_id = request.args.get('doc_id')
        url = request.args.get('url')
        container_name = request.args.get('containerName')
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

        # Download the PDF from the URL and save it to a temporary file
        response = requests.get(url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        # Load the PDF file in low resolution
        loader_1 = PDFMinerLoader(temp_file_path)
        data = loader_1.load()
        data[0].metadata["doc_id"] = doc_id

        data_time_1 = time.time()
        print(f"Data 1 loaded in {data_time_1 - start_time} seconds")
        print(data)

        translate_time = time.time()
        # To detect the language use only the first 500 characters
        detect_body = [{
        'text': data[0].page_content[:500],
        }]  
        # If you encounter any issues with the base_url or path, make sure
        # that you are using the latest endpoint: https://docs.microsoft.com/azure/cognitive-services/translator/reference/v3-0-languages
        base_url = 'https://api.cognitive.microsofttranslator.com/'
        path = '/detect?api-version=3.0'
        constructed_url = base_url + path

        headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
        }

        ms_detect_request = requests.post(constructed_url, headers=headers, json=detect_body)
        response = ms_detect_request.json()

        print(json.dumps(response, sort_keys=True, indent=4, ensure_ascii=False, separators=(',', ': ')))
        language = response[0]['language']
        # See if the language is supported by DeepL
        deepl_code = get_deepl_code(language)
        if deepl_code is not None:
            print(f"Language {language} is supported by DeepL with code {deepl_code}")
            # Translate the text to English
            print(len(data[0].page_content))
            # The maximum number of characters that can be translated per request is 100000, so we need to split the text
            # into chunks of 100000 characters
            chunks = [data[0].page_content[i:i+100000] for i in range(0, len(data[0].page_content), 100000)]
            print(len(chunks))

            completed_translation = ""
            for i, chunk in enumerate(chunks):
                translated_text = translate(chunk, "EN-US")
                completed_translation += translated_text
                print(f"Chunk {i} translated")
                print(len(completed_translation))
        else:
            print(f"Language {language} is not supported by DeepL")
            print(response[0]['language'])

            path_2 = '/translate?api-version=3.0'
            params = '&from=' + language + '&to=en'
            constructed_url_2 = base_url + path_2 + params

            print(len(data[0].page_content))
            # The maximum number of characters that can be translated per request is 50000, so we need to split the text
            # into chunks of 50000 characters
            chunks = [data[0].page_content[i:i+50000] for i in range(0, len(data[0].page_content), 50000)]
            print(len(chunks))

            completed_translation = ""
            for i, chunk in enumerate(chunks):
                body = [{
                'text': chunk,
                }]
                trans_response = requests.post(constructed_url_2, headers=headers, json=body)
                translated_text = trans_response.json()[0]['translations'][0]['text']
                completed_translation += translated_text
                print(f"Chunk {i} translated")
                print(len(completed_translation))

            end_translate_time = time.time()
            print(f"Translation done in {end_translate_time - translate_time} seconds")

        # Send the data to Azure Blob Storage
        container_name = url.split("/")[3]
        container_client = blob_service_client.get_container_client(container_name)
        # Split the URL by '/'
        parts = url.split('/')

        # Upload the original PDF text
        fast_result_name = f"{parts[4]}/{parts[5]}/{parts[6]}/{parts[7]}/{parts[8]}/fast_extracted.txt"
        print(fast_result_name)
        fast_block_blob_client = container_client.get_blob_client(fast_result_name)
        fast_upload_blob_response = fast_block_blob_client.upload_blob(data=data[0].page_content)
        print(fast_upload_blob_response)

        message["status"] = "fast_extracted done"
        service_client.send_to_group(
            user_id,
            json.dumps(message),
            content_type="application/json"
        )

        # Upload the translated PDF text
        trans_result_name = f"{parts[4]}/{parts[5]}/{parts[6]}/{parts[7]}/{parts[8]}/fast_extracted_translated.txt"
        print(trans_result_name)
        trans_block_blob_client = container_client.get_blob_client(trans_result_name)
        trans_upload_blob_response = trans_block_blob_client.upload_blob(data=completed_translation)
        print(trans_upload_blob_response)

        message["status"] = "fast_extracted_translated done"
        service_client.send_to_group(
            user_id,
            json.dumps(message),
            content_type="application/json"
        )

        # Upload the PDF language as a text file
        lang_result_name = f"{parts[4]}/{parts[5]}/{parts[6]}/{parts[7]}/{parts[8]}/language.txt"
        print(lang_result_name)
        lang_block_blob_client = container_client.get_blob_client(lang_result_name)
        lang_upload_blob_response = lang_block_blob_client.upload_blob(data=language)
        print(lang_upload_blob_response)

        # Process the PDF data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        data[0].page_content = completed_translation
        texts = text_splitter.split_documents(data)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        metadatas = []
        for text in texts:
            metadatas.append({
                "source": text.metadata["source"],
                "doc_id": text.metadata["doc_id"],
            })

        try:
            pinecone_time = time.time()
            active_indexes = pinecone.list_indexes()
            if not index_name in active_indexes:
                print("Creating index")
                pinecone.create_index(index_name, dimension=1536, metric="cosine", pods=1, pod_type="p1.x1")

            vectorstore = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name, metadatas=metadatas)
            print("Index created", vectorstore)
            print(f"Indexing done in {time.time() - pinecone_time} seconds PINECONE")
        
        except Exception as e:
            message["status"] = "index creation failed"
            service_client.send_to_group(
                user_id,
                json.dumps(message),
                content_type="application/json"
            )
            print(e)
            raise e
            
        else:
            message["status"] = "index created, let chat"
            service_client.send_to_group(
                user_id,
                json.dumps(message),
                content_type="application/json"
            )

        urlanalize_doc = request.args.get('urlanalizeDoc')
        call_analize_doc(container_name, urlanalize_doc, doc_id, filename, index_name, user_id)
        #service_client.send_to_all(message = {'from': user_id,'data': 'index created, let chat'})

        # TODO: match text lang from ms_detect_request to unstructured available languages
        # Extract the text from the PDF in high resolution
        unstructured_kwargs = {
            "strategy": "ocr_only",
            "ocr_languages": "spa",
        }
        loader_2 = UnstructuredFileLoader(temp_file_path, **unstructured_kwargs)
        data_2 = loader_2.load()
        data_2[0].metadata["doc_id"] = doc_id

        data_time = time.time()
        print(f"Data 2 loaded in {data_time - start_time} seconds")
        print(data_2)

        result_name = f"{parts[4]}/{parts[5]}/{parts[6]}/{parts[7]}/{parts[8]}/extracted.txt"
        print(result_name)
        block_blob_client = container_client.get_blob_client(result_name)
        upload_blob_response = block_blob_client.upload_blob(data=data_2[0].page_content)
        print(upload_blob_response)

        if deepl_code is not None:
            print(f"Language {language} is supported by DeepL with code {deepl_code}")
            # Translate the text to English
            print(len(data_2[0].page_content))
            # The maximum number of characters that can be translated per request is 100000, so we need to split the text
            # into chunks of 100000 characters
            chunks = [data_2[0].page_content[i:i+100000] for i in range(0, len(data_2[0].page_content), 100000)]
            print(len(chunks))

            completed_translation = ""
            for i, chunk in enumerate(chunks):
                translated_text = translate(chunk, "EN-US")
                completed_translation += translated_text
                print(f"Chunk {i} translated")
                print(len(completed_translation))
        else:
            print(len(data_2[0].page_content))
            # The maximum number of characters that can be translated per request is 50000, so we need to split the text
            # into chunks of 50000 characters
            chunks_2 = [data_2[0].page_content[i:i+50000] for i in range(0, len(data_2[0].page_content), 50000)]
            print(len(chunks_2))

            completed_translation = ""
            for i, chunk in enumerate(chunks_2):
                body = [{
                'text': chunk,
                }]
                trans_response = requests.post(constructed_url_2, headers=headers, json=body)
                translated_text = trans_response.json()[0]['translations'][0]['text']
                completed_translation += translated_text
                print(f"Chunk {i} translated")
                print(len(completed_translation))

        end_translate_time = time.time()
        print(f"Translation done in {end_translate_time - translate_time} seconds")

        message["status"] = "hi_res extracted done"
        service_client.send_to_group(
            user_id,
            json.dumps(message),
            content_type="application/json"
        )

        # Process the PDF data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        data_2[0].page_content = completed_translation
        texts = text_splitter.split_documents(data_2)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        metadatas = []
        for text in texts:
            metadatas.append({
                "source": text.metadata["source"],
                "doc_id": text.metadata["doc_id"],
            })

        try:
            active_indexes = pinecone.list_indexes()
            if not index_name in active_indexes:
                print("Creating index")
                pinecone.create_index(index_name, dimension=1536, metric="cosine", pods=1, pod_type="p1.x1")

            vectorstore = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name, metadatas=metadatas)
            print("Index created", vectorstore)
            
        except Exception as e:
            message["status"] = "index creation failed"
            service_client.send_to_group(
                user_id,
                json.dumps(message),
                content_type="application/json"
            )
            print(e)
            raise e
            
        else:
            message["status"] = "index improved"
            service_client.send_to_group(
                user_id,
                json.dumps(message),
                content_type="application/json"
            )
            

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
    except Exception as e:
        print(e)
        message2 = {"docId": doc_id, "status": "error processing the file", "filename": filename}
        message2["error"] = str(e)
        service_client.send_to_group(
            user_id,
            json.dumps(message2),
            content_type="application/json"
        )
        raise e

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
