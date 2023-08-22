from flask import Blueprint, request, jsonify
from myapp.config.config import Config
from myapp.services.azure import AzureBlobService, AzureCognitiveService
from myapp.services.pinecone import PineconeService
from myapp.services.openai import OpenAIService
from myapp.services.webpubsub import WebPubSubClientWrapper
from myapp.utils.translation import Translator
from myapp.utils.text_processing import TextProcessor
from myapp.utils.file_handling import FileHandler
from myapp.utils.request_params import ProcessDataParams
from myapp.utils.node_server_caller import NodeServerCaller
import time

main = Blueprint('main', __name__)

@main.route('/triggerCreateBook', methods=['POST'])
def process_data():
    try:
        start_time = time.time()
        config = Config()
        params = ProcessDataParams.from_request(request)
        azure_blob_service = AzureBlobService(config.AZURE_CONNECTION_STRING, params.container_name, params.url)
        azure_cognitive_service = AzureCognitiveService(config.VECTOR_STORE_ADDRESS, config.VECTOR_STORE_PASSWORD, config.OPENAI_API_KEY, config.OPENAI_API_BASE, config.OPENAI_API_VERSION, config.OPENAI_API_TYPE)
        webpubsub_service = WebPubSubClientWrapper(config.WEBPUBSUB_ENDPOINT, config.WEBPUBSUB_KEY)
        # pinecone_service = PineconeService(config.PINECONE_API_KEY, config.PINECONE_ENVIRONMENT, config.OPENAI_API_KEY)
        translator = Translator(config.MS_TRANSLATOR_SUBSCRIPTION_KEY, config.MS_TRANSLATOR_REGION, config.DEEPL_AUTH_KEY)
        text_processor = TextProcessor()
        node_server_caller = NodeServerCaller(config.NODE_SERVER_URL)

        # Start the processing
        message = {"docId": params.doc_id, "status": "procesando", "filename": params.filename}
        webpubsub_service.send_to_group(params.user_id, message)
        print(f'Procesando index: {params.index_name}, doc_id: {params.doc_id}, doc_url: {params.url}')

        # Download the file
        file_path = FileHandler.download_file(params.url)

        # Extract the file
        ocr_data = text_processor.load_pdf(file_path, strategy="ocr_only", ocr_languages="spa+eng", doc_id=params.doc_id)
        print(f"OCR Data loaded in {time.time() - start_time} seconds")

        # Translate the file
        translate_time = time.time()
        ocr_translation, language = translator.detect_and_translate(ocr_data[0].page_content)
        print(f"OCR translation done in {time.time() - translate_time} seconds")

        # Store the language in Azure Blob Storage
        azure_response = azure_blob_service.upload_text_to_blob("language", language)
        print(f"Language stored in Azure: {azure_response}")

        # Store the file in Azure Blob Storage
        azure_response = azure_blob_service.upload_text_to_blob("extracted", ocr_data[0].page_content)
        print(f"OCR stored in Azure: {azure_response}")
        message["status"] = "extracted done"
        webpubsub_service.send_to_group(params.user_id, message)

        # Store the translation in Azure Blob Storage
        azure_response = azure_blob_service.upload_text_to_blob("extracted_translated", ocr_translation)
        print(f"OCR translation stored in Azure: {azure_response}")
        message["status"] = "extracted_translated done"
        webpubsub_service.send_to_group(params.user_id, message)

        # Process the translated data
        texts, metadatas = text_processor.process_data(data=ocr_data, chunk_size=1400, chunk_overlap=400, completed_translation=ocr_translation)

        # Create an index if it doesn't exist
        azure_vectorstore = azure_cognitive_service.create_index_if_none(params.index_name)

        # Index texts
        try:
            azure_time = time.time()
            azure_cognitive_service.index_texts(texts, metadatas, azure_vectorstore)
            print(f"Texts indexed in Azure Cognitive {azure_vectorstore} in {time.time() - azure_time} seconds")
        except Exception as e:
            message["status"] = "index creation failed"
            webpubsub_service.send_to_group(params.user_id, message)
            raise e
        
        else:
            message["status"] = "index created, ocr"
            webpubsub_service.send_to_group(params.user_id, message)

        # Extract the file fast
        fast_data = text_processor.load_pdf(file_path, strategy="fast", doc_id=params.doc_id)
        print(f"Fast Data loaded in {time.time() - start_time} seconds")

        # Translate the file fast
        translate_time = time.time()
        fast_translation, _ = translator.detect_and_translate(fast_data[0].page_content)
        print(f"Fast translation done in {time.time() - translate_time} seconds")
        
        # Store the file in Azure Blob Storage
        azure_response = azure_blob_service.upload_text_to_blob("fast_extracted", fast_data[0].page_content)
        print(f"Fast stored in Azure: {azure_response}")
        message["status"] = "fast_extracted done"
        webpubsub_service.send_to_group(params.user_id, message)

        # Store the translation in Azure Blob Storage
        azure_response = azure_blob_service.upload_text_to_blob("fast_extracted_translated", fast_translation)
        print(f"Fast translation stored in Azure: {azure_response}")
        message["status"] = "fast_extracted_translated done"
        webpubsub_service.send_to_group(params.user_id, message)

        # Process the translated data
        fast_texts, fast_metadatas = text_processor.process_data(data=fast_data, chunk_size=1000, chunk_overlap=200, completed_translation=fast_translation)

        # Index texts
        try:
            azure_time = time.time()
            azure_cognitive_service.index_texts(fast_texts, fast_metadatas, azure_vectorstore)
            print(f"Fast texts indexed in Azure Cognitive {azure_vectorstore} in {time.time() - azure_time} seconds")
        except Exception as e:
            message["status"] = "index creation failed"
            webpubsub_service.send_to_group(params.user_id, message)
            raise e
        else:
            message["status"] = "index created, let chat"
            webpubsub_service.send_to_group(params.user_id, message)

        # Call the Node server
        node_server_caller.call_analize_doc(params.container_name, params.url_analize_doc, params.doc_id, params.filename, params.index_name, params.user_id)

        # Delete the file
        FileHandler.delete_file(file_path)

        finish_time = time.time()
        print("@@ TOTAL TIME: ", finish_time - start_time)
        return jsonify({"msg": "done", "status": 200})

    except Exception as e:
        print(e)
        message2 = {"docId": params.doc_id, "status": "error processing the file", "filename": params.filename}
        message2["error"] = str(e)
        webpubsub_service.send_to_group(params.user_id, message2)
        raise e