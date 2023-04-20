from flask import Flask, request, jsonify
import logging
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PDFMinerLoader, UnstructuredFileLoader
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

app = Flask(__name__)

pinecone.init(
    api_key="ffd7561b-2d58-4a54-8e90-967898669b5a",
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
    index_name = request.args.get('index')
    doc_id = request.args.get('doc_id')
    url = request.args.get('url')

    print(f'index: {index_name}, doc_id: {doc_id}, url: {url}')

    active_indexes = pinecone.list_indexes()
    if not index_name in active_indexes:
        print("Creating index")
        pinecone.create_index(index_name, dimension=1536, metric="cosine", pods=1, pod_type="p1.x1")

    # Download the PDF from the URL and save it to a temporary file
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    # Load the PDF file
    # loader_1 = PDFMinerLoader(temp_file_path)
    # data = loader_1.load()
    loader_2 = UnstructuredFileLoader(temp_file_path)
    data = loader_2.load()
    print(data)

    # Send the data to Azure Blob Storage
    container_name = url.split("/")[3]
    container_client = blob_service_client.get_container_client(container_name)
    # Split the URL by '/'
    parts = url.split('/')
    # Concatenate the relevant parts
    result_name = f"{parts[4]}/{parts[5]}/{parts[6]}/{parts[7]}/{parts[8]}/extracted.txt"
    print(result_name)
    block_blob_client = container_client.get_blob_client(result_name)
    upload_blob_response = block_blob_client.upload_blob(data=data[0].page_content)
    print(upload_blob_response)

    for d in data:
        d.metadata["doc_id"] = doc_id

    # Process the PDF data option 1
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(data)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    metadatas = []
    for text in texts:
        metadatas.append({
            "source": text.metadata["source"],
            "doc_id": text.metadata["doc_id"],
            "option": "1"
        })

    vStore = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name, metadatas=metadatas)

    # Process the PDF data option 2
    text_splitter_2 = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    texts_2 = text_splitter_2.split_documents(data)
    
    metadatas_2 = []
    for text in texts_2:
        metadatas_2.append({
            "source": text.metadata["source"],
            "doc_id": text.metadata["doc_id"],
            "option": "2"
        })

    vStore_2 = Pinecone.from_texts([t.page_content for t in texts_2], embeddings, index_name=index_name, metadatas=metadatas_2)

    # Get the index
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

    # Clean up the temporary file
    os.remove(temp_file_path)

    # Create a preview of the index
    # For this we are going to ask the index (some_medical_report.pdf) for the following questions:

    # Create the model
    retriever = VectorStoreRetriever(vectorstore=vectorstore, search_kwargs={"filter":{"doc_id": doc_id}, "include_metadata": True})
    # docs = retriever.get_relevant_documents("patient's name?")
    model = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="stuff", retriever=retriever)

    # Define the facts and create an empty dictionary for answers
    facts = ["name", "age", "gender", "symptoms", "medications", "allergies", "diagnoses", "treatments"]
    answers = {fact: [] for fact in facts}

    # Map each question to its corresponding fact
    question_to_fact = [
        ("What is the patient's name?", "name"),
        ("How old is the patient?", "age"),
        ("What is the patient's date of birth?", "age"),
        ("What is the gender of the patient?", "gender"),
        ("What are the patient's symptoms?", "symptoms"),
        ("Which symptoms has the patient been experiencing?", "symptoms"),
        ("What are the patient's medications?", "medications"),
        ("What medications has the patient been taking?", "medications"),
        ("What are the patient's allergies?", "allergies"),
        ("What is the main diagnosis in the report?", "diagnoses"),
        ("What is the history of the patient's medical diagnoses?", "diagnoses"),
        ("Which are the patient's diagnosis?", "diagnoses"),
        ("Are there any comorbidities or underlying conditions the patient has?", "diagnoses"),
        ("What non-pharmacological treatments has the patient?", "treatments"),
        ("What is the patient's treatment plan?", "treatments"),
        ("Has the patient received any alternative or complementary therapies?", "treatments"),
    ]

    # Iterate over the questions and their corresponding facts
    for question, fact in question_to_fact:
        answer = model.run(question)
        answers[fact].append(answer)
        print(f"{question}: {answer}")

    # Now ask the model to create a table with the answers
    
    prompt = '''\n This in an example of valid formatted response: {"name":"Pedro gomez","age":"Not provided","gender":"male","symptoms":[{"name":"Seizures", "checked":true},{"name":"Insomnia", "checked":true}],"medications": [{"name":"Sertralina 75mg in breakfast", "checked":true},{"name":"Olanzapina 2.5mg every 12 hours", "checked":true},{"name":"Trazodona 100mg at dinner", "checked":true}],"allergies": [{"name":"Animal fur allergy", "checked":true}],"diagnoses": [{"name":"Hypothyroidism", "checked":true}, {"name":"Scoliosis", "checked":true}], "treatments": [{"name":"Not provided", "checked":true}]
    \n Other example of valid formatted response: {"name":"Pedro gomez","age":"Not provided","gender":"male","symptoms":[{"name":"Deterioration in functional ability", "checked":true},{"name":"Not provided", "checked":true}],"medications": [{"name":"Sertralina 75mg in breakfast", "checked":true},{"name":"paracetamol 2.5mg every 7 hours", "checked":true},{"name":"Hizentra (GGSC) 5g/week", "checked":true}],"allergies": [{"name":"nuts allergy", "checked":true}],"diagnoses": [{"name":"inmunodeficiency associated with X", "checked":true}], "treatments": [{"name":"avoid certain foods", "checked":true},{"name":"recommended logopedia", "checked":true},{"name":"have a control imaging test", "checked":true}]}
    \n Example of invalid unformatted response: {"name": "Pedro gomez","age": "Not provided","gender": "male","symptoms": [{name:"Deterioration in functional ability", checked:true}\n,{name:"Insomnia", checked:true}],"medications": [{name:"Sertralina 75mg in breakfast", checked:true},{name:"Olanzapina 2.5mg every 12 hours", checked:true},{name:"Trazodona 100mg at dinner", checked:true}],"allergies": [],"diagnoses": [{name:"DFT variant conductual", checked:true}]}, "treatments": [{"name":rehabilitation, "checked":true}].'''
    
    prompt += 'Please create the JSON result with the info from following questions (If you don\'t know which items go to some field, please write "Not provided" in the JSON response):\n'
    for fact in facts:
        answer_text = ""
        for answer in answers[fact]:
            answer_text += f"{answer}; "
        prompt += f"{fact} of the patient: {answer_text}\t"
    
    # Run the model with a clean llm
    chat = ChatOpenAI(openai_api_key=openai_api_key, max_tokens=2000)
    messages = [
        SystemMessage(content='''You are an outstanding doctor and you are reading a medical report, you are extracting the information and creating a JSON response with it.
        The format of the JSON response is the following:
        {"name": "some name","age": "some age","gender": "some gender","symptoms": [{"name": "some symptom", "checked": true}, {"name": "some other symptom", "checked": true}],"medications": [{"name": "some medication", "checked": true}, {"name": "some other medication", "checked": true}],"allergies": [{"name": "some allergy", "checked": true}, {"name": "some other allergy", "checked": true}],"diagnoses": [{"name": "some diagnosis", "checked": true}, {"name": "some other diagnosis", "checked": true}],"treatments": [{"name": "some non-pharmacological treatment", "checked": true}, {"name": "some other non-pharmacological treatment", "checked": true}]}
        PLEASE NOTE: the JSON response must be a valid JSON, if you don't know how to create a valid JSON, please check the examples below. If you don't know which items go to a field, please write "Not provided" in the JSON response. 
        '''),
        HumanMessage(content=prompt),
    ]
    table = chat(messages)
    print(table)

    # Convertir el diccionario en una cadena JSON
    response = {
        "msg": "done",
        "status": 200,
        "table": table.content,
    }
    json_response = json.dumps(response)

    return json_response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
