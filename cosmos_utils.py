import os
import openai
from pymongo import MongoClient

openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key

def setup_cosmos_connection(DB_NAME, COLLECTION_NAME):
    COSMOS_CLUSTER_CONNECTION_STRING = os.environ.get("COSMOS_VCORE_URI")
    cosmosclient = MongoClient(COSMOS_CLUSTER_CONNECTION_STRING)

    db = cosmosclient[DB_NAME]
    collection = cosmosclient[DB_NAME][COLLECTION_NAME]

    # Send a ping to confirm a successful connection
    try:
        cosmosclient.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        print("You are using the database:", db)
        print("You are using the collection:", collection)

    except Exception as e:
        print(e)
    return collection, db

def create_embeddings_with_openai(input):
    ###### public openai model usage and comment code above
    embeddings = openai.Embedding.create(
        model='text-embedding-ada-002', 
        input=input)["data"][0]["embedding"]
    
    # Number of embeddings    
    # print(len(embeddings))

    return embeddings

def prepare_content(text_content):
  embeddings = create_embeddings_with_openai(text_content)
  request = [
    {
    "textContent": text_content, 
    "vectorContent": embeddings}
  ]
  return request

def insert_requests(text_input, collection):
    request = prepare_content(text_input)
    return collection.insert_many(request).inserted_ids

def create_index(collection, db, COLLECTION_NAME):
  
  # delete and recreate the index. This might only be necessary once.
#   collection.drop_indexes()

  embedding_len = 1536
  print(f'creating index with embedding length: {embedding_len}')
  db.command({
    'createIndexes': COLLECTION_NAME,
    'indexes': [
      {
        'name': 'vectorSearchIndex',
        'key': {
          "vectorContent": "cosmosSearch"
        },
        'cosmosSearchOptions': {
          'kind': 'vector-ivf',
          'numLists': 100,
          'similarity': 'COS',
          'dimensions': embedding_len
        }
      }
    ]
  })

  # Cosmos DB Vector Search API Command
def vector_search(vector_query, collection, max_number_of_results=2):
  results = collection.aggregate([
    {
      '$search': {
        "cosmosSearch": {
          "vector": vector_query,
          "path": "vectorContent",
          "k": max_number_of_results
        },
      "returnStoredSource": True
      }
    }
  ])
  return results

# openAI request - ChatGPT 3.5 Turbo Model
def openai_request(prompt, model_engine='gpt-4-0314'):
    completion = openai.ChatCompletion.create(model=model_engine, messages=prompt, temperature=0, max_tokens=500)
    return completion.choices[0].message.content


# define OpenAI Prompt for News Tweet
def create_tweet_prompt(user_question, result_json):
    instructions = f'You are an assistant that answers questions based on sources provided. \
    If the information is not in the provided source, you answer with "I don\'t know". '
    task = f"{user_question}? /n \
    source: {result_json}"
    
    prompt = [{"role": "system", "content": instructions }, 
              {"role": "user", "content": task }]

    return prompt