import os

class Config(object):
    OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE')
    OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
    OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    BLOBNAME = os.getenv('BLOBNAME')
    BLOBKEY = os.getenv('BLOBKEY')
    BLOB = {
    "KEY" : BLOBKEY,
    "NAMEBLOB": BLOBNAME,
    }
    AZURE_CONNECTION_STRING = f'DefaultEndpointsProtocol=https;AccountName={BLOB["NAMEBLOB"]};AccountKey={BLOB["KEY"]};EndpointSuffix=core.windows.net'

    MS_TRANSLATOR_SUBSCRIPTION_KEY = os.getenv('TRANSLATOR_SUBSCRIPTION_KEY')
    MS_TRANSLATOR_REGION = 'northeurope'

    DEEPL_AUTH_KEY = os.getenv('DEEPL_API_KEY')

    WEBPUBSUB_ENDPOINT = os.getenv('WEBPUBSUB_ENDPOINT')
    WEBPUBSUB_KEY = os.getenv('WEBPUBSUB_KEY')

    NODE_SERVER_URL = os.getenv('SERVER_URL')

    VECTOR_STORE_ADDRESS = os.getenv('VECTOR_STORE_ADDRESS')
    VECTOR_STORE_PASSWORD = os.getenv('VECTOR_STORE_PASSWORD')