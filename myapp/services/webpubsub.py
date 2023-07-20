from azure.messaging.webpubsubservice import WebPubSubServiceClient
from azure.core.credentials import AzureKeyCredential
import json

class WebPubSubClientWrapper:
    def __init__(self, endpoint, key):
        # Configura el cliente de Web PubSub
        self.hub_name = "Hub"
        self.client = WebPubSubServiceClient(endpoint=endpoint, credential=AzureKeyCredential(key), hub=self.hub_name)

    def send_to_group(self, user_id, message):
        """Send a message to a group of users."""
        self.client.send_to_group(user_id, json.dumps(message), content_type="application/json")
