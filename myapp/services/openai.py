import openai

class OpenAIService:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key
