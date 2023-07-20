import requests
import uuid

class MS_Translator:
    def __init__(self, subscription_key, region):
        self.subscription_key = subscription_key
        self.region = region
        self.base_url = 'https://api.cognitive.microsofttranslator.com/'
        self.headers = {
            'Ocp-Apim-Subscription-Key': self.subscription_key,
            'Ocp-Apim-Subscription-Region': self.region,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

    def detect_language(self, text):
        path = '/detect?api-version=3.0'
        constructed_url = self.base_url + path
        detect_body = [{'text': text}]
        response = requests.post(constructed_url, headers=self.headers, json=detect_body)
        language = response.json()[0]['language']
        return language
    
    def translate_ms(self, text, language):
        translate_url = self.base_url + '/translate?api-version=3.0&from=' + language + '&to=en'
        chunks = [text[i:i+50000] for i in range(0, len(text), 50000)]
        completed_translation = ""
        for chunk in chunks:
            body = [{'text': chunk}]
            translate_response = requests.post(translate_url, headers=self.headers, json=body)
            translated_text = translate_response.json()[0]['translations'][0]['text']
            completed_translation += translated_text
        return completed_translation
