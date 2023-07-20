from myapp.services.deepl import DeepLService
from myapp.services.ms_translator import MS_Translator

class Translator:
    def __init__(self, subscription_key, region, auth_key):
        self.ms_translator = MS_Translator(subscription_key=subscription_key, region=region)
        self.deepl_service = DeepLService(auth_key=auth_key)

    def detect_and_translate(self, text):
        # detect language with MS Translator
        ms_code = self.ms_translator.detect_language(text)
        # get the corresponding DeepL code
        deepl_code = self.deepl_service.get_deepl_code(ms_code)

        # translate with DeepL if possible, otherwise with MS Translator
        if deepl_code is not None:
            return self.deepl_service.translate_text(text), ms_code 
        else:
            return self.ms_translator.translate_ms(text, ms_code), ms_code