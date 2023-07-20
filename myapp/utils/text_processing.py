from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFMinerLoader, UnstructuredFileLoader

class TextProcessor:
    def __init__(self, openai):
        self.openai = openai

    def load_pdf(self, temp_file_path, strategy="ocr_only", ocr_languages="spa+eng", doc_id=None):
        if strategy == "ocr_only":
            kwargs = {
                "strategy": strategy,
                "ocr_languages": ocr_languages,
            }
        else:
            kwargs = {
                "strategy": strategy,
            }
        
        loader = UnstructuredFileLoader(temp_file_path, **kwargs)
        data = loader.load()
        data[0].metadata["doc_id"] = doc_id
        return data

    def process_data(self, data, chunk_size, chunk_overlap, completed_translation):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        data[0].page_content = completed_translation
        texts = text_splitter.split_documents(data)

        metadatas = []
        for text in texts:
            metadatas.append({
                "source": text.metadata["source"],
                "doc_id": text.metadata["doc_id"],
            })
        return texts, metadatas
