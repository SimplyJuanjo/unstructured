from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
import subprocess
import tiktoken

def num_tokens_from_string(string: str,) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4-0613")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def convert_docx_to_pdf(input_path, output_dir='/tmp'):
    subprocess.run([
        "soffice", 
        "--headless",
        "--convert-to", "pdf:writer_pdf_Export",
        "--outdir", output_dir,
        input_path
    ])

class TextProcessor:
    def load_file(self, temp_file_path, strategy="ocr_only", ocr_languages=["spa","eng"], doc_id=None, suffix=None):
        if strategy == "hi_res":
            kwargs = {
                "strategy": strategy,
                "languages": ocr_languages,
                # "chunking_strategy": "by_title",
                # "combine_under_n_chars": 1000,
                # "new_after_n_chars": 2000,

            }
        else:
            kwargs = {
                "strategy": strategy,
            }

        if suffix==".docx":
            #Convert to pdf
            convert_docx_to_pdf(temp_file_path)
            loader = UnstructuredFileLoader(temp_file_path.replace(".docx", ".pdf"), **kwargs)
        else:
            loader = UnstructuredFileLoader(temp_file_path, mode="single", **kwargs)

        data = loader.load()
        # for doc in data:
        #     print(num_tokens_from_string(doc.page_content), )
        #     #print(doc.metadata)
        #     print("----")
        #     if num_tokens_from_string(doc.page_content) < 100:
        #         print(doc.page_content)
            
            #doc.metadata["doc_id"] = doc_id
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
