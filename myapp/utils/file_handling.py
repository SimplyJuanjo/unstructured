import requests
import tempfile
import os

class FileHandler:
    @staticmethod
    def download_file(url, suffix):
        """Downloads a file and returns the path to the temporary file."""
        response = requests.get(url)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(response.content)
        return str(temp_file.name)

    @staticmethod
    def delete_file(file_path):
        """Deletes a file."""
        os.remove(file_path)
