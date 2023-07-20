import os
import requests
import json

class NodeServerCaller:
    def __init__(self, node_server_url):
        self.node_server_url = node_server_url + '/api/analizeDoc'

    def call_analize_doc(self, container_name, url_analize_doc, document_id, filename, patient_id, user_id):
        headers = {'Content-Type': 'application/json'}
        payload = {
            'containerName': container_name,
            'url': url_analize_doc,
            'documentId': document_id,
            'filename': filename,
            'patientId': patient_id,
            'userId': user_id
        }
        requests.post(self.node_server_url, data=json.dumps(payload), headers=headers)
