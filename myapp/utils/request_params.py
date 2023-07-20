from dataclasses import dataclass

@dataclass
class ProcessDataParams:
    index_name: str
    doc_id: str
    url: str
    container_name: str
    user_id: str
    filename: str
    url_analize_doc: str

    @classmethod
    def from_request(cls, request):
        return cls(
            index_name=request.args.get('index'),
            doc_id=request.args.get('doc_id'),
            url=request.args.get('url'),
            container_name=request.args.get('containerName'),
            user_id=request.args.get('userId'),
            filename=request.args.get('filename'),
            url_analize_doc = request.args.get('urlanalizeDoc')
        )
