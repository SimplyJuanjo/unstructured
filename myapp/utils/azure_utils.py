from langchain.embeddings.openai import OpenAIEmbeddings
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    ScoringProfile,
    TextWeights,
)

model: str = "text-embedding-ada-002"

embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model=model, chunk_size=1)
embedding_function = embeddings.embed_query

FIELDS = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name="content",
        type=SearchFieldDataType.String,
        searchable=True,
        retrievable=True,
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1536,
        vector_search_configuration="default",
    ),
    SearchableField(
        name="metadata",
        type=SearchFieldDataType.String,
        searchable=True,
        retrievable=True,
        filterable=True,
    ),
    # Additional field to store the title
    SearchableField(
        name="source",
        type=SearchFieldDataType.String,
        searchable=True,
        retrievable=True,
        filterable=True,
    ),
    # Additional field for filtering on document source
    SearchableField(
        name="doc_id",
        type=SearchFieldDataType.String,
        searchable=True,
        retrievable=True,
        filterable=True,
    ),
]