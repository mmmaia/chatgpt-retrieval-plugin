import pytest
import asyncio
from datastore import PgVectorDataStore
from models.models import Document, Query
from services.chunks import get_document_chunks
from services.openai import get_embeddings


# You can replace this with your actual PostgreSQL connection string
DB_URL = "postgresql+asyncpg://username:password@localhost:5432/test_db"

@pytest.fixture
async def datastore():
    ds = PgVectorDataStore(DB_URL)
    yield ds

    # Clean up the database after each test
    await ds.delete(delete_all=True)

@pytest.mark.asyncio
async def test_upsert(datastore):
    documents = [
        Document(id="doc1", text="This is a sample document."),
        Document(id="doc2", text="This is another sample document."),
    ]

    document_ids = await datastore.upsert(documents)
    assert set(document_ids) == {"doc1", "doc2"}

@pytest.mark.asyncio
async def test_query(datastore):
    documents = [
        Document(id="doc1", text="This is a sample document."),
        Document(id="doc2", text="This is another sample document."),
    ]

    await datastore.upsert(documents)

    queries = [Query(query="sample", top_k=2)]

    query_results = await datastore.query(queries)
    assert len(query_results) == 1
    assert query_results[0].query == "sample"
    assert len(query_results[0].results) == 2

    doc_ids = {result.document_id for result in query_results[0].results}
    assert doc_ids == {"doc1", "doc2"}

@pytest.mark.asyncio
async def test_delete(datastore):
    documents = [
        Document(id="doc1", text="This is a sample document."),
        Document(id="doc2", text="This is another sample document."),
    ]

    await datastore.upsert(documents)

    await datastore.delete(ids=["doc1"])

    queries = [Query(query="sample", top_k=2)]

    query_results = await datastore.query(queries)
    assert len(query_results) == 1
    assert query_results[0].query == "sample"
    assert len(query_results[0].results) == 1

    doc_ids = {result.document_id for result in query_results[0].results}
    assert doc_ids == {"doc2"}

import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@pytest.mark.asyncio
async def test_cosine_similarity_ordering(datastore):
    documents = [
        Document(id="doc1", text="This is a sample document."),
        Document(id="doc2", text="This is another sample document."),
        Document(id="doc3", text="This is a third sample document."),
    ]

    await datastore.upsert(documents)

    queries = [Query(query="sample document", top_k=3)]

    query_results = await datastore.query(queries)
    assert len(query_results) == 1
    assert query_results[0].query == "sample document"
    assert len(query_results[0].results) == 3

    query_embedding = get_embeddings(["sample document"])[0]
    similarities = [
        cosine_similarity(query_embedding, get_embeddings([result.text])[0])
        for result in query_results[0].results
    ]

    # Check if the results are ordered by cosine similarity
    assert similarities == sorted(similarities, reverse=True)
