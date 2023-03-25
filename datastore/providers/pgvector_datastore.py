import os
from typing import Dict, List, Optional
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models.models import (
    Document,
    DocumentChunk,
    DocumentMetadataFilter,
    Query,
    QueryResult,
    QueryWithEmbedding,
)
from services.chunks import get_document_chunks
from services.openai import get_embeddings
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Index, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

# Read the table name from the PGVECTOR_COLLECTION environment variable
table_name = os.getenv("PGVECTOR_COLLECTION", "vector_documents")

class VectorDocument(Base):
    __tablename__ = table_name

    id = Column(String, primary_key=True)
    document_id = Column(String)
    text = Column(String)
    embedding = Column(Vector)

    # Add a Cosine Distance index for faster querying
    index = Index(
        "vector_cosine_idx",
        embedding,
        postgresql_using="ivfflat",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


class PgVectorDataStore(DataStore):
    def __init__(self):
        # Read the database URL from the PGVECTOR_URL environment variable
        db_url = os.getenv("PGVECTOR_URL")
        if not db_url:
            raise ValueError("PGVECTOR_URL environment variable is not set")

        self.engine = create_engine(db_url, future=True)
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)
    

    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        async with self.session_factory() as session:
            for document_id, document_chunks in chunks.items():
                for chunk in document_chunks:
                    vector_doc = VectorDocument(
                        id=chunk.id,
                        document_id=document_id,
                        text=chunk.text,
                        embedding=chunk.embedding,
                    )
                    session.merge(vector_doc)

            await session.commit()

        return list(chunks.keys())

    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        async with self.session_factory() as session:
            results = []
            for query in queries:
                query_embedding = query.embedding
                stmt = (
                    select(VectorDocument)
                    .order_by(VectorDocument.embedding.cosine_distance(query_embedding))
                    .limit(query.top_k)
                )
                matched_documents = await session.execute(stmt)
                matched_documents = matched_documents.scalars().all()

                # Calculate cosine similarity from cosine distance
                query_results = [
                    {
                        "document_id": doc.document_id,
                        "text": doc.text,
                        "score": 1 - doc.embedding.cosine_distance(query_embedding),
                    }
                    for doc in matched_documents
                ]

                results.append(QueryResult(query=query.query, results=query_results))

            return results

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[DocumentMetadataFilter] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        async with self.session_factory() as session:
            stmt = select(VectorDocument)

            if ids:
                stmt = stmt.where(VectorDocument.id.in_(ids))

            if filter and filter.document_id:
                stmt = stmt.where(VectorDocument.document_id == filter.document_id)

            if delete_all:
                stmt = stmt.where(True)

            result = await session.execute(stmt)
            vector_documents = result.scalars().all()

            for vector_document in vector_documents:
                session.delete(vector_document)

            await session.commit()

        return True
