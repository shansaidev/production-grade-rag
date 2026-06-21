import asyncio
import asyncpg
from qdrant_client import AsyncQdrantClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

async def test_all_services():
        # PostgreSQL
        print("1. Connecting to Postgres...")
        conn = await asyncpg.connect("postgresql://raguser:ragpassword@localhost/ragdb")
        print("2. Connected")
        result = await conn.fetchval("SELECT count(*) from documents")
        print(f"3. Count={result}")
        assert result == 0, "documents table should exist and be empty"
        await conn.close()
        print("4. Connecting to Qdrant...")
        # Qdrant
        qdrant_client = AsyncQdrantClient(host="localhost", port=6333)
        await qdrant_client.create_collection(collection_name="rag_chunks",vectors_config=VectorParams(
            size=384,distance=Distance.COSINE))

        info = await qdrant_client.get_collection("rag_chunks")
        print("6. Collection retrieved")
        assert info.status.value == "green"
        print("7. Qdrant collection is healthy")

        print("✅ All infrastructure checks passed")


asyncio.run(test_all_services())        



