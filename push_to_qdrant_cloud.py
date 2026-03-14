"""Upload local Qdrant collections to Qdrant Cloud.

Usage:
    set QDRANT_URL=https://xxx.aws.cloud.qdrant.io:6333
    set QDRANT_API_KEY=your-api-key
    python push_to_qdrant_cloud.py
"""
import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

LOCAL_PATH = os.path.join(os.path.dirname(__file__), "qdrant_data")
CLOUD_URL = os.environ.get("QDRANT_URL", "")
CLOUD_KEY = os.environ.get("QDRANT_API_KEY", "")

if not CLOUD_URL or not CLOUD_KEY:
    print("Set QDRANT_URL and QDRANT_API_KEY environment variables first.")
    sys.exit(1)

local = QdrantClient(path=LOCAL_PATH)
cloud = QdrantClient(url=CLOUD_URL, api_key=CLOUD_KEY)

collections = [c.name for c in local.get_collections().collections]
print(f"Found {len(collections)} local collections: {collections}")

for name in collections:
    info = local.get_collection(name)
    print(f"\n--- {name}: {info.points_count} points, dim={info.config.params.vectors.size} ---")

    # Create collection on cloud (delete first if exists)
    if cloud.collection_exists(name):
        cloud.delete_collection(name)
    cloud.create_collection(
        collection_name=name,
        vectors_config=info.config.params.vectors,
    )

    # Upload in batches
    BATCH = 100
    offset = None
    total = 0
    while True:
        result, next_offset = local.scroll(
            collection_name=name,
            limit=BATCH,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        if not result:
            break
        points = [
            PointStruct(id=r.id, vector=r.vector, payload=r.payload)
            for r in result
        ]
        cloud.upsert(
            collection_name=name,
            points=points,
        )
        total += len(result)
        print(f"  Uploaded {total}/{info.points_count}")
        offset = next_offset
        if offset is None:
            break

    cloud_info = cloud.get_collection(name)
    print(f"  Cloud count: {cloud_info.points_count}")

local.close()
print("\nDone! All collections uploaded to Qdrant Cloud.")
