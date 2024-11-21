from pymilvus import Collection
import numpy as np

class VectorSearcher:
    def __init__(self, collection_name="documents"):
        self.collection = Collection(collection_name)
        self.collection.load()
    
    def search(self, query_embedding, top_k=5):
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k
        )
        return results
