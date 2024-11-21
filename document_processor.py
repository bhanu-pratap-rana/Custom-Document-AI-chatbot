import os
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Tuple
from functools import partial
from tqdm import tqdm
import numpy as np

import cv2
import pytesseract
import spacy
from sentence_transformers import SentenceTransformer
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from sqlalchemy import create_engine, Table, Column, Integer, String, Text, MetaData
from sqlalchemy.orm import sessionmaker
from dynamic_summarizer import DynamicSummarizer
from tenacity import retry, stop_after_attempt, wait_exponential

BATCH_SIZE = 32
COLLECTION_NAME = "documents"
VECTOR_DIM = 384

class EnhancedDocumentProcessor:
    def __init__(self):
        self.base_dir = Path('C:/Users/HP/Desktop/Project ISL')
        self.dataset_dir = self.base_dir / 'dataset'
        self.output_dir = self.base_dir / 'output_result'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.nlp = spacy.load('en_core_web_sm')
        self.summarizer = DynamicSummarizer()
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        self.init_mysql_schema()
        self.collection = self.init_vector_db()

    def init_mysql_schema(self):
        engine = create_engine('mysql+pymysql://root:root@localhost/documents_db')
        metadata = MetaData()
        
        self.documents_table = Table(
            'documents', metadata,
            Column('id', Integer, primary_key=True),
            Column('filename', String(255)),
            Column('text', Text),
            Column('summary', Text),
            Column('entities', Text)
        )
        
        metadata.create_all(engine)

    def connect_to_milvus(self):
        """Connect to Milvus server"""
        connections.connect(
            alias="default",
            host='localhost',
            port='19530',
            timeout=30
        )

    def init_vector_db(self):
        """Initialize Milvus vector database collection"""
        self.connect_to_milvus()

        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        ]

        schema = CollectionSchema(fields=fields, description="Document embeddings collection")
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        return collection

    @staticmethod
    def preprocess_image(img):
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.fastNlMeansDenoising(binary)

    @staticmethod
    def process_batch_images(batch_paths: List[Path], nlp, summarizer, embedder) -> List[Dict]:
        results = []
        for image_path in batch_paths:
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                
                processed_img = EnhancedDocumentProcessor.preprocess_image(img)
                text = pytesseract.image_to_string(processed_img)
                
                doc = nlp(text)
                cleaned_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                
                summary = summarizer.summarize(cleaned_text)
                embedding = embedder.encode(cleaned_text)
                
                results.append({
                    'filename': str(image_path),
                    'text': text,
                    'cleaned_text': cleaned_text,
                    'entities': entities,
                    'summary': summary,
                    'embedding': embedding.tolist()
                })
                
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")
                continue
                
        return results

    def store_batch_results(self, batch_results: List[Dict]):
        if not batch_results:
            return

        try:
            # MySQL storage
            engine = create_engine('mysql+pymysql://root:root@localhost/documents_db')
            with engine.connect() as connection:
                mysql_data = [{
                    'filename': result['filename'],
                    'text': result['text'],
                    'summary': result['summary'],
                    'entities': json.dumps(result['entities'])
                } for result in batch_results]
                connection.execute(self.documents_table.insert(), mysql_data)
                connection.commit()

            # Milvus storage
            milvus_data = []
            for result in batch_results:
                milvus_data.append({
                    'filename': result['filename'],
                    'embedding': result['embedding']
                })
            
            self.collection.insert(milvus_data)
            self.collection.flush()

        except Exception as e:
            self.logger.error(f"Error in batch storage: {str(e)}")
            raise

    def process_all_documents(self, num_workers: int = 4):
        image_paths = list(self.dataset_dir.glob('**/*.[jp][pn][g]'))
        self.logger.info(f"Found {len(image_paths)} images to process")
        
        batches = [image_paths[i:i + BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)]
        process_count = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            process_func = partial(
                self.process_batch_images,
                nlp=self.nlp,
                summarizer=self.summarizer,
                embedder=self.embedder
            )
            
            for batch_results in tqdm(executor.map(process_func, batches), 
                                    total=len(batches), 
                                    desc="Processing batches"):
                if batch_results:
                    self.store_batch_results(batch_results)
                    process_count += len(batch_results)
        return process_count  

    def cleanup(self):
        try:
            connections.disconnect("default")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    processor = EnhancedDocumentProcessor()
    try:
        processor.process_all_documents()
    finally:
        processor.cleanup()
