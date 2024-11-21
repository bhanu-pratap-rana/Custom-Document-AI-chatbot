from groq import Groq
import chainlit as cl
from typing import Dict, List
import asyncio
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer

class DocumentChatbot:
    def __init__(self):
        self.groq_client = Groq(api_key="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")
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.conversation_history = []
        self.connect_to_milvus()
    
    def connect_to_milvus(self):
        connections.connect(alias="default", host="localhost", port="19530")
        self.collection = Collection("documents")
        self.collection.load()

    async def search_documents(self, query: str, limit: int = 5):
        query_embedding = self.embedder.encode(query)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["filename"]
        )
        return results

    async def generate_response(self, query: str) -> str:
        search_results = await self.search_documents(query)
        relevant_docs = []
        
        for hit in search_results[0]:
            try:
                filename = hit.entity.get('filename')
                relevant_docs.append(filename)
            except (AttributeError, KeyError):
                if hasattr(hit, 'id'):
                    relevant_docs.append(f"Document_{hit.id}")
        
        context = f"Found relevant documents: {', '.join(relevant_docs) if relevant_docs else 'No specific documents'}"
        
        prompt = f"""Based on the following context and user query, provide a helpful response:
        Context: {context}
        User Query: {query}
        Please provide a detailed response incorporating the relevant document information."""

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant with access to document database and IT support expertise. Provide helpful, natural responses."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"I found these relevant documents: {', '.join(relevant_docs)}. How can I help you understand them better?"

    async def on_message(self, message: cl.Message) -> str:
        message_content = message.content
        response = await self.generate_response(message_content)
        self.conversation_history.append({"role": "user", "content": message_content})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def cleanup(self):
        connections.disconnect("default")
