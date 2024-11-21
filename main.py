import os
import chainlit as cl
from document_processor import EnhancedDocumentProcessor
from bot import DocumentChatbot
from typing import Optional

@cl.on_chat_start
async def start():
    chatbot = DocumentChatbot()
    cl.user_session.set("chatbot", chatbot)
    
    # Enhanced welcome message with more features
    welcome_message = """
    🌟 Welcome to your Document Assistant! 
    
    I can help you with:
    📚 Finding specific documents
    🔍 Answering questions from your document database
    💡 Providing relevant information from your files
    📊 Analyzing document content
    🔄 Processing new documents
    
    What would you like to know?
    """
    
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def main(message: str):
    chatbot = cl.user_session.get("chatbot")
    async with cl.Step("Processing your request..."):
        response = await chatbot.on_message(message)
    await cl.Message(content=response).send()

@cl.on_stop
def stop():
    chatbot = cl.user_session.get("chatbot")
    if chatbot:
        chatbot.cleanup()

if __name__ == "__main__":
    cl.start()
