import json
import logging
import uuid
import boto3
import chromadb
from chromadb.config import Settings

from utils import llm, prompt_template 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ChromaDB Manager
class ChromaDBManager:
    """
    Manages a local ChromaDB instance (persistent) for storing documents along with metadata.
    """
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "feed_metadata"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(collection_name)

    def generate_embedding(self, enriched_text: str):
        """
        This method is patched by the Enricher so that embeddings can be generated.
        """
        raise NotImplementedError("Embedding generation function not provided.")

    def store_document(self, enriched_text: str, json_summary: dict):
        # Use the table name as the document id. Fallback to a random UUID if not provided.
        table_name = json_summary.get("name")
        if not table_name:
            table_name = str(uuid.uuid4())
        embedding = self.generate_embedding(enriched_text)

        metadata = {
            "enriched_text": enriched_text,
            "json_summary": json.dumps(json_summary)
        }
        # Add the document to ChromaDB using table_name as the id.
        self.collection.add(
            ids=[table_name],
            embeddings=[embedding],  # embedding is now a list of floats
            metadatas=[metadata]
        )
        logger.info(f"Stored document with table name '{table_name}' in ChromaDB.")


# Enricher Module
class Enricher:
    """
    Enriches a table's metadata using an LLM, generates embeddings via Amazon Titan Text Embeddings V2,
    and stores both the enriched text and the original JSON summary in ChromaDB.
    """
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "feed_metadata"):
        # Initialize AWS Bedrock client for Titan embed text model
        self.bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.model_id = 'amazon.titan-embed-text-v2:0'
        
        self.db_manager = ChromaDBManager(db_path=db_path, collection_name=collection_name)
        # Patch the generate_embedding method
        self.db_manager.generate_embedding = self.generate_embedding

    def generate_embedding(self, text: str):
        """
        Generate an embedding for the given text using Amazon Titan Text Embeddings V2.
        Payload format is based on AWS documentation.
        """
        payload = json.dumps({
            "inputText": text,
            "embeddingTypes": ["float"]
        })
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                contentType='application/json',
                accept='application/json',
                body=payload
            )
            result = json.loads(response.get('body').read())
            # Retrieve the float embedding from the result.
            embedding = result['embeddingsByType']['float']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise e
        return embedding

    def enrich_table_metadata(self, json_summary: dict) -> str:
        """
        Generate an enriched summary for the table by sending its full JSON summary to the LLM.
        """
        full_json_str = json.dumps(json_summary, indent=2)
        prompt_str = prompt_template.replace("{json_summary}", full_json_str)
        messages = [
            {"role": "user", "content": prompt_str}
        ]
        try:
            response = llm.invoke(str(messages))
            enriched_text = response.content.strip()
        except Exception as e:
            logger.error(f"Error enriching table '{json_summary.get('name', '')}': {e}")
            enriched_text = f"LLM call failed; no enriched summary generated. Error details: {str(e)}"
        return enriched_text

    def process_and_store_table(self, json_summary: dict) -> dict:
        """
        Processes one table's metadata: enriches it, generates its embedding, and stores it in ChromaDB.
        """
        enriched_text = self.enrich_table_metadata(json_summary)
        self.db_manager.store_document(enriched_text, json_summary)
        return {"enriched_text": enriched_text, "json_summary": json_summary}
