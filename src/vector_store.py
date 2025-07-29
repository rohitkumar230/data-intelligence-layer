import json
import logging
import uuid
from sentence_transformers import SentenceTransformer
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
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )
        logger.info(f"Stored document with table name '{table_name}' in ChromaDB.")


# Enricher Module
class Enricher:
    """
    Enriches a table's metadata using an LLM, generates embeddings, and stores both
    the enriched text and the original JSON summary in ChromaDB.
    """
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "feed_metadata"):

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.db_manager = ChromaDBManager(db_path=db_path, collection_name=collection_name)
        self.db_manager.generate_embedding = self.generate_embedding

    def generate_embedding(self, text: str):
        """
        Generate an embedding for the given text.
        """
        return self.embedding_model.encode(text, convert_to_numpy=True)

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
