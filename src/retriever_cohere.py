import json
import logging
import re
import boto3
import chromadb

from utils import llm, rerank_prompt_template  

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Retriever for VectorDB
class Retriever:
    """
    Retrieves the top-k most similar documents from ChromaDB for a given query and then uses an LLM to re-rank them.
    """
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "feed_metadata"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)
        
        # Initialize AWS Bedrock client for Titan embed text model
        self.bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.model_id = 'amazon.titan-embed-text-v2:0'

    def generate_embedding(self, text: str):
        """
        Generate a vector embedding for the given text using Amazon Titan Text Embeddings V2.
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
            embedding = result['embeddingsByType']['float']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise e
        return embedding

    def retrieve_top_k(self, query: str, k: int = 3):
        """
        Retrieves the top-k documents from ChromaDB using cosine similarity.
        """
        query_embedding = self.generate_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        # Flatten nested lists if necessary.
        if isinstance(results.get("ids"), list) and results["ids"] and isinstance(results["ids"][0], list):
            results["ids"] = results["ids"][0]
        if isinstance(results.get("metadatas"), list) and results["metadatas"] and isinstance(results["metadatas"][0], list):
            results["metadatas"] = results["metadatas"][0]
        return results

    def retrieve_and_rerank(self, query: str, k: int = 3) -> dict:
        """
        Retrieves the top-k documents, builds a prompt with their summaries, and uses the LLM to decide the best matching table.
        """
        # Retrieve top-k documents.
        results = self.retrieve_top_k(query, k=k)

        candidate_summaries = []
        # Process each metadata entry (which is now a flat dictionary).
        for i, meta in enumerate(results["metadatas"]):
            json_str = meta.get("json_summary", "").strip()
            if json_str:
                try:
                    table_json = json.loads(json_str)
                    table_name = table_json.get("name", f"Candidate_{i+1}")
                    json_pretty = json.dumps(table_json, indent=2)
                    snippet = f"Candidate #{i+1} - Table Name: {table_name}\n{json_pretty}"
                except Exception as e:
                    logger.error(f"Failed to parse JSON in rerank prompt for candidate #{i+1}: {e}. Using enriched_text instead.")
                    snippet = f"Candidate #{i+1} - {meta.get('enriched_text', '')}"
            else:
                snippet = f"Candidate #{i+1} - {meta.get('enriched_text', '')}"
            candidate_summaries.append(snippet)

        schemas_text = "\n".join(candidate_summaries)
        best_table_name = self.rerank(query, schemas_text)

        return {
            "best_table_name": best_table_name,
            "top_k_docs": results
        }

    def rerank(self, query: str, schemas_text: str) -> str:
        """
        Uses the LLM to decide which table best fits the query.
        """
        prompt = rerank_prompt_template.format(q=query, s=schemas_text)
        messages = [
            {"role": "user", "content": prompt}
        ]
        try:
            response = llm.invoke(str(messages))
            response_text = response.content.strip()
            table_name = self._parse_table_name(response_text)
            return table_name
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return "UnknownTable"

    def _parse_table_name(self, text: str) -> str:
        """
        Parses and returns a table name from the LLM response.
        """
        match = re.search(r":\s*([\w\.]+)", text)
        if match:
            return match.group(1).strip()
        else:
            return text.strip()
