from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import json
import logging
import os
import difflib
from typing import List, Optional

# Import the necessary classes
from databricks_fetcher import DatabricksDataFetcher
from s3_fetcher import S3DataFetcher
from summarizer import Summarizer
from vector_store_cohere import Enricher
from retriever_cohere import Retriever

app = FastAPI(title="Data Access Layer API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper to parse an S3 path of the form "s3://bucket-name/path/to/folder"
def parse_s3_path(s3_path: str):
    if not s3_path.startswith("s3://"):
        raise ValueError("S3 path must start with 's3://'")
    path_without_scheme = s3_path[len("s3://"):]
    parts = path_without_scheme.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix

# Unified Data Fetcher Endpoints
@app.get("/tables")
async def list_tables(
    source: str = Query(..., description="Data source: 'databricks' or 's3'"),
    catalog_name: Optional[str] = None,
    schema_name: Optional[str] = None,
    s3_folder_path: Optional[str] = None
):
    """
    Lists table names.
      - For Databricks: returns tables in the given catalog and schema.
      - For S3: returns valid CSV/Parquet file keys under the specified folder path.
    """
    if source.lower() == "databricks":
        if not catalog_name or not schema_name:
            raise HTTPException(status_code=400, detail="catalog_name and schema_name are required for Databricks.")
        try:
            fetcher = DatabricksDataFetcher()
            tables = fetcher.list_tables_in_schema(catalog_name, schema_name)
            return {"tables": tables}
        except Exception as e:
            logger.error(f"Error listing tables in Databricks: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    elif source.lower() == "s3":
        if not s3_folder_path:
            raise HTTPException(status_code=400, detail="s3_folder_path is required for S3.")
        try:
            fetcher = S3DataFetcher()
            bucket, prefix = parse_s3_path(s3_folder_path)
            all_files = fetcher.list_files(bucket, prefix=prefix)
            valid_files = [f for f in all_files if f.lower().endswith(('.csv', '.parquet'))]
            return {"tables": valid_files}
        except Exception as e:
            logger.error(f"Error listing tables in S3: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="source must be either 'databricks' or 's3'.")

@app.get("/sample")
async def sample_rows(
    source: str = Query(..., description="Data source: 'databricks' or 's3'"),
    catalog_name: Optional[str] = None,
    schema_name: Optional[str] = None,
    table_name: Optional[str] = None,
    s3_folder_path: Optional[str] = None,
    file_key: Optional[str] = None,
    num_rows: int = 5
):
    """
    Fetches a sample of rows.
      - For Databricks: requires catalog_name, schema_name, and table_name.
      - For S3: requires s3_folder_path; if file_key is not provided, uses the first valid CSV/Parquet file.
    """
    try:
        if source.lower() == "databricks":
            if not (catalog_name and schema_name and table_name):
                raise HTTPException(status_code=400, detail="catalog_name, schema_name, and table_name are required for Databricks.")
            fetcher = DatabricksDataFetcher()
            df = fetcher.sample_rows(catalog_name, schema_name, table_name, num_rows)
        elif source.lower() == "s3":
            if not s3_folder_path:
                raise HTTPException(status_code=400, detail="s3_folder_path is required for S3.")
            fetcher = S3DataFetcher()
            bucket, prefix = parse_s3_path(s3_folder_path)
            all_files = fetcher.list_files(bucket, prefix=prefix)
            valid_files = [f for f in all_files if f.lower().endswith(('.csv', '.parquet'))]
            if not valid_files:
                raise HTTPException(status_code=404, detail="No CSV/Parquet files found in the provided S3 folder.")
            chosen_key = file_key if file_key else valid_files[0]
            df = fetcher.sample_rows(bucket, chosen_key, num_rows)
        else:
            raise HTTPException(status_code=400, detail="source must be 'databricks' or 's3'.")
        return {"data": df.to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Error sampling rows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/columns")
async def list_columns(
    source: str = Query(..., description="Data source: 'databricks' or 's3'"),
    catalog_name: Optional[str] = None,
    schema_name: Optional[str] = None,
    table_name: Optional[str] = None,
    s3_folder_path: Optional[str] = None,
    file_key: Optional[str] = None
):
    """
    Lists columns (and their schema) for the specified table/file.
      - For Databricks: requires catalog_name, schema_name, and table_name.
      - For S3: requires s3_folder_path; if file_key is not provided, uses the first valid CSV/Parquet file.
    """
    try:
        if source.lower() == "databricks":
            if not (catalog_name and schema_name and table_name):
                raise HTTPException(status_code=400, detail="catalog_name, schema_name, and table_name are required for Databricks.")
            fetcher = DatabricksDataFetcher()
            columns = fetcher.list_columns_in_table(catalog_name, schema_name, table_name)
        elif source.lower() == "s3":
            if not s3_folder_path:
                raise HTTPException(status_code=400, detail="s3_folder_path is required for S3.")
            fetcher = S3DataFetcher()
            bucket, prefix = parse_s3_path(s3_folder_path)
            all_files = fetcher.list_files(bucket, prefix=prefix)
            valid_files = [f for f in all_files if f.lower().endswith(('.csv', '.parquet'))]
            if not valid_files:
                raise HTTPException(status_code=404, detail="No CSV/Parquet files found in the provided S3 folder.")
            chosen_key = file_key if file_key else valid_files[0]
            columns = fetcher.list_columns_in_file(bucket, chosen_key)
        else:
            raise HTTPException(status_code=400, detail="source must be 'databricks' or 's3'.")
        return {"columns": columns}
    except Exception as e:
        logger.error(f"Error listing columns: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/schema")
async def get_schema(
    source: str = Query(..., description="Data source: 'databricks' or 's3'"),
    catalog_name: Optional[str] = None,
    schema_name: Optional[str] = None,
    table_names: List[str] = Query(..., description="List of table names"),
    s3_folder_path: Optional[str] = None
):
    """
    Generates a unified schema mapping each table name to its column definitions,
    and saves it to a Python file (schema.py).
    """
    try:
        result_schema = {}
        if source.lower() == "databricks":
            if not (catalog_name and schema_name):
                raise HTTPException(status_code=400, detail="catalog_name and schema_name are required for Databricks.")
            fetcher = DatabricksDataFetcher()
            for table in table_names:
                columns = fetcher.list_columns_in_table(catalog_name, schema_name, table)
                result_schema[table] = columns
        elif source.lower() == "s3":
            if not s3_folder_path:
                raise HTTPException(status_code=400, detail="s3_folder_path is required for S3.")
            fetcher = S3DataFetcher()
            bucket, prefix = parse_s3_path(s3_folder_path)
            for table in table_names:
                # In S3, the table name is considered to be the file key.
                columns = fetcher.list_columns_in_file(bucket, table)
                result_schema[table] = columns
        else:
            raise HTTPException(status_code=400, detail="source must be either 'databricks' or 's3'.")

        # Build a Python file content with the schema variable
        python_schema = "schema = {\n"
        for table, columns in result_schema.items():
            # Wrap the string in triple quotes to preserve newline formatting
            python_schema += f"    '{table}': '''{columns}''',\n"
        python_schema += "}\n"

        # Save the schema to a Python file (schema.py)
        with open("schema.py", "w") as f:
            f.write(python_schema)

        return {"schema": result_schema}
    except Exception as e:
        logger.error(f"Error generating schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add this at the top with other imports
import numpy as np
import pandas as pd

import numpy as np
import json

def clean_nan_values(obj):
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(v) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None  # or you could use "NaN" as a string
    return obj

# Independent Summarizer Endpoints
@app.get("/summarizer/summarize")
async def summarize_table_get(
    source: str = Query(..., description="Data source: 'databricks' or 's3'"),
    catalog_name: Optional[str] = None,
    schema_name: Optional[str] = None,
    table_name: Optional[str] = None,
    table_names: Optional[List[str]] = Query(None, description="List of table names for summarization (Databricks only)"),
    s3_folder_path: Optional[str] = None,
    file_key: Optional[str] = None,
    n_samples: int = 3,
    use_llm: bool = True
):
    """
    (GET) Summarizes table(s).
      - For Databricks: If 'table_names' (a list) is provided, returns summaries for each table.
        Otherwise, uses 'table_name' for a single table summarization.
      - For S3: requires s3_folder_path; if file_key is not provided, uses the first valid CSV/Parquet file.
    Returns the enriched summary (or summaries).
    """
    try:
        import pandas as pd
        summarizer = Summarizer()
        if source.lower() == "databricks":
            if not (catalog_name and schema_name):
                raise HTTPException(status_code=400, detail="catalog_name and schema_name are required for Databricks.")
            fetcher = DatabricksDataFetcher()
            # If a list of table names is provided, iterate over each table.
            if table_names:
                summaries = {}
                for tname in table_names:
                    df = fetcher.sample_rows(catalog_name, schema_name, tname, num_rows=300000)
                    #check for null values in df
                    # if df.isnull().values.any():
                    #     raise HTTPException(status_code=400, detail="Null values present in the table")
                    summaries[tname] = summarizer.summarize(df, tname, n_samples, use_llm)
                    #pass info that summary generation is successful
                    logger.info(f"Summary successfully generated for table: {tname}")
                    # logger.info(f"Summary: {summaries[tname]}")


                return {"summaries": clean_nan_values(summaries)}
            elif table_name:
                df = fetcher.sample_rows(catalog_name, schema_name, table_name, num_rows=10)
                summary = summarizer.summarize(df, table_name, n_samples, use_llm)
                return {"summary": summary}
            else:
                raise HTTPException(status_code=400, detail="Either table_name or table_names must be provided for Databricks.")
        elif source.lower() == "s3":
            if not s3_folder_path:
                raise HTTPException(status_code=400, detail="s3_folder_path is required for S3.")
            fetcher = S3DataFetcher()
            bucket, prefix = parse_s3_path(s3_folder_path)
            all_files = fetcher.list_files(bucket, prefix=prefix)
            valid_files = [f for f in all_files if f.lower().endswith(('.csv', '.parquet'))]
            if not valid_files:
                raise HTTPException(status_code=404, detail="No CSV/Parquet files found in the provided S3 folder.")
            # For S3, we only expect a single file representing one table.
            chosen_key = file_key if file_key else valid_files[0]
            df = fetcher.sample_rows(bucket, chosen_key, num_rows=10)
            summary = summarizer.summarize(df, chosen_key, n_samples, use_llm)
            return {"summary": summary}
        else:
            raise HTTPException(status_code=400, detail="source must be either 'databricks' or 's3'.")
    except Exception as e:
        logger.error(f"Error summarizing table via GET: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Independent Vector Store Endpoint (state-changing, so remains POST)
class StoreDocumentRequest(BaseModel):
    enriched_text: str
    json_summary: dict
    db_path: str = "./chroma_db"
    collection_name: str = "feed_metadata"

@app.post("/vectorstore/store")
async def store_document(request: StoreDocumentRequest):
    """
    Stores a document (enriched text and JSON summary) in the vector store.
    """
    try:
        enricher = Enricher(db_path=request.db_path, collection_name=request.collection_name)
        result = enricher.process_and_store_table(request.json_summary)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error storing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Independent Retriever Endpoints
class RetrieveRequest(BaseModel):
    query: str
    k: int = 3
    db_path: str = "./chroma_db"
    collection_name: str = "feed_metadata"

@app.get("/retriever/retrieve")
async def retrieve_and_rerank_get(
    query: str,
    k: int = 3,
    db_path: str = "./vector_db_adapt",
    collection_name: str = "adapt"
):
    """
    (GET) Retrieves top-k documents for the given query and re-ranks them using the LLM.
    """
    try:
        retriever = Retriever(db_path=db_path, collection_name=collection_name)
        result = retriever.retrieve_and_rerank(query, k)
        return result
    except Exception as e:
        logger.error(f"Error retrieving documents via GET: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Updated the uvicorn command to reference this module as 'app'
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
