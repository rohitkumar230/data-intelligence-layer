# Semantic Data Layer API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

An intelligent middleware that bridges the gap between a complex data lake (Databricks, S3) and natural language-powered agents. This project automates dataset discovery and summarization, enabling semantic search to find the most relevant dataset for any given query.

## Overview

Modern data lakes are vast and complex, making it difficult for users and AI agents to know which dataset to use for a specific task. This project solves that problem by creating an intelligent layer that:

1.  **Connects** to data sources like Databricks and S3 to discover available datasets.
2.  **Summarizes** each dataset by combining statistical analysis with LLM-powered enrichment to understand its structure and semantic meaning.
3.  **Indexes** these rich summaries into a vector database, creating a searchable long-term memory.
4.  **Provides** a simple API endpoint that takes a natural language query and returns the single best dataset to answer it.

This enables downstream applications, such as Text-to-SQL agents, to be more accurate and efficient by always starting with the correct data context.

## Architecture Flow

The system operates in two main phases: an offline **Indexing Phase** to build the knowledge base, and an online **Querying Phase** to serve user requests.

```
----------------------------- [ Indexing Phase (One-Time / Periodic) ] -----------------------------

[Data Lake: Databricks / S3]
           |
           v
[1. Senses (Data Fetchers)] -> Fetch sample data & schema
           |
           v
[2. Brain (Summarizer + LLM)] -> Create a Rich JSON Summary (stats + semantic descriptions)
           |
           v
[3. Memory (Enricher + Vector Store)] -> Generate embeddings & store summaries in ChromaDB


----------------------------------- [ Querying Phase (Live) ] ------------------------------------

[User's Natural Language Query]
           |
           v
[4. API Layer (FastAPI)] -> Entrypoint: /retriever/retrieve
           |
           v
[5. Recall (Retriever)]
      |
      +---> [Step A: Fast Semantic Search] -> Query ChromaDB to find top-k relevant datasets
      |
      +---> [Step B: LLM Re-ranking] -> Analyze top-k schemas to select the single best match
           |
           v
[Final Answer: Best Table Name + Rich JSON Summary]

```

## Core Features

-   **Unified Data Connectors**: Connects to both structured (Databricks Unity Catalog) and semi-structured (S3 files) data sources.
-   **AI-Powered Summarization**: Augments statistical profiles with LLM-generated descriptions and semantic type analysis for deep data understanding.
-   **Semantic Search**: Utilizes a vector database (ChromaDB) to find datasets based on conceptual meaning, not just keyword matches.
-   **LLM Re-ranking**: Employs a sophisticated two-stage retrieval process where an LLM reasons over the top candidates to make a final, highly-accurate selection.
-   **RESTful API**: Exposes all functionality through a clean, fast, and easy-to-use API built with FastAPI.

## Tech Stack

-   **Backend Framework**: FastAPI
-   **Data Handling**: Pandas
-   **AI / LLM Orchestration**: LangChain
-   **Vector Database**: ChromaDB
-   **Embedding Model**: Sentence-Transformers (`all-MiniLM-L6-v2`)
-   **LLM Integrations**: AWS Bedrock (Claude), Google Vertex AI (Gemini)
-   **Data Source Connectors**: `databricks-sql-connector`, `boto3`

## Setup and Installation

**1. Prerequisites**
-   Python 3.9+
-   Access to a Databricks workspace, an S3 bucket, and LLM providers (AWS Bedrock / Google Vertex AI).

**2. Clone the Repository**
```bash
git clone <your-repository-url>
cd <your-repository-name>
```

**3. Set up a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**4. Install Dependencies**
```bash
pip install -r requirements.txt
```
*(Note: A `requirements.txt` file would contain: `fastapi uvicorn pandas langchain chromadb sentence-transformers boto3 databricks-sql-connector python-dotenv langchain-google-vertexai`)*

**5. Configure Environment Variables**
Create a `.env` file in the root directory by copying the example file:
```bash
cp .env.example .env
```
Now, edit the `.env` file with your credentials:
```env
# Databricks Credentials
DATABRICKS_HOST="[https://your-workspace.cloud.databricks.com](https://your-workspace.cloud.databricks.com)"
DATABRICKS_TOKEN="dapi..."
HTTP_PATH="sql/protocolv1/o/..."

# AWS Credentials for Bedrock and S3
AWS_ACCESS_KEY_ID="your-aws-access-key"
AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
AWS_REGION="us-east-1"

# Google Cloud Credentials for Vertex AI (Optional)
# Point this to the path of your service account JSON file
GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service_account_key.json"
```

**6. Run the API Server**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
The API will now be running, and you can access the auto-generated documentation at `http://localhost:8000/docs`.

## Usage Workflow

The system is designed to be used in two distinct phases.

### Phase 1: Indexing Your Data Lake

This process is run once (or periodically) to populate the vector database with knowledge about your datasets.

**Step 1: Discover available tables**
Use the `GET /tables` endpoint to find the names of the tables/files you want to index.

**Step 2: Generate and enrich summaries**
Use the `GET /summarizer/summarize` endpoint with the list of table names from the previous step. This will generate the rich JSON summaries for all of them.

**Step 3: Store the summaries in the vector database**
For each summary generated, call the `POST /vectorstore/store` endpoint to process and save it to the knowledge base.

### Phase 2: Querying for Datasets

This is the live endpoint that your agents or applications will call.

**Step 1: Ask a question in natural language**
Make a `GET` request to the `/retriever/retrieve` endpoint with your question.

**Example using `curl`:**
```bash
curl -X GET "http://localhost:8000/retriever/retrieve?query=show%20me%20revenue%20by%20advertiser%20on%20CTV%20platforms"
```

**Step 2: Use the result**
The API will respond with the name of the best-matching table and its detailed JSON summary, which you can then use for downstream tasks like Text-to-SQL generation.

```json
{
  "best_table_name": "report_ctv_performance",
  "top_k_docs": {
    "ids": [
      "report_ctv_performance",
      "emea_quarterly_spend"
    ],
    "metadatas": [
      {
        "enriched_text": "This table contains detailed performance metrics for Connected TV (CTV) campaigns, including spend, impressions, clicks, and revenue, broken down by advertiser and date...",
        "json_summary": "{...}"
      },
      {...}
    ]
  }
}
```

## API Endpoint Highlights

| Endpoint                      | Method | Description                                                                    |
| ----------------------------- | ------ | ------------------------------------------------------------------------------ |
| `/summarizer/summarize`         | `GET`  | Generates a rich, AI-enhanced JSON summary for one or more datasets.           |
| `/vectorstore/store`            | `POST` | Stores a generated summary in the vector database (the indexing step).         |
| `/retriever/retrieve`           | `GET`  | **Primary endpoint.** Takes a natural language query and returns the best dataset. |
| `/tables`                       | `GET`  | Lists available tables/files from a specified data source.                     |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
