import re
import pandas as pd
import logging
import json
import os
import boto3
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_community.chat_models import BedrockChat
from langchain_google_vertexai import ChatVertexAI

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def check_if_vectordb_exists(schema_name: str) -> bool:
    """
    Checks if the folder ./chroma_db/<schema_name> exists and contains any files.
    """
    db_path = f"./chroma_db/{schema_name}"
    exists = os.path.isdir(db_path) and any(os.scandir(db_path))
    logger.debug(f"Checking if VectorDB exists at {db_path}: {exists}")
    return exists

def convert_json_structure(json_data):
    """
    Creates a json structured schema, given column names of each table.
    """
    result = ""
    for item in json_data:
        name = item['name']
        type_name = item['type_name'].lower()
        nullable = "true" if item['nullable'] else "false"
        result += f"|-- {name}: {type_name} (nullable = {nullable})\n"
    return result

def extract_json(text: str) -> str:
    """
    Extracts the JSON object from the given text by locating the first '{' and the last '}'.
    """
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1].strip()
    else:
        raise ValueError("No JSON object found in the response.")

def remove_trailing_commas(json_str: str) -> str:
    """
    Removes trailing commas before closing braces or brackets.
    """
    return re.sub(r',\s*(\}|\])', r'\1', json_str)

# System prompt template for dataset enrichment.
system_prompt = """
You are an expert data analyst working in the ad-tech industry. Your task is to annotate and enrich dataset summaries.
Follow these guidelines:
1. Always generate the dataset 'name' and a comprehensive 'dataset_description'.
2. For each field, provide a clear 'description' and assign a single-word 'semantic_type'
   (e.g., company, city, number, supplier, location, gender, longitude, latitude, url, ip, zip, email, etc.).
3. Use industry-specific terms accurately (e.g., LTV for Linear TV, CTV for Connected TV).
Return the updated JSON dictionary without any additional commentary.
"""

# Prompt tempelate to enrich summary description
prompt_template = """
You are a Data Analyst in a Advertising Tech company. You have been given a json structured summary. The json summary is structured as each object inside it being a single column with its metadata and contextual description. Each column structure is (there can be several columns):

{
    "column": "column_name",
    "properties": {
        "dtype": "dtype",
        "samples": ["a few samples"],
        "num_unique_values": "number of unique values within 10 selected sample rows",
        "semantic_type": "semantic type",
        "description": "A brief description"
    }
}

The json summary is 
{json_summary}

The json summary will also contain name, file name, and whole dataset description at top, and all field names (i.e. column names at the bottom). Apart from the json summary use your own adtech domain knowledge also. Try to include relevant KPI metrices, adtech keywords and semantics in the enriched summary.
Your output will be used for efficient RAG purposes primarily used by adtech Data Analyst. Thus try to include all the keywords and provided information too in your summary.
Only return the detailed enriched summary with no other explanation or starting sentence. Return at least one table name.

Enriched summary:"""

# Prompt tempelate for reranking
rerank_prompt_template = """
You are an expert at Adtech and also an expert Data Analyst. You have been given a user query and possible table schemas to run the query.

Think step by step:
- Firstly you have to analyse the user query thoroughly.
- Then you have to think about the columns that are involved in the query.
- Then you have to analyze the schemas provided to you.
- Finally, you have to give me the name of the table that best fits the query based on the schema analysis. This schema should have all the relevant columns needed to calculate and answer the query.
- Remember to return the actual table name mentioned (eg: report_inventory_appnexus) ONLY.

Query : {q}

Schemas : {s}

Only provide the name of the best matching table and no other explanation.
"""

# Define LLM 
def llm_pydantic():
    """
    Instantiate and return an LLM model using BedrockChat with Anthropic Claude.
    """
    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )
    model_kwargs = { 
        "temperature": 0.2,
        "max_tokens": 10000,
        "top_k": 50,
        "top_p": 0.9
    }
    model = BedrockChat(
        client=bedrock_runtime,
        model_id=model_id,
        model_kwargs=model_kwargs,
        streaming=False
    )
    return model

# Instantiate the LLM for use in other modules if needed.
llm = llm_pydantic()

def llm_gemini():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/ajay.kumar/Documents/bitbucket/dna-dataintelligence-playground-api/configs/service_account_key.json"
    model = ChatVertexAI(
        model="gemini-2.0-flash-001",
        temperature=0.2,
        max_tokens=8192,
        max_retries=6,
        stop=None,
        top_k=20,
        top_p=0.9,)

    return model

llm_new = llm_gemini()

# print(llm_new.invoke("whats up?"))

# # to invoke the LLM
# llm.invoke("you_prompt_here")
