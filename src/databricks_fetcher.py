import requests
import json
import pandas as pd
from databricks import sql
from typing import List, Optional
import logging
from configs import DATABRICKS_HOST, DATABRICKS_TOKEN, http_path

from utils import convert_json_structure

class DatabricksDataFetcher:
    """
    Fetches table information from Databricks Unity Catalog via REST API.
    """

    def __init__(self):
        self.host = DATABRICKS_HOST.rstrip("/")
        self.token = DATABRICKS_TOKEN
        self.http_path = http_path

        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        })

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def list_tables_in_schema(self, catalog_name: str, schema_name: str) -> List[str]:
        """
        Lists all table names in the specified catalog and schema.
        """
        endpoint = f"{self.host}/api/2.1/unity-catalog/tables"
        params = {
            "catalog_name": catalog_name,
            "schema_name": schema_name,
            "max_results": 100
        }
        self.logger.info(f"Listing tables in {catalog_name}.{schema_name}")
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()

        json_response = response.json()
        tables = json_response.get("tables", [])
        table_names = [table.get("name") for table in tables]
        return table_names

    def sample_rows(self, catalog_name: str, schema_name: str, table_name: str, num_rows: int = 5) -> pd.DataFrame:
        """
        Fetches a sample of rows from the specified table using Databricks SQL connector.
        Returns a DataFrame containing the sample rows.
        """
        self.logger.info(f"Sampling {num_rows} rows from table: {catalog_name}.{schema_name}.{table_name}")

        try:
            connection = sql.connect(
                server_hostname=self.host.replace("https://", ""),
                http_path=self.http_path,
                access_token=self.token
            )

            query = f"""
            SELECT * FROM {catalog_name}.{schema_name}.{table_name}
            TABLESAMPLE ({num_rows} ROWS)
            """
            cursor = connection.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
            cursor.close()
            connection.close()
            return df

        except Exception as e:
            self.logger.error(f"Error sampling rows: {str(e)}")
            raise

    def list_columns_in_table(self, catalog_name: str, schema_name: str, table_name: str):
        """
        Retrieves the columns for the specified table from the Databricks Unity Catalog API.
        The function returns the converted column structure.
        """
        endpoint = f"{self.host}/api/2.1/unity-catalog/tables/{catalog_name}.{schema_name}.{table_name}"
        self.logger.info(f"Fetching columns for table: {catalog_name}.{schema_name}.{table_name}")
        response = self.session.get(endpoint)
        response.raise_for_status()

        response_json = response.json()
        columns = response_json.get("columns", [])
        converted_columns = convert_json_structure(columns)
        return converted_columns
