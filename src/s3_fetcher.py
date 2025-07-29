import boto3
import pandas as pd
import io
import logging
from typing import List, Optional

from utils import convert_json_structure

class S3DataFetcher:

    def __init__(self, region_name: Optional[str] = None):

        self.s3_client = boto3.client('s3', region_name=region_name)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def list_files(self, bucket_name: str, prefix: Optional[str] = None) -> List[str]:
        """
        Lists all object keys in the specified S3 bucket.
        """
        self.logger.info(f"Listing files in bucket '{bucket_name}' with prefix '{prefix}'")
        file_keys = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                for obj in page.get('Contents', []):
                    file_keys.append(obj['Key'])
            return file_keys
        except Exception as e:
            self.logger.error(f"Error listing files: {e}")
            raise

    def sample_rows(self, bucket_name: str, key: str, num_rows: int = 5) -> pd.DataFrame:
        """
        Fetches a sample of rows from a CSV or Parquet file stored in S3.
        Returns a DataFrame containing the first num_rows.
        """
        self.logger.info(f"Sampling {num_rows} rows from '{key}' in bucket '{bucket_name}'")
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            file_content = response['Body'].read()

            if key.lower().endswith('.csv'):
                df = pd.read_csv(io.BytesIO(file_content), nrows=num_rows)
            elif key.lower().endswith('.parquet'):
                # For Parquet, load the file and then take the top rows.
                df = pd.read_parquet(io.BytesIO(file_content), engine='pyarrow').head(num_rows)
            else:
                msg = "Unsupported file type. Only CSV and Parquet files are supported."
                self.logger.error(msg)
                raise ValueError(msg)
            return df
        except Exception as e:
            self.logger.error(f"Error sampling rows from '{key}': {e}")
            raise

    def list_columns_in_file(self, bucket_name: str, key: str) -> str:
        """
        Retrieves column metadata from a CSV or Parquet file stored in S3.
        Builds a JSON-like structure.
        """
        self.logger.info(f"Listing columns for file '{key}' in bucket '{bucket_name}'")
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            file_content = response['Body'].read()
            
            # Use a small sample to infer schema.
            if key.lower().endswith('.csv'):
                # Read only the header (or a few rows) to infer data types.
                df = pd.read_csv(io.BytesIO(file_content), nrows=100)
            elif key.lower().endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(file_content), engine='pyarrow')
            else:
                msg = "Unsupported file type. Only CSV and Parquet files are supported."
                self.logger.error(msg)
                raise ValueError(msg)
            
            # Build a list of dictionaries for each column.
            columns_metadata = []
            for col in df.columns:
                # Use the column's inferred dtype as type_name.
                col_dtype = str(df[col].dtype)
                # Infer nullability: here we assume column is nullable if any NaNs are present in sample.
                nullable = df[col].isnull().any()
                columns_metadata.append({
                    "name": col,
                    "type_name": col_dtype,
                    "nullable": nullable
                })
            
            return convert_json_structure(columns_metadata)
        except Exception as e:
            self.logger.error(f"Error listing columns for '{key}': {e}")
            raise