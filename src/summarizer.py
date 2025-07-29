import pandas as pd
import warnings
import json
import logging
import re
from typing import Union, List, Dict
import datetime

from utils import system_prompt, llm, llm_new, extract_json, remove_trailing_commas

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Summarizer:
    def __init__(self) -> None:
        self.summary = None

    def check_type(self, dtype: str, value):
        """
        Cast value to the correct type for JSON serialization.
        """
        if "float" in str(dtype):
            return float(value)
        elif "int" in str(dtype):
            return int(value)
        else:
            return value

    def serialize_value(self, value):
        """
        Convert date or pandas.Timestamp objects to ISO formatted strings.
        """
        if isinstance(value, (datetime.date, pd.Timestamp)):
            try:
                return value.isoformat()
            except Exception:
                return str(value)
        return value

    def get_column_properties(self, df: pd.DataFrame, n_samples: int = 3) -> List[Dict]:
        """
        Create a list of dictionaries where each dictionary represents the properties of a DataFrame column (name, data type, stats, sample values, etc.).
        """
        properties_list = []

        for column in df.columns:
            dtype = df[column].dtype
            properties = {}

            # Identify numeric columns
            if dtype in [int, float, complex] or pd.api.types.is_numeric_dtype(dtype):
                properties["dtype"] = "number"
                properties["std"] = self.check_type(dtype, df[column].std())
                properties["min"] = self.check_type(dtype, df[column].min())
                properties["max"] = self.check_type(dtype, df[column].max())

            # Identify boolean columns
            elif dtype == bool:
                properties["dtype"] = "boolean"

            # Identify object columns (possible string, date, or category)
            elif dtype == object:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df[column], errors='raise')
                    properties["dtype"] = "date"
                except ValueError:
                    if df[column].nunique() / len(df[column]) < 0.5:
                        properties["dtype"] = "category"
                    else:
                        properties["dtype"] = "string"

            # Identify categorical columns
            elif pd.api.types.is_categorical_dtype(df[column]):
                properties["dtype"] = "category"

            # Identify datetime columns
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                properties["dtype"] = "date"

            else:
                properties["dtype"] = str(dtype)

            # For date columns, get min and max and convert to string.
            if properties["dtype"] == "date":
                try:
                    min_val = df[column].min()
                    max_val = df[column].max()
                except TypeError:
                    cast_date_col = pd.to_datetime(df[column], errors='coerce')
                    min_val = cast_date_col.min()
                    max_val = cast_date_col.max()
                properties["min"] = self.serialize_value(min_val)
                properties["max"] = self.serialize_value(max_val)

            # Gather sample values.
            nunique = df[column].nunique()
            non_null_values = df[column].dropna().unique()
            sample_count = min(n_samples, len(non_null_values))
            if sample_count > 0:
                samples = (pd.Series(non_null_values)
                           .sample(sample_count, random_state=42)
                           .tolist())
                # Convert any date-like sample values.
                samples = [self.serialize_value(s) for s in samples]
            else:
                samples = []

            properties["samples"] = samples
            properties["num_unique_values"] = nunique

            # Placeholders for LLM enrichment.
            properties["semantic_type"] = ""
            properties["description"] = ""

            properties_list.append({"column": column, "properties": properties})

        return properties_list

    def enrich(self, base_summary: dict) -> dict:
        """
        Enrich the base summary using the LLM.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(base_summary)}
        ]
        response_text = ""
        try:
            # response = llm.invoke(str(messages)) 
            response = llm.invoke(messages) 
            response_text = response.content.strip()
            logger.info("LLM response received. Extracting JSON...")
            logger.info(response_text)

            # Attempt to isolate the JSON
            json_string = extract_json(response_text)
            json_string = remove_trailing_commas(json_string)
            enriched_summary = json.loads(json_string)

        except Exception as e:
            logger.error(f"Error during LLM enrichment: {e}")
            logger.info("Raw LLM response:")
            logger.info(response_text)
            raise ValueError("LLM enrichment failed. Please ensure the LLM returns valid JSON.")

        return enriched_summary

    def summarize(self, df: pd.DataFrame,
        table_name: str,
        n_samples: int = 3,
        use_llm: bool = True
    ) -> dict:
        """
        Generate a structured summary of the table (DataFrame) and optionally enrich it with LLM.
        """
        data_properties = self.get_column_properties(df, n_samples)

        base_summary = {
            "name": table_name,
            "file_name": table_name,
            "dataset_description": "",
            "fields": data_properties
        }

        if use_llm:
            final_summary = self.enrich(base_summary)
        else:
            final_summary = base_summary

        final_summary["field_names"] = list(df.columns)
        return final_summary