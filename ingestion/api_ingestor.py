"""
API ingestor for handling REST API data sources with authentication.
"""

import pandas as pd
import requests
from typing import Dict, Any, Optional
import json
from .base_ingestor import BaseIngestor, IngestionError
from .dataset import Dataset
from .metadata import MetadataManager


class APIIngestor(BaseIngestor):
    """
    Ingestor for REST API sources with support for authentication.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the API ingestor.

        Args:
            config: Configuration dictionary containing:
                - url: API endpoint URL
                - method: HTTP method (GET, POST, etc.)
                - headers: Optional HTTP headers
                - params: Optional query parameters
                - data: Optional request body data
                - auth_type: Authentication type (api_key, bearer, basic)
                - auth_config: Authentication configuration
                - json_path: Optional JSON path to extract data from response
                - timeout: Request timeout in seconds
        """
        super().__init__(config)
        self.url = config.get('url')
        self.method = config.get('method', 'GET').upper()
        self.headers = config.get('headers', {})
        self.params = config.get('params', {})
        self.data = config.get('data')
        self.auth_type = config.get('auth_type')
        self.auth_config = config.get('auth_config', {})
        self.json_path = config.get('json_path')
        self.timeout = config.get('timeout', 30)
        self.metadata_manager = MetadataManager()

        if not self.url:
            raise ValueError("url is required in config")

    def validate_config(self) -> bool:
        """Validate the API configuration."""
        if not self.url.startswith(('http://', 'https://')):
            self.logger.error("URL must start with http:// or https://")
            return False

        if self.method not in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']:
            self.logger.error(f"Unsupported HTTP method: {self.method}")
            return False

        return True

    def ingest(self) -> Dataset:
        """
        Ingest data from the API endpoint.

        Returns:
            Dataset: Standardized dataset object

        Raises:
            IngestionError: If ingestion fails
        """
        try:
            self._log_ingestion_start(f"API: {self.url}")

            if not self.validate_config():
                raise IngestionError("Invalid API configuration")

            # Make API request
            response_data = self._make_request()

            # Parse response data
            data = self._parse_response(response_data)

            # Validate and clean data
            data = self._validate_and_clean_data(data)

            # Generate metadata
            data_info = {
                'data': data,
                'url': self.url,
                'method': self.method,
                'response_status': response_data.get('status_code'),
                'response_headers': dict(response_data.get('headers', {}))
            }

            metadata = self.metadata_manager.generate_metadata(
                dataset_id=f"api_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                source_type='api',
                source_config=self.config,
                data_info=data_info
            )

            dataset = Dataset(data, metadata)

            self._log_ingestion_success(f"API: {self.url}", len(data))

            return dataset

        except requests.RequestException as e:
            self._log_ingestion_error(f"API: {self.url}", e)
            raise IngestionError(f"API request failed: {str(e)}")
        except Exception as e:
            self._log_ingestion_error(f"API: {self.url}", e)
            raise IngestionError(f"Failed to ingest from API: {str(e)}")

    def _make_request(self) -> Dict[str, Any]:
        """Make the HTTP request to the API."""
        # Set up authentication
        headers = self.headers.copy()
        if self.auth_type:
            headers.update(self._setup_authentication())

        # Make request
        try:
            response = requests.request(
                method=self.method,
                url=self.url,
                headers=headers,
                params=self.params,
                json=self.data if isinstance(self.data, dict) else None,
                data=self.data if isinstance(self.data, str) else None,
                timeout=self.timeout
            )

            response.raise_for_status()  # Raise exception for bad status codes

            return {
                'status_code': response.status_code,
                'headers': response.headers,
                'content': response.json() if 'application/json' in response.headers.get('content-type', '') else response.text
            }

        except requests.exceptions.RequestException as e:
            raise IngestionError(f"HTTP request failed: {str(e)}")

    def _setup_authentication(self) -> Dict[str, str]:
        """Set up authentication headers based on auth_type."""
        auth_headers = {}

        if self.auth_type == 'api_key':
            key_name = self.auth_config.get('key_name', 'X-API-Key')
            key_value = self.auth_config.get('key_value')
            if key_value:
                auth_headers[key_name] = key_value

        elif self.auth_type == 'bearer':
            token = self.auth_config.get('token')
            if token:
                auth_headers['Authorization'] = f"Bearer {token}"

        elif self.auth_type == 'basic':
            username = self.auth_config.get('username')
            password = self.auth_config.get('password')
            if username and password:
                import base64
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                auth_headers['Authorization'] = f"Basic {credentials}"

        return auth_headers

    def _parse_response(self, response_data: Dict[str, Any]) -> pd.DataFrame:
        """Parse the API response into a DataFrame."""
        content = response_data['content']

        # Handle JSON responses
        if isinstance(content, dict):
            if self.json_path:
                # Extract data from specific JSON path
                data = self._extract_json_path(content, self.json_path)
            else:
                data = content

            # Convert to DataFrame
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Single record, wrap in list
                return pd.DataFrame([data])
            else:
                raise IngestionError("Unable to parse JSON response into DataFrame")

        # Handle text/CSV responses
        elif isinstance(content, str):
            try:
                # Try to parse as CSV
                from io import StringIO
                return pd.read_csv(StringIO(content))
            except Exception:
                raise IngestionError("Unable to parse text response as CSV")

        else:
            raise IngestionError(f"Unsupported response content type: {type(content)}")

    def _extract_json_path(self, data: Dict, json_path: str):
        """Extract data from a JSON path (simple implementation)."""
        keys = json_path.split('.')
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and key.isdigit():
                current = current[int(key)]
            else:
                raise IngestionError(f"Invalid JSON path: {json_path}")

        return current

    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and perform basic cleaning on the API data.

        Args:
            data: Raw DataFrame from API

        Returns:
            Cleaned DataFrame
        """
        # Check for empty data
        if data.empty:
            raise IngestionError("API returned no data")

        # Normalize column names
        data.columns = [str(col).strip().lower().replace(' ', '_') for col in data.columns]

        # Basic data type inference
        data = data.convert_dtypes()

        # Log data quality issues
        missing_pct = (data.isnull().sum() / len(data) * 100).round(2)
        if (missing_pct > 50).any():
            self.logger.warning(f"High missing value percentages: {missing_pct[missing_pct > 50].to_dict()}")

        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            self.logger.info(f"Found {duplicate_count} duplicate rows")

        return data
