"""
Database ingestor for handling connections to various database systems.
"""

import pandas as pd
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from .base_ingestor import BaseIngestor, IngestionError
from .dataset import Dataset
from .metadata import MetadataManager


class DatabaseIngestor(BaseIngestor):
    """
    Ingestor for database sources including PostgreSQL, MySQL, and SQLite.
    """

    SUPPORTED_DBS = {
        'postgresql': 'postgresql://',
        'mysql': 'mysql://',
        'sqlite': 'sqlite:///'
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the database ingestor.

        Args:
            config: Configuration dictionary containing:
                - connection_string: Database connection string
                - db_type: Database type (postgresql, mysql, sqlite)
                - host: Database host
                - port: Database port
                - database: Database name
                - username: Database username
                - password: Database password
                - query: SQL query to execute
                - table: Table name to select from
                - limit: Optional row limit
        """
        super().__init__(config)
        self.connection_string = config.get('connection_string')
        self.db_type = config.get('db_type', 'postgresql')
        self.host = config.get('host', 'localhost')
        self.port = config.get('port')
        self.database = config.get('database')
        self.username = config.get('username')
        self.password = config.get('password')
        self.query = config.get('query')
        self.table = config.get('table')
        self.limit = config.get('limit')
        self.metadata_manager = MetadataManager()

        # Build connection string if not provided
        if not self.connection_string:
            self.connection_string = self._build_connection_string()

    def _build_connection_string(self) -> str:
        """Build database connection string from individual components."""
        if self.db_type not in self.SUPPORTED_DBS:
            raise ValueError(f"Unsupported database type: {self.db_type}")

        base_url = self.SUPPORTED_DBS[self.db_type]

        if self.db_type == 'sqlite':
            return f"{base_url}{self.database or ':memory:'}"

        # For other databases
        port_str = f":{self.port}" if self.port else ""
        return f"{base_url}{self.username}:{self.password}@{self.host}{port_str}/{self.database}"

    def validate_config(self) -> bool:
        """Validate the database configuration."""
        if not self.connection_string:
            self.logger.error("No connection string provided")
            return False

        if not self.query and not self.table:
            self.logger.error("Either 'query' or 'table' must be specified")
            return False

        # Test connection
        try:
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.logger.info("Database connection validated successfully")
            return True
        except SQLAlchemyError as e:
            self.logger.error(f"Database connection failed: {str(e)}")
            return False

    def ingest(self) -> Dataset:
        """
        Ingest data from the database.

        Returns:
            Dataset: Standardized dataset object

        Raises:
            IngestionError: If ingestion fails
        """
        try:
            self._log_ingestion_start(f"database: {self.db_type}")

            if not self.validate_config():
                raise IngestionError("Invalid database configuration")

            # Build query
            query = self._build_query()

            # Execute query and get data
            data = self._execute_query(query)

            # Validate and clean data
            data = self._validate_and_clean_data(data)

            # Generate metadata
            data_info = {
                'data': data,
                'query': query,
                'db_type': self.db_type,
                'table': self.table
            }

            metadata = self.metadata_manager.generate_metadata(
                dataset_id=f"db_{self.db_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                source_type='database',
                source_config=self.config,
                data_info=data_info
            )

            dataset = Dataset(data, metadata, source_type='database')

            self._log_ingestion_success(f"database: {self.db_type}", len(data))

            return dataset

        except Exception as e:
            self._log_ingestion_error(f"database: {self.db_type}", e)
            raise IngestionError(f"Failed to ingest from database: {str(e)}")

    def _build_query(self) -> str:
        """Build the SQL query to execute."""
        if self.query:
            query = self.query
        elif self.table:
            query = f"SELECT * FROM {self.table}"
        else:
            raise IngestionError("No query or table specified")

        # Add LIMIT if specified
        if self.limit:
            query += f" LIMIT {self.limit}"

        return query

    def _execute_query(self, query: str) -> pd.DataFrame:
        """Execute the SQL query and return results as DataFrame."""
        try:
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                result = conn.execute(text(query))
                # Get column names
                columns = result.keys()
                # Convert to DataFrame
                data = pd.DataFrame(result.fetchall(), columns=columns)
            return data
        except SQLAlchemyError as e:
            raise IngestionError(f"Query execution failed: {str(e)}")

    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and perform basic cleaning on the database data.

        Args:
            data: Raw DataFrame from database

        Returns:
            Cleaned DataFrame
        """
        # Check for empty data
        if data.empty:
            raise IngestionError("Query returned no data")

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

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a database table.

        Args:
            table_name: Name of the table to inspect

        Returns:
            Dictionary containing table information
        """
        try:
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                # Get column information
                if self.db_type == 'sqlite':
                    result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                    columns = [row[1] for row in result.fetchall()]
                else:
                    result = conn.execute(text(f"""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = '{table_name}'
                        ORDER BY ordinal_position
                    """))
                    columns = [row[0] for row in result.fetchall()]

                # Get row count
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = result.fetchone()[0]

                return {
                    'table_name': table_name,
                    'columns': columns,
                    'row_count': row_count
                }
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get table info: {str(e)}")
            return {}
