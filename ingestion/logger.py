"""
Logging configuration and error handling for the ingestion layer.
"""

import logging
import logging.handlers
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime


class IngestionLogger:
    """
    Centralized logging system for the data ingestion layer.
    """

    def __init__(self, log_level: str = 'INFO', log_file: str = 'logs/ingestion.log',
                 max_bytes: int = 10*1024*1024, backup_count: int = 5):
        """
        Initialize the ingestion logger.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to the log file
            max_bytes: Maximum size of log file before rotation
            backup_count: Number of backup log files to keep
        """
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_file = Path(log_file)
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        # Create log directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure logger
        self.logger = logging.getLogger('data_ingestion')
        self.logger.setLevel(self.log_level)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self, name: str = None) -> logging.Logger:
        """
        Get a logger instance for a specific module.

        Args:
            name: Name of the logger (usually __name__)

        Returns:
            Logger instance
        """
        if name:
            return self.logger.getChild(name)
        return self.logger

    def log_ingestion_attempt(self, source_type: str, source_info: str,
                            config: Dict[str, Any]):
        """
        Log the start of an ingestion attempt.

        Args:
            source_type: Type of data source
            source_info: Information about the source
            config: Configuration used (sensitive info will be sanitized)
        """
        sanitized_config = self._sanitize_config(config)
        self.logger.info(f"Starting ingestion from {source_type}: {source_info}")
        self.logger.debug(f"Ingestion config: {json.dumps(sanitized_config, default=str)}")

    def log_ingestion_success(self, source_type: str, source_info: str,
                            record_count: int, dataset_id: str):
        """
        Log successful ingestion.

        Args:
            source_type: Type of data source
            source_info: Information about the source
            record_count: Number of records ingested
            dataset_id: Generated dataset ID
        """
        self.logger.info(
            f"Successfully ingested {record_count} records from {source_type}: {source_info}. "
            f"Dataset ID: {dataset_id}"
        )

    def log_ingestion_error(self, source_type: str, source_info: str,
                           error: Exception, error_context: Optional[Dict[str, Any]] = None):
        """
        Log ingestion errors with context.

        Args:
            source_type: Type of data source
            source_info: Information about the source
            error: The exception that occurred
            error_context: Additional context about the error
        """
        error_msg = f"Failed to ingest from {source_type}: {source_info}. Error: {str(error)}"

        if error_context:
            error_msg += f" Context: {json.dumps(error_context, default=str)}"

        self.logger.error(error_msg, exc_info=True)

        # Log stack trace for debugging
        self.logger.debug(f"Stack trace for {source_type} ingestion error", exc_info=True)

    def log_data_quality_issue(self, dataset_id: str, issue_type: str, details: Dict[str, Any]):
        """
        Log data quality issues found during ingestion.

        Args:
            dataset_id: Dataset identifier
            issue_type: Type of quality issue
            details: Details about the issue
        """
        self.logger.warning(
            f"Data quality issue in dataset {dataset_id}: {issue_type}. "
            f"Details: {json.dumps(details, default=str)}"
        )

    def log_security_alert(self, source_type: str, source_info: str, alert_type: str, details: Dict[str, Any]):
        """
        Log security-related alerts.

        Args:
            source_type: Type of data source
            source_info: Information about the source
            alert_type: Type of security alert
            details: Details about the alert
        """
        self.logger.critical(
            f"SECURITY ALERT - {alert_type} from {source_type}: {source_info}. "
            f"Details: {json.dumps(details, default=str)}"
        )

    def log_performance_metrics(self, operation: str, duration: float, record_count: int = None):
        """
        Log performance metrics for ingestion operations.

        Args:
            operation: Name of the operation
            duration: Time taken in seconds
            record_count: Number of records processed (optional)
        """
        if record_count:
            records_per_sec = record_count / duration if duration > 0 else 0
            self.logger.info(
                f"Performance: {operation} completed in {duration:.2f}s. "
                f"Processed {record_count} records at {records_per_sec:.2f} records/sec"
            )
        else:
            self.logger.info(f"Performance: {operation} completed in {duration:.2f}s")

    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize configuration to remove sensitive information before logging.

        Args:
            config: Original configuration

        Returns:
            Sanitized configuration
        """
        sanitized = config.copy()
        sensitive_keys = ['password', 'token', 'key', 'secret', 'auth']

        def sanitize_dict(d):
            for key, value in d.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    d[key] = '***REDACTED***'
                elif isinstance(value, dict):
                    sanitize_dict(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            sanitize_dict(item)

        sanitize_dict(sanitized)
        return sanitized


# Global logger instance
_ingestion_logger = None


def get_ingestion_logger() -> IngestionLogger:
    """Get the global ingestion logger instance."""
    global _ingestion_logger
    if _ingestion_logger is None:
        _ingestion_logger = IngestionLogger()
    return _ingestion_logger


def setup_ingestion_logging(log_level: str = 'INFO', log_file: str = 'logs/ingestion.log') -> logging.Logger:
    """
    Set up and return the ingestion logger.

    Args:
        log_level: Logging level
        log_file: Path to log file

    Returns:
        Logger instance
    """
    logger_instance = IngestionLogger(log_level=log_level, log_file=log_file)
    return logger_instance.get_logger()
