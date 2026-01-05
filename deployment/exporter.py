import os
import json
import pandas as pd
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from helpers.visualization_exporter import VisualizationExporter, StatisticalSummaryExporter

logger = logging.getLogger(__name__)


class ExportManager:
    """Main class for managing exports of various artifacts."""

    def __init__(self, export_dir: str = "exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _generate_metadata(self, artifact_type: str, **kwargs) -> Dict[str, Any]:
        """Generate standard metadata for exported artifacts."""
        return {
            "export_timestamp": self.timestamp,
            "artifact_type": artifact_type,
            "version": "1.0",
            "exported_by": "DataLabV.01",
            **kwargs
        }

    def export_dataset(self, df: pd.DataFrame, filename: str,
                      formats: List[str] = None) -> Dict[str, str]:
        """
        Export dataset in multiple formats.

        Args:
            df: DataFrame to export
            filename: Base filename (without extension)
            formats: List of formats ['csv', 'parquet', 'excel']

        Returns:
            Dict mapping format to file path
        """
        if formats is None:
            formats = ['csv']

        exported_files = {}
        base_path = self.export_dir / "datasets"

        for fmt in formats:
            base_path.mkdir(exist_ok=True)

            if fmt == 'csv':
                path = base_path / f"{filename}.csv"
                df.to_csv(path, index=False)
            elif fmt == 'parquet':
                path = base_path / f"{filename}.parquet"
                df.to_parquet(path, index=False)
            elif fmt == 'excel':
                path = base_path / f"{filename}.xlsx"
                df.to_excel(path, index=False)
            else:
                logger.warning(f"Unsupported format: {fmt}")
                continue

            exported_files[fmt] = str(path)
            logger.info(f"Exported dataset to {path}")

        # Export metadata
        metadata = self._generate_metadata(
            "dataset",
            shape=df.shape,
            columns=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()}
        )

        metadata_path = base_path / f"{filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        return exported_files

    def export_statistics(self, stats: Dict[str, Any], filename: str) -> str:
        """
        Export statistical summaries and metrics.

        Args:
            stats: Dictionary containing statistics
            filename: Output filename

        Returns:
            Path to exported file
        """
        base_path = self.export_dir / "statistics"
        base_path.mkdir(exist_ok=True)

        path = base_path / f"{filename}.json"

        # Add metadata
        export_data = {
            "metadata": self._generate_metadata("statistics"),
            "statistics": stats
        }

        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported statistics to {path}")
        return str(path)

    def export_model(self, model, model_name: str, preprocessing_pipeline=None,
                    metadata: Dict[str, Any] = None, formats: List[str] = None) -> Dict[str, str]:
        """
        Export trained model with preprocessing pipeline and metadata.

        Args:
            model: Trained model object
            model_name: Name of the model
            preprocessing_pipeline: Preprocessing pipeline (optional)
            metadata: Additional model metadata
            formats: List of formats to export ['joblib', 'json', 'png']

        Returns:
            Dict with paths to exported files
        """
        if formats is None:
            formats = ['joblib', 'json']

        base_path = self.export_dir / "models" / model_name
        base_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        if 'joblib' in formats:
            # Prefer exporting the underlying fitted estimator if the provided
            # object is a wrapper (e.g., BaseRegressor/BaseClassifier instances)
            obj_to_dump = getattr(model, 'model', model)

            # Export model
            model_path = base_path / f"{model_name}.joblib"
            joblib.dump(obj_to_dump, model_path)
            exported_files['joblib'] = str(model_path)

            # Export preprocessing pipeline if available
            if preprocessing_pipeline:
                pipeline_path = base_path / "preprocessing_pipeline.joblib"
                joblib.dump(preprocessing_pipeline, pipeline_path)
                exported_files['pipeline'] = str(pipeline_path)

        if 'json' in formats:
            # Export metadata
            model_metadata = self._generate_metadata(
                "model",
                model_type=type(model).__name__,
                **(metadata or {})
            )

            metadata_path = base_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2, default=str)

            exported_files['json'] = str(metadata_path)

        if 'png' in formats:
            # For models, PNG might not be applicable, but we can skip or add a placeholder
            # Perhaps in the future, generate a model diagram or something
            pass

        logger.info(f"Exported model {model_name} to {base_path}")
        return exported_files

    def export_visualization(self, fig, name: str,
                           formats: List[str] = None) -> Dict[str, str]:
        """
        Export visualization in multiple formats.

        Args:
            fig: Matplotlib or Plotly figure
            name: Visualization name
            formats: List of formats ['png', 'html', 'svg']

        Returns:
            Dict mapping format to file path
        """
        if formats is None:
            formats = ['png']

        exported_files = {}
        base_path = self.export_dir / "visualizations"
        base_path.mkdir(exist_ok=True)

        for fmt in formats:
            if fmt == 'png':
                if hasattr(fig, 'savefig'):  # matplotlib
                    path = base_path / f"{name}.png"
                    fig.savefig(path, dpi=100, bbox_inches='tight')
                else:  # plotly
                    path = base_path / f"{name}.png"
                    fig.write_image(path)
            elif fmt == 'html':
                if hasattr(fig, 'write_html'):  # plotly
                    path = base_path / f"{name}.html"
                    fig.write_html(path)
                else:  # matplotlib - convert to plotly for HTML
                    # This would require additional conversion logic
                    logger.warning("HTML export for matplotlib figures not implemented")
                    continue
            elif fmt == 'svg':
                if hasattr(fig, 'write_image'):  # plotly
                    path = base_path / f"{name}.svg"
                    fig.write_image(path, format='svg')
                else:  # matplotlib
                    path = base_path / f"{name}.svg"
                    fig.savefig(path, format='svg', bbox_inches='tight')
            else:
                logger.warning(f"Unsupported visualization format: {fmt}")
                continue

            exported_files[fmt] = str(path)
            logger.info(f"Exported visualization {name} to {path}")

        return exported_files

    def generate_report(self, report_data: Dict[str, Any], filename: str,
                       template: str = "default") -> str:
        """
        Generate comprehensive analytical report.

        Args:
            report_data: Dictionary containing report components
            filename: Output filename
            template: Report template to use

        Returns:
            Path to generated report
        """
        base_path = self.export_dir / "reports"
        base_path.mkdir(exist_ok=True)

        path = base_path / f"{filename}.json"

        report = {
            "metadata": self._generate_metadata("report", template=template),
            "content": report_data
        }

        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Generated report {filename} at {path}")
        return str(path)


# Convenience functions
def export_results(data: pd.DataFrame, stats: Dict[str, Any] = None,
                  export_dir: str = "exports") -> Dict[str, str]:
    """Export analytical results and statistics."""
    manager = ExportManager(export_dir)
    results = {}

    if data is not None:
        results.update(manager.export_dataset(data, "results"))

    if stats:
        results['statistics'] = manager.export_statistics(stats, "analysis_stats")

    return results

def export_model(model, model_name: str, preprocessing_pipeline=None,
                metadata: Dict[str, Any] = None, export_dir: str = "exports", formats: List[str] = None) -> Dict[str, str]:
    """Export trained model."""
    manager = ExportManager(export_dir)
    return manager.export_model(model, model_name, preprocessing_pipeline, metadata, formats)

def export_visualizations(visualizations: Dict[str, Any],
                         export_dir: str = "exports") -> Dict[str, str]:
    """Export visualizations."""
    manager = ExportManager(export_dir)
    results = {}

    for name, fig in visualizations.items():
        results.update(manager.export_visualization(fig, name))

    return results

def generate_report(report_data: Dict[str, Any], filename: str,
                   export_dir: str = "exports") -> str:
    """Generate comprehensive report."""
    manager = ExportManager(export_dir)
    return manager.generate_report(report_data, filename)
