import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class VisualizerManager:
    def __init__(self, visualizers=None):
        self.visualizers = visualizers or []
        self.last_output = None
        self.visualization_metadata = {}

    def add_visualizer(self, visualizer):
        self.visualizers.append(visualizer)

    def run(self, df, target_column=None):
        """Generate all visualizations for a dataset.

        Returns:
            Dict mapping visualization names to matplotlib figure objects
        """
        all_figs = {}
        for viz in self.visualizers:
            figs = viz.visualize(df, target_column)
            all_figs.update(figs)
        self.last_output = all_figs
        return all_figs

    def export_json(self):
        """Export last visualization output as JSON with base64 images."""
        if self.last_output is None:
            logger.warning("No visualizations generated yet")
            return None

        from helpers.visualization_exporter import VisualizationExporter
        return VisualizationExporter.figs_to_json(self.last_output)

    def export_images(self, output_dir: str):
        """Export visualization figures to PNG files in output directory."""
        if self.last_output is None:
            logger.warning("No visualizations generated yet")
            return []

        from pathlib import Path
        from helpers.visualization_exporter import VisualizationExporter

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for name, fig in self.last_output.items():
            if hasattr(fig, 'savefig'):
                path = str(output_dir / f"{name}.png")
                VisualizationExporter.fig_to_png_file(fig, path)
                saved_paths.append(path)

        return saved_paths

    def get_summary(self) -> Dict[str, Any]:
        """Return summary of generated visualizations."""
        if self.last_output is None:
            return {'status': 'no_visualizations_generated'}

        return {
            'total_visualizations': len(self.last_output),
            'visualization_names': list(self.last_output.keys()),
            'metadata': self.visualization_metadata,
        }
