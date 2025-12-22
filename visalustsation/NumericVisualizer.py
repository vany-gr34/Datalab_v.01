import matplotlib.pyplot as plt
import seaborn as sns
from visalustsation.BaseVisualizer import BaseVisualizer

class NumericVisualizer(BaseVisualizer):
    def __init__(self, features=None):
        self.features = features

    def visualize(self, df, target_column=None):
        numeric_features = df.select_dtypes(include="number").columns.tolist()
        if target_column and target_column in numeric_features:
            numeric_features.remove(target_column)
        if self.features:
            numeric_features = [f for f in numeric_features if f in self.features]

        figs = {}
        for col in numeric_features:
            fig, axes = plt.subplots(1,2, figsize=(12,4))
            
            # Histogram
            sns.histplot(df[col], kde=True, ax=axes[0])
            axes[0].set_title(f"Histogram of {col}")
            
            # Boxplot
            sns.boxplot(x=df[col], ax=axes[1])
            axes[1].set_title(f"Boxplot of {col}")

            # Fix potential 'none' colors for Plotly conversion
            for axis in axes:
                for patch in getattr(axis, 'patches', []):
                    if patch.get_facecolor() == (0.0, 0.0, 0.0, 0.0):
                        patch.set_facecolor('rgba(0,0,0,0)')

            plt.tight_layout()
            figs[col] = fig
            # Close figure after storing if you want to save memory
            # plt.close(fig)

        return figs
