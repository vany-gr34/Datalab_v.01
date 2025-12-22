import matplotlib.pyplot as plt
import seaborn as sns
from visalustsation.BaseVisualizer import BaseVisualizer

class CategoricalVisualizer(BaseVisualizer):
    def __init__(self, features=None):
        self.features = features

    def visualize(self, df, target_column=None):
        categorical_features = df.select_dtypes(exclude="number").columns.tolist()
        if target_column and target_column in categorical_features:
            categorical_features.remove(target_column)
        if self.features:
            categorical_features = [f for f in categorical_features if f in self.features]

        figs = {}
        for col in categorical_features:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.countplot(data=df, x=col, ax=ax)
            ax.set_title(f"Countplot of {col}")
            ax.tick_params(axis='x', rotation=45)

            # Fix potential 'none' colors for Plotly conversion
            for patch in ax.patches:
                if patch.get_facecolor() == (0.0, 0.0, 0.0, 0.0):  # fully transparent
                    patch.set_facecolor('rgba(0,0,0,0)')

            figs[col] = fig
            # Close the figure if you don’t need to keep it in memory
            # plt.close(fig)

        return figs
