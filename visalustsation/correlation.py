import seaborn as sns
import matplotlib.pyplot as plt 
from visalustsation.BaseVisualizer import BaseVisualizer


class CorrelationVisualizer(BaseVisualizer):
    def visualize(self, df, target_column=None):
        numeric_features = df.select_dtypes(include="number").columns.tolist()
        if target_column and target_column in numeric_features:
            numeric_features.remove(target_column)

        corr = df[numeric_features].corr()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        return {"correlation_heatmap": fig}
