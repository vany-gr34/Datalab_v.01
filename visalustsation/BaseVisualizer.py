from abc import ABC, abstractmethod

class BaseVisualizer(ABC):
    @abstractmethod
    def visualize(self, df, target_column=None):
        """
        Generate plots for the dataframe.
        Returns a dict of figure objects keyed by feature name.
        """
        pass
