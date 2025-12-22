


class VisualizerManager:
    def __init__(self, visualizers=None):
        self.visualizers = visualizers or []

    def add_visualizer(self, visualizer):
        self.visualizers.append(visualizer)

    def run(self, df, target_column=None):
        all_figs = {}
        for viz in self.visualizers:
            figs = viz.visualize(df, target_column)
            all_figs.update(figs)
        return all_figs
