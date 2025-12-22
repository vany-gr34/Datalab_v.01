from visalustsation.SmartVisualizer import SmartVisualizer
from visalustsation.manager import VisualizerManager
from visalustsation.NumericVisualizer import NumericVisualizer
from visalustsation.CategoricalVisualizer import CategoricalVisualizer
from visalustsation.correlation import CorrelationVisualizer


def build_visualizer(df):
    smart = SmartVisualizer()
    recommendations = smart.recommend(df)

    visualizers = []

    if any(r['type'] == 'numeric' for r in recommendations):
        visualizers.append(NumericVisualizer())

    if any(r['type'] == 'categorical' for r in recommendations):
        visualizers.append(CategoricalVisualizer())

    if any(r.get('plots') == ['correlation_heatmap'] for r in recommendations):
        visualizers.append(CorrelationVisualizer())

    return VisualizerManager(visualizers)
