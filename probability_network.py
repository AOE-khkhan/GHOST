from probability_graph import ProbabilityGraph
from context_manager import ContextManager

class ProbabilityNetwork:
    def __init__(self, size):
        # create context
        self.context_manager = ContextManager(size)

        # all pgrapghs
        self.probability_graphs = [ProbabilityGraph(idx) for idx in range(len(self.context_manager.indices))]

    def run(self, expected):
        max_accuracy, max_confidence, max_prediction = 0, 0, 'VOID'

        for index, probability_graph in enumerate(self.probability_graphs):
            if index >= len(self.context_manager.contexts):
                continue

            accuracy, prediction, confidence = probability_graph.run(
                self.context_manager.contexts[index], expected, self.context_manager.context
            )

            if accuracy < max_accuracy:
                continue
            
            max_accuracy = accuracy
            max_confidence = confidence
            max_prediction = prediction

        self.context_manager.add(expected)
        return f'({max_prediction:>4}, {max_confidence:.4f})'

    def save(self):
        for probability_graph in self.probability_graphs:
            probability_graph.save()
