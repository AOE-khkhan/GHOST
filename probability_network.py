from probability_graph import ProbabilityGraph
from context_manager import ContextManager

class ProbabilityNetwork:
    def __init__(self, size):
        # create context
        self.context_manager = ContextManager(size)

        # all pgrapghs
        self.probability_graphs = [ProbabilityGraph(idx) for idx in range(len(self.context_manager.indices))]

    def run(self, input_value, verbose=0):
        # the previous context information before update
        old_context = self.context_manager.context
        old_contexts = self.context_manager.contexts.copy()

        # update the context manager
        self.context_manager.add(input_value)

        # the initials
        max_confidence, max_prediction = 0, 'VOID'

        for index, probability_graph in enumerate(self.probability_graphs):
            if index >= len(old_contexts):
                continue
            
            # update the p graph
            probability_graph.update(old_contexts[index], input_value, old_context)

            # infer with the pgraph
            prediction, confidence = probability_graph.predict(
                self.context_manager.contexts[index], self.context_manager.context
            )

            # if self.context_manager.context == '~co':
            #     print(self.context_manager.context, self.context_manager.contexts[index], prediction, confidence)

            if confidence < max_confidence:
                continue
            
            max_confidence = confidence
            max_prediction = prediction

        return f'({max_prediction:>4}, {max_confidence:.4f})'

    def save(self):
        for probability_graph in self.probability_graphs:
            probability_graph.save()
