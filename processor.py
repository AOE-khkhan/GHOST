'''
Author: Joshua, Christian r0b0tx
Date: 1 Nov 2018
File: processor.py
Project: GHOST
'''

# import from python standard lib

class Processor:
    """docstring for Processor"""    
    
    def __init__(self, n_sensors=1, state_size=8, size=4, no_transformation=False):
        # if to show output
        self.log_state = True

        self.no_transformation = no_transformation

        # processor size in bits
        self.SIZE = n_sensors * size
        self.CONCEPT_SIZE = sum([2**x for x in range(self.SIZE)])

        # sensory binary data size
        self.STATE_SIZE = 2**state_size

        # matrix of nodes that process and project data
        self.nodes = []
        self.node_state_freq = []
        self.node_npdf_freq = [] #the freq of the concept being undecidable by pdf

        # holds the last information of size length
        self.context = []

        # holds the last concepts
        self.last_concepts = []
        self.last_concept_indices = []

        # holds the hidden relation between states and concepts
        self.node_transformations = []
        self.node_transformation_total_freq = []
        self.node_transformation_freq = []

        # used to model
        self.last_pdf_predictions = []
        self.last_transformation_models = []

        # the concept nodes index that are not found in the nodes list
        self.last_concepts_node_indicies_not_found = []
        for _ in range(self.STATE_SIZE):
            self.last_transformation_models.append([])
        
        # for dynamic programming
        self.memoized = {}
        self.MEMOIZED_SIZE = 256

    def addNode(self, concept_model):
        # add concepts
        self.nodes.append(concept_model)

        # matrix of concpts to states
        self.node_state_freq.append([0.0 for _ in range(self.STATE_SIZE)])
        self.node_npdf_freq.append([0.0 for _ in range(2)])
        
        # matrix of concepts to transformations
        self.node_transformations.append([]), self.node_transformation_freq.append([])
        for i in range(self.CONCEPT_SIZE):
            self.node_transformations[-1].append(Network((bin(i+1).count('1'), 1, 1), (Relu, Relu)))
            self.node_transformation_freq[-1].append(0)

    def addToContext(self, state):
        if len(self.context) >= self.SIZE:
            for _ in state:
                self.context.pop()
        
        for s in state:
            self.context.insert(0, s)
        return

    def getMaxProbabilityStates(self, data):
        mps = {}
        for concept_index, concept in enumerate(self.nodes):
            states = self.node_state_freq[concept_index]
            max_probability = max(states)
            max_probability_states = [ix for ix, x in enumerate(states) if x == max_probability]
            if data in max_probability_states:
                mps[concept] = max_probability_states

        return mps

    def getConcepts(self, context=None, size=None, reverse=False):
        normal_concepts = self.generateConcepts(context, size, reverse)

        # avoid repeating elements of concept consideration
        return normal_concepts

        context = self.context.copy()
        context_set = list(set(context))

        if len(context_set) == len(context):
            return normal_concepts

        for i, c in enumerate(context_set):
            if context.count(c) < 2:
                continue

            while c in context:
                context[context.index(c)] = str(i)

        other_concepts = self.generateConcepts(context, size, reverse)
        
        concepts = [] + normal_concepts
        for concept in other_concepts:
            if concept not in concepts:
                concepts.append(concept)

        return concepts

    def generateConcepts(self, context=None, size=None, reverse=False):
        if context == None: context = self.context.copy()
        if size == None: size = self.SIZE

        if reverse and len(context) < size:
            return []

        val = '0' if reverse else '1'
        concepts = []

        # get concept range based on precent concept as the context might not be up to max
        concept_range = sum([2**x for x in range(len(context))])
        m = size - 1
        for n in range(concept_range):
            bin_li = list(self.toBin(n+1, size))
            concept = [context[m - i] for i, x in enumerate(bin_li) if x == val]
            if reverse:
                concept.reverse()
            concept = tuple(concept)
            concepts.append(concept)
        return concepts

    def getMaxValueIndices(self, processes, return_max_weight=False):
        m = max(processes) if len(processes) > 0 else 0
        predicted_outputs = [state for state, x in enumerate(processes) if x == m]
        if return_max_weight:
            return predicted_outputs, m

        else:
            return predicted_outputs

    def getTransformations(self, concept, data):
        NN = Network((len(concept), 1, 1), (Relu, Relu))

        N = 100
        for i, xi in enumerate(list(range(1, N))):
            x = np.array([[xi]])
            y = x + 7

            NN.fit(x, y, loss=MSE, epochs=.1, batch_size=1, learning_rate=1e-1)

            prediction = NN.predict(x)

            y_true = []
            y_pred = []
            for index in range(len(y)):
                y_pred.append(np.argmax(prediction[index]))
                y_true.append(np.argmax(y[index]))

                print('iteration = {}, x = {}, pred = {}, y = {}'.format(i+1, x[index], prediction[index], y[index]))
        
    def log(self, output, title=None):
        if not self.log_state:
            return

        log(output, title)
        return

    def mean(self, li):
        if len(li) > 0:
            return sum(li)/len(li)

        else:
            return 0

    def normalize(self, li):
        s = sum(li) if type(li) != dict else sum(li.values())
        if s > 0:
            if type(li) == dict:
                return {x:li[x]/s for x in li}

            else:
                return [x/s for x in li]

        else:
            if type(li) == dict:
                return {x:0 for x in li}

            else:
                return [0 for _ in li]

    def process(self, state, data):
        if len(self.context) < self.SIZE:
            self.addToContext(state)
            return

        # update the node network
        self.update(data)

        # add data to context
        self.addToContext(state)

# -------------------------------------------the process instance-------------------------------------
        # the variables
        pdf_processes, transformation_processes = [], []
        transformation_models = []

        for _ in range(self.STATE_SIZE):
            # track the processesing
            pdf_processes.append(0)
            transformation_processes.append(0)

            transformation_models.append([])

        # get the states at this instance
        concepts = self.getConcepts()

        # index of the concepts
        concept_node_indices = []
        concepts_node_indicies_not_found = []

        # check if pdf can compute solution
        pdf_solvable = True

        for concept_index, concept in enumerate(concepts):
            concept_model = (concept_index, concept)
            compute_pdf = True

            if concept_model not in self.nodes:
                self.addNode(concept_model)

                # the id of the concept
                concept_node_index = self.nodes.index(concept_model)

                # save indices
                concept_node_indices.append(concept_node_index)
                concepts_node_indicies_not_found.append(concept_node_index)

                compute_pdf = False

# ===============================================probability prediction===============================
            if compute_pdf:

                # the id of the concept
                concept_node_index = self.nodes.index(concept_model)
                
                # save indices
                concept_node_indices.append(concept_node_index)

                # check the no pdf weight to see if the concept can not be solved by the pdf
                a, b = self.node_npdf_freq[concept_node_index]
                c = a / b if b > 0 else 0

                # if self.context[-2] == 57 and self.context[-3] == 96:
                #     print(concept_model, c)

                # check if it is pdf solvable
                pdf_solvable = True if c <= 0.5 and pdf_solvable == True else False
            
                # the max states:Purpose is to avoid noise
                concept2states_freq, state_weight = self.getMaxValueIndices(self.normalize(self.node_state_freq[concept_node_index]), True)
                
                factor = len(concept2states_freq)**-1

                for state in concept2states_freq:
                    weight = state_weight * factor

                    # include the weight in the processes
                    if weight > pdf_processes[state]:
                        pdf_processes[state] = weight

                        # if len(self.context) > 2 and self.context[-2] == 57 and self.context[-3] == 96:
                        #     print('cm = {}, state = {}, concept_weight = {} * {} = {}'.format(concept_model, state, state_weight, factor, weight))
            
            if self.no_transformation:
                continue
# ==============================================transformations================================================
            transformations = self.node_transformations[concept_node_index]

            tf = self.node_transformation_freq[concept_node_index]      #transformation freq

            # teh influence of the concept to transformation: decides the best transf for a concept
            transformation_weights = self.normalize(tf)

            ttf = sum(tf)
            transformation_weights = [tw * self.trustFactor(ttf) for tw in transformation_weights]

            max_transformation_weight_ids, max_transformation_weight = self.getMaxValueIndices(transformation_weights, True)

            for transformation_index in range(len(transformations)):
            # for transformation_index in max_transformation_weight_ids:
                # transformation influence
                transformation_weight = transformation_weights[transformation_index]

                # the transformation model
                transformation_model = transformations[transformation_index]

                # the tarsnformation model index
                transformation_model_id = (concept_node_index, transformation_index)

                # the ids for the transformations
                transformation, concept_y_index = transformation_model

                # transform the transformation
                concept_transform = self.solveTransformation(transformation, concepts[-1])

                concept_transform_model = (concept_y_index, concept_transform)
                if concept_transform_model not in self.nodes:
                    continue

                # concept transf index
                concept_transform_node_index = self.nodes.index(concept_transform_model)

                # the max states:Purpose is to avoid noise
                concept2states_freq, state_weight = self.getMaxValueIndices(self.normalize(self.node_state_freq[concept_transform_node_index]), True)
                
                factor = len(concept2states_freq)**-1
                
                # avoid noise of un influencial dataset
                if state_weight == 0:
                    continue

                for state in concept2states_freq:
                    # weight = max_transformation_weight * state_weight
                    weight = transformation_weights[transformation_index] * state_weight
                    transformation_models[state].append((concept_node_index, transformation_index))

                    if weight > transformation_processes[state]:
                        # track the transforation process
                        transformation_processes[state] = weight

                        # if self.context[-3] == 96 and self.context[-2] == 48:
                        # if self.context[-2] == 57 and self.context[-3] == 96:
                        if self.context[-1] == 57 and self.context[-2] == 49 and self.context[-3] == 96:
                            print('concept = {}-{}, transf_weight => {} / {} = {} * {} = {}-{}, ct = {}, s= {}-{}'.format(
                                concept_index, concepts[concept_index], format(tf[transformation_index], '.3f'), format(sum(tf), '.3f'), format(transformation_weight, '.3f'), format(state_weight, '.3f'), format(weight, '.3f'), format(factor, '.3f'), concept_transform_model, transformation_model, state
                                )
                            )
                        
        # if all the concepts are found then teh pdf solvable does not matter
        all_concepts_found = True if len(concepts_node_indicies_not_found) == 0 else False
        pdf_solvable = True if all_concepts_found else pdf_solvable

        # get the max vals
        pdf_predicted_outputs, pdf_max_weight = self.getMaxValueIndices(pdf_processes, True)
        transformation_predicted_outputs, transformation_max_weight = self.getMaxValueIndices(transformation_processes, True)
        
        # decide the algorithm value to use
        if pdf_solvable == False and self.no_transformation == False:
            predicted_outputs, max_weight = (transformation_predicted_outputs.copy(), transformation_max_weight)

        else:
            predicted_outputs, max_weight = (pdf_predicted_outputs.copy(), pdf_max_weight)            
        
        po = [pdf_solvable]

        self.last_concepts = concepts.copy()
        self.last_concept_indices = concept_node_indices.copy()
        self.last_pdf_predictions = pdf_predicted_outputs.copy()
        self.last_concepts_node_indicies_not_found = concepts_node_indicies_not_found.copy()
        self.last_transformation_models = transformation_models.copy()
        return predicted_outputs, format(max_weight, '.4f'), po

    def solveTransformation(self, transformation, concept):
        transform = list(transformation)
        for i, t in enumerate(transformation):
            if type(t) != str:
                continue

            index = int(t)
            transform[i] = concept[index]
        return tuple(transform)

    def toBin(self, c, n=8):
        if type(c) == str:
            c = ord(c)
            
        b = bin(c)[2:]
        return ''.join(['0' for _ in range(n - len(b))]) + b


    def trustFactor(self, x):
        if type(x) == list:
            x = len(x)
        return x / (1 + x) 

    def update(self, data):
        # get the nodes that match the reply in pdf
        max_probability_states = self.getMaxProbabilityStates(data)    

        # get the prvious values
        last_pdf_predictions = self.last_pdf_predictions
        last_concepts_node_indicies_not_found = lcinf = self.last_concepts_node_indicies_not_found

        last_concepts_node_indicies_found = [cni for cni in self.last_concept_indices if cni not in lcinf]
        all_concepts_found = True if len(last_concepts_node_indicies_not_found) == 0 else False

        last_transformation_models = self.last_transformation_models[data]
# ======================================updating pdf, pdf_concept and creating transformations===============        
        for concept_index, concept in enumerate(self.last_concepts):
            concept_model = (concept_index, concept)
            concept_node_index = self.nodes.index(concept_model)

# =====================================increment the node for pdf===============================
            self.node_state_freq[concept_node_index][data] += 1

# =========================================increment the node for not pdf solvable==============================
            if not all_concepts_found:
                self.node_npdf_freq[concept_node_index][0] += 1 if data not in last_pdf_predictions else 0
                self.node_npdf_freq[concept_node_index][1] += 1
            
            if self.no_transformation:
                continue

# ========================create transformation and update transf weights=========================================
            # at transformation is only fromed when there are missing concepts
            if all_concepts_found:
                continue

            for cni in last_concepts_node_indicies_found:
                for transformation_index, transformation in enumerate(self.node_transformations):
                    
                    # train the transformation
                    transformation.fit([list(concept)], [[data]], loss=MSE, epochs=.1, batch_size=1, learning_rate=1e-1)

                    # increment the value
                    if (cni, transformation_index) in last_transformation_models:
                        self.node_transformation_freq[cni][transformation_index] += 1

        return
