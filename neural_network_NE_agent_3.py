import numpy as np
from matplotlib import style
from numpy import random
# from data_gen import spiral_data
import random
import math
import matplotlib.pyplot as plt

# =---------------------------------------------------------------------------------------

# |||||||||||||||||||||||||||||PLAN|||||||||||||||||||||||||||||

# first we need get the initial data set
# then compute for each genome in generation
# apply the highest confidence choice to the game (i.e if node 3 is 0.7 then the button press is going to be 'w')
# then we get the new dataset from the main game program by using a return function, then we compute again, if player or
# agent dies we stop the compute and get the final score for each genome and start the breeding of the child generation
# now we repeat by getting the initial dataset and then compute for each genome in each generation.


# need functions for getting the datasets from the main game program by calling it
# need a score function to call after the game has terminated
# need to change the evolution controller to manage the datasets and the scores and breeding
# we need to change the start_evolution function to not calculate score right after each compute

# need to ensure that we are able to make the input data into an array
# then we need to set the output layer to the number of possible choices of buttons

# Number of outputs '4'
# We are running on normal difficulty which has 8 bots, and speed 4
# Number of inputs are position[1] + "Me.pos", position[2] + "Me.ctpos", velocity[1] + "Me.Uv", velocity[2] + "Me.Rv",
# position_enemy[n] + "Chaser_n_.pos", position_enemy[n] + "Chaser_n_.ctpos",
# Hence 38 inputs; as (8 * 2 * 2) + (2 * 1 * 2)
# 8 number of bots
# 2 number of variables for each bots
# 2 data points for each variable in chaser
# 1 player
# 2 variables for each player
# 2 data points for each variable in player

# ------------------------------------------------------------------------------------------

# the datasets are defaulted to the variable X, y: X being the dataset itself, while Y is the actual answer. When
# implementing the neural network into the game, we would just use the from the game itself, to work with evolution

# Making A Layer Object For First Genration

file_best_weights = open('weights_best.txt', "a+")
file_best_biases = open('biases_best.txt', "a+")
genomes = 10

class layer_random:
    def __init__(self, inputs_n, neuron_n):
        self.inputs_n = inputs_n
        self.neuron_n = neuron_n
        self.weights = 0.1 * np.random.randn(inputs_n, neuron_n)
        self.biases = (np.random.randn(neuron_n)).reshape(1, neuron_n)

    def forward_pass_activated(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.output = np.maximum(0, self.output)
        return self.output


# Making A Layer Object Where We Can We Set Weights
class layers:
    def __init__(self, inputs_n, neuron_n, weights, biases):
        self.inputs_n = inputs_n
        self.neuron_n = neuron_n
        self.weights = weights
        self.biases = biases

    def forward_pass_activated(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.output = np.maximum(0, self.output)
        return self.output


# Making NN_Model
class NNM_1_Cascading_random:
    def __init__(self, input_layer, neurons_in_layer_n, output_layer):
        self.neurons_in_layer_n = neurons_in_layer_n
        self.output_layer = output_layer
        self.input_layer = input_layer
        self.neurons_in_next_layer = output_layer + 1  # vars to keep the while loop in check, initialized at outputlayer+1 so that we are able to enter the while loop
        self.neurons_in_layer = 0  # vars to keep the while loop in check
        self.loop_counter = 0  # vars to keep the while loop in check
        self.layers = [layer_random(input_layer, neurons_in_layer_n)]  # This is only thing that should be self...
        self.neuron_number = [
            neurons_in_layer_n]  # To keep in track of the number of neurons in network to calculate consecutive neurons in the layers

        while self.neurons_in_next_layer > output_layer:  # check var is assigned a higher var than output_layer to start the loop
            self.neurons_in_layer = math.floor(self.neuron_number[self.loop_counter] / (1+(1/7)))
            self.neurons_in_next_layer = math.floor(self.neurons_in_layer / (1+(1/7)))
            self.neuron_number.append(self.neurons_in_layer)
            self.loop_counter = self.loop_counter + 1

        for o in range(len(self.neuron_number) - 1):
            self.layers.append(layer_random(int(self.neuron_number[o]), int(self.neuron_number[o + 1])))

        self.layers.append(layer_random(int(self.neurons_in_layer), int(output_layer)))

    def Compute(self, input_data):
        self.output = self.layers[0].forward_pass_activated(input_data)
        self.layers_output = [self.output]
        for i in range(1, len(self.layers)):
            self.output = self.layers[i].forward_pass_activated(self.output)
            self.layers_output.append(self.output)
        # Softmax Activation
        exp_values = np.exp(self.layers_output[-1] - np.max(self.layers_output[-1], axis=1, keepdims=True))
        output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return output

class NNM_1_Cascading:
    def __init__(self, input_layer, neurons_in_layer_n, output_layer, weights, biases):
        self.weights = weights
        self.biases = biases
        self.neurons_in_layer_n = neurons_in_layer_n
        self.output_layer = output_layer
        self.input_layer = input_layer
        self.neurons_in_next_layer = output_layer + 1  # vars to keep the while loop in check, initialized at outputlayer+1 so that we are able to enter the while loop
        self.neurons_in_layer = 0  # vars to keep the while loop in check
        self.loop_counter = 0  # vars to keep the while loop in check
        self.layers = [
            layers(input_layer, neurons_in_layer_n, weights[0], biases[0])]  # This is only thing that should be self...
        self.neuron_number = [
            neurons_in_layer_n]  # To keep in track of the number of neurons in network to calculate consecutive neurons in the layers

        while self.neurons_in_next_layer > output_layer:  # check var is assigned a higher var than output_layer to start the loop
            self.neurons_in_layer = math.floor(self.neuron_number[self.loop_counter] / (1+(1/7)))
            self.neurons_in_next_layer = math.floor(self.neurons_in_layer / (1+(1/7)))
            self.neuron_number.append(self.neurons_in_layer)
            self.loop_counter = self.loop_counter + 1

        # make weights an array so that we can create the layers in the network
        for o in range(len(self.neuron_number) - 1):
            self.layers.append(layers(int(self.neuron_number[o]), int(self.neuron_number[o + 1]), self.weights[o + 1],
                                      self.biases[o + 1]))

        self.layers.append(layers(int(self.neurons_in_layer), int(output_layer), self.weights[-1], self.biases[-1]))

    def Compute(self, input_data):
        self.output = self.layers[0].forward_pass_activated(input_data)
        self.layers_output = [self.output]
        for i in range(1, len(self.layers)):
            self.output = self.layers[i].forward_pass_activated(self.output)
            self.layers_output.append(self.output)
        # Softmax Activation
        exp_values = np.exp(self.layers_output[-1] - np.max(self.layers_output[-1], axis=1, keepdims=True))
        output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return output


def first_generation(number_genomes, input_layer_data, neurons_in_layer_data, output_layer_data):
    generation_of_genomes_models = []
    for i in range(number_genomes):
        generation_of_genomes_models.append(
            NNM_1_Cascading_random(input_layer_data, neurons_in_layer_data, output_layer_data))
    return generation_of_genomes_models


# for simulating the generation consider or rather change it to simulate only one data point and then
# calculate score for only one simulation hence no need for an average

class Evolution:
    def __init__(self):
        global genomes

        self.genome_counter = -1
        self.sorted_neural_net = []
        self.child_neural_net_bias = []
        self.child_neural_net_weights = []
        self.child_generation = []
        self.biases_of_neural_nets = []
        self.weights_of_neural_nets = []
        self.generation = first_generation(genomes, 12288, 64, 15)
        self.scores_nets = [0] * len(self.generation)
        self.generation_number = 0
        self.manual_mutation_parent_1 = 0
        self.manual_mutation_parent_2 = 0
        self.manual_breed_rate = 0
        self.manual_child_per_fam = 0
        self.manual_mutation_overall = 0
        self.highest_score = -1
        self.tolerance = 0
        self.environmental_stress_levels = 0

    def merge_generation_n_score(self, list1, list2):
        merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
        return merged_list

    def simulate_generation(self, observation, is_dead, score, first_time):


        if is_dead == True and first_time == True:
            self.latest_score = 0
            self.__init__(self=Evolution)

        if is_dead == True:
            self.latest_score = 0
            self.genome_counter = 1 + self.genome_counter

        output_signal = []

        if is_dead == False:
            self.latest_score += score
            # make a function which gets the dataset here
            data = self.generation[self.genome_counter].Compute(observation)
            # need to add code here which will talk with the main program which will change the keypress
            # 0 = W, 1 = A, 2 = S, 3 = D
            location_of_highest_confidence = np.where(data[0] == np.max(data[0]))
            if location_of_highest_confidence[0][0] == 0:
                output_signal = 0
            if location_of_highest_confidence[0][0] == 1:
                output_signal = 1
            if location_of_highest_confidence[0][0] == 2:
                output_signal = 2
            if location_of_highest_confidence[0][0] == 3:
                output_signal = 3
            if location_of_highest_confidence[0][0] == 4:
                output_signal = 4
            if location_of_highest_confidence[0][0] == 5:
                output_signal = 5
            if location_of_highest_confidence[0][0] == 6:
                output_signal = 6
            if location_of_highest_confidence[0][0] == 7:
                output_signal = 7
            if location_of_highest_confidence[0][0] == 8:
                output_signal = 8
            if location_of_highest_confidence[0][0] == 9:
                output_signal = 9
            if location_of_highest_confidence[0][0] == 10:
                output_signal = 10
            if location_of_highest_confidence[0][0] == 11:
                output_signal = 11
            if location_of_highest_confidence[0][0] == 12:
                output_signal = 12
            if location_of_highest_confidence[0][0] == 13:
                output_signal = 13
            if location_of_highest_confidence[0][0] == 14:
                output_signal = 14
            self.scores_nets[self.genome_counter] = self.latest_score
            return output_signal


        if self.genome_counter == len(self.generation):
            self.sort_weights_and_biases(self=Evolution)


    def sort_weights_and_biases(self):
        __1_layer_weights = []
        __1_layer_biases = []

        self.weights_of_neural_nets = []
        self.biases_of_neural_nets = []

        # we get a matrix of the weights of the good performing genomes in the generation, then using we send the weights of the good performing genomes in the generation to another function where we can mutate them.
        self.scores_nets = list(self.scores_nets)
        self.generation = list(self.generation)
        self.sorted_neural_net = self.merge_generation_n_score(self=Evolution, list1=self.scores_nets, list2=self.generation)
        self.sorted_neural_net.sort(key=lambda x: x[0], reverse=True)
        for g in self.sorted_neural_net:
            for o in g[1].layers:
                __1_layer_weights.append(o.weights)
                __1_layer_biases.append(o.biases)
            self.weights_of_neural_nets.append(__1_layer_weights)
            __1_layer_weights = []
            self.biases_of_neural_nets.append(__1_layer_biases)
            __1_layer_biases = []
            # we send the layer to the mutate function here (redacted but not)
        self.mutatations(self=Evolution)

    # first we check whether or not we want to randomize the weights(when we are randomizing the weights we need to
    # keep a randomizing rate i.e we only want 20%) if yes we would want to randomize the weights for the new
    # generation using a random number gen from ranges between -1 to 1 if we decide not to randomize the weights we
    # decide to breed or not using a breed rate we decide how many we would want to breed if we would want to breed
    # we would run a random number check that spits out a number 1 or 0 which we us to decide whether or not we pick
    # the weight from parent A or B, then using this iterative process we decide all the weights in the matrix if we
    # do not want to breed we just copy the best performing neural nets from the previous generation the percentages
    # for whether we mutate(randomize) = 25% :: Percentage for breeding is = 50% :: Percentage for copies from
    # previous generation is 25% making a 100%

    # also the best neural net is at the first index in the weights array while te worst is in the last
    # parent a is the always the better parent.

    # setting randomization rate into the code better for running program.

    def mutatations(self):

        child_weight = []
        child_node = []
        child_net = []

        child_bias = []
        child_node_bias = []
        child_net_bias = []

        number_of_total_biases = 0
        number_of_total_weights = 0

        average_of_biases = 0
        average_of_weights = 0
        # running average for mutation range parent 2
        for q in self.biases_of_neural_nets[0]:
            for u in q:
                for t in u:
                    number_of_total_biases += 1
                    average_of_biases = average_of_biases + t

        average_of_biases = average_of_biases / number_of_total_biases

        for q in self.weights_of_neural_nets[0]:
            for u in q:
                for t in u:
                    number_of_total_weights += 1
                    average_of_weights = average_of_weights + t

        average_of_weights = average_of_weights / number_of_total_weights

        randomization_rate = 0.1978 + self.manual_mutation_overall
        randomization_rate_for_parent_2 = 0.6 + self.manual_mutation_parent_2
        randomization_range_for_parent_2_biases = average_of_biases / 2
        randomization_range_for_parent_2_weights = average_of_weights / 2
        randomization_rate_for_parent_1 = 0.09 + self.manual_mutation_parent_1
        randomization_range_for_parent_1_biases = average_of_biases / 8
        randomization_range_for_parent_1_weights = average_of_weights / 8
        breed_rate = 0.3 + self.manual_breed_rate  # closer to one means it picks better parent more often that worse parent, while closer to 1 is otherwise
        number_of_children_per_fam = 2 + self.manual_child_per_fam


        exit_breeding_weights = False
        exit_breeding_biases = False
        # loop for going through the array of weights and breeding

        self.child_neural_net_weights = []
        self.child_neural_net_bias = []

        for i in range(1, len(self.biases_of_neural_nets)):
            breed_y_n_for_bias = True
            for o in range(len(self.biases_of_neural_nets)):
                if o == i or o > i:
                    breed_y_n_for_bias = False
                if breed_y_n_for_bias:
                    parent_a_bias = self.biases_of_neural_nets[i - 1]
                    parent_b_bias = self.biases_of_neural_nets[o]

                for m in range(number_of_children_per_fam):

                    # counters for arranging the biases into the right place
                    biases_current = 0

                    # counters for checking the same position of weight in parent b in network
                    layer_counter_bias = 0
                    node_counter_bias = 0
                    bias_in_node_counter = 0

                    if not exit_breeding_biases:
                        for d in parent_a_bias:
                            layer_counter_bias += 1
                            for v in d:
                                biases_current = len(v)
                                node_counter_bias += 1
                                for h in v:
                                    bias_in_node_counter += 1
                                    choice_1_randomization = np.array(
                                        random.choices([0, 1], weights=((1 - randomization_rate), randomization_rate),
                                                       k=1))
                                    choice_1_breed = np.array(
                                        random.choices([0, 1], weights=((1 - breed_rate), breed_rate), k=1))

                                    if choice_1_randomization == 1:
                                        rand_bias = random.uniform(-1, 1)
                                        child_bias.append(rand_bias)

                                    if choice_1_randomization == 0:
                                        if choice_1_breed == 1:
                                            trait_1 = parent_b_bias[layer_counter_bias - 1][node_counter_bias - 1][
                                                bias_in_node_counter - 1]
                                            choice_2_randomization_parent_2 = np.array(random.choices([0, 1], weights=(
                                            (1 - randomization_rate_for_parent_2), randomization_rate_for_parent_2),
                                                                                                      k=1))

                                            if choice_2_randomization_parent_2 == 1:
                                                trait_1 = trait_1 + random.uniform(
                                                    -randomization_range_for_parent_2_biases,
                                                    randomization_range_for_parent_2_biases)
                                                child_bias.append(trait_1)

                                            if choice_2_randomization_parent_2 == 0:
                                                child_bias.append(trait_1)
                                        if choice_1_breed == 0:
                                            trait_2 = parent_b_bias[layer_counter_bias - 1][node_counter_bias - 1][
                                                bias_in_node_counter - 1]
                                            choice_2_randomization_parent_1 = np.array(random.choices([0, 1], weights=(
                                            (1 - randomization_rate_for_parent_1), randomization_rate_for_parent_1),
                                                                                                      k=1))

                                            if choice_2_randomization_parent_1 == 1:
                                                trait_2 = trait_2 + random.uniform(
                                                    -randomization_range_for_parent_1_biases,
                                                    randomization_range_for_parent_1_biases)
                                                child_bias.append(trait_2)

                                            if choice_2_randomization_parent_1 == 0:
                                                child_bias.append(trait_2)

                                child_node_bias.append(child_bias)
                                child_bias = []
                                bias_in_node_counter = 0
                            node_counter_bias = 0
                            child_node_bias = np.array(child_node_bias).reshape(1, biases_current)
                            child_net_bias.append(child_node_bias)
                            biases_current = 0
                            child_node_bias = []
                        layer_counter_bias = 0
                        self.child_neural_net_bias.append(child_net_bias)
                        child_net_bias = []
                        if len(self.child_neural_net_bias) >= math.ceil(len(self.generation) / 2):
                            exit_breeding_biases = True
                            break
            break

        for i in range(1, len(self.weights_of_neural_nets)):
            breed_y_n = True
            for o in range(len(self.weights_of_neural_nets)):
                if o == i or o > i:
                    breed_y_n = False
                if breed_y_n:
                    parent_b = self.weights_of_neural_nets[o]
                    parent_a = self.weights_of_neural_nets[i - 1]

                for n in range(number_of_children_per_fam):
                    # loop for going through the weights and them mutating them.

                    # counters for arranging the weights into the right shape
                    neurons_next = 0
                    neurons_current = 0

                    # counters for checking the same position of weight in parent b network
                    layer_counter = 0
                    node_counter = 0
                    weight_in_node_counter = 0

                    if not exit_breeding_weights:
                        for r in parent_a:
                            if not exit_breeding_weights:
                                layer_counter += 1
                            for q in r:
                                node_counter += 1
                                neurons_next += 1
                                neurons_current = len(q)
                                for w in q:
                                    weight_in_node_counter += 1
                                    choice = np.array(
                                        random.choices([0, 1], weights=((1 - randomization_rate), randomization_rate),
                                                       k=1))
                                    choice_breed = np.array(
                                        random.choices([0, 1], weights=((1 - breed_rate), breed_rate), k=1))

                                    if choice == 1:  # this is mutate
                                        rand_weight = random.uniform(-1, 1)
                                        child_weight.append(rand_weight)

                                    if choice == 0:
                                        if choice_breed == 1:
                                            trait_1_weight = parent_b[layer_counter - 1][node_counter - 1][
                                                weight_in_node_counter - 1]
                                            choice_2_randomization_weight_parent_2 = np.array(random.choices([0, 1], weights =((1 - randomization_rate_for_parent_2),randomization_rate_for_parent_2),k=1))
                                            if choice_2_randomization_weight_parent_2 == 1:
                                                trait_1_weight = trait_1_weight + random.uniform(
                                                    -randomization_range_for_parent_2_weights,
                                                    randomization_range_for_parent_2_weights)
                                                child_weight.append(trait_1_weight)

                                            if choice_2_randomization_weight_parent_2 == 0:
                                                child_weight.append(trait_1_weight)
                                        if choice_breed == 0:
                                            trait_2_weight = parent_b[layer_counter - 1][node_counter - 1][
                                                weight_in_node_counter - 1]
                                            choice_2_randomization_parent_1_weight = np.array(random.choices([0, 1], weights=((1 - randomization_rate_for_parent_1),randomization_rate_for_parent_1),k=1))
                                            if choice_2_randomization_parent_1_weight == 1:
                                                trait_2_weight = trait_2_weight + random.uniform(
                                                    -randomization_range_for_parent_1_weights,
                                                    randomization_range_for_parent_1_weights)
                                                child_weight.append(trait_2_weight)

                                            if choice_2_randomization_parent_1_weight == 0:
                                                child_weight.append(trait_2_weight)

                                child_node.append(child_weight)
                                child_weight = []
                                weight_in_node_counter = 0
                            node_counter = 0
                            child_node = np.array(child_node).reshape(neurons_next, neurons_current)
                            child_net.append(child_node)
                            neurons_current = 0
                            neurons_next = 0
                            child_node = []
                        layer_counter = 0
                        self.child_neural_net_weights.append(child_net)
                        child_net = []
                        if len(self.child_neural_net_weights) >= math.ceil(len(self.generation) / 2):
                            exit_breeding_weights = True
                            break
            break
        self.create_generation(self=Evolution)

    def create_generation(self):

        new_score = np.max(self.scores_nets)

        global file_best_weights, file_best_biases

        if new_score > self.highest_score:
            self.highest_score = new_score
            self.tolerance = 0
            if self.environmental_stress_levels > 0:
                self.manual_mutation_parent_2 -= 0.015
                self.manual_child_per_fam -= 1
                self.manual_breed_rate += 0.015
                self.manual_mutation_parent_1 -= 0.055
                self.environmental_stress_levels -= 1

        if new_score == self.highest_score or new_score < self.highest_score:
            self.tolerance += 1

        if self.tolerance > 2:
            # we can try different attributes here
            self.environmental_stress_levels += 1
            self.manual_child_per_fam += 1
            if self.manual_mutation_parent_2 < 0.5:
                self.manual_mutation_parent_2 += 0.015
            if self.manual_breed_rate > -0.26:
                self.manual_breed_rate -= 0.015
            if self.manual_mutation_parent_1 < 0.9:
                self.manual_mutation_parent_1 += 0.055
            self.tolerance = 0

        percentage_copied = 0.9
        percentage_random = 0.1

        self.child_generation = []
        number_of_genomes = len(self.generation)
        input_layer_data = self.generation[1].input_layer
        neurons_in_layer_data = self.generation[1].neurons_in_layer_n
        output_layer_data = self.generation[1].output_layer

        for q in range(len(self.child_neural_net_weights)):
            self.child_generation.append(
                NNM_1_Cascading(input_layer=input_layer_data, neurons_in_layer_n=neurons_in_layer_data,
                                output_layer=output_layer_data, weights=self.child_neural_net_weights[q],
                                biases=self.child_neural_net_bias[q]))

        number_of_copied_genomes = math.floor(
            (number_of_genomes - len(self.child_neural_net_weights)+1) * percentage_copied)
        number_of_random_genomes = math.floor(
            (number_of_genomes - len(self.child_neural_net_weights)+1) * percentage_random)

        while (number_of_random_genomes + number_of_copied_genomes + len(
                self.child_neural_net_weights)) > number_of_genomes+1:
            number_of_random_genomes -= 1
            number_of_copied_genomes -= 1

        for q in range(number_of_random_genomes):
            self.child_generation.append(
                NNM_1_Cascading_random(input_layer=input_layer_data, neurons_in_layer_n=neurons_in_layer_data,
                                       output_layer=output_layer_data))

        for q in range(number_of_copied_genomes):
            self.child_generation.append(self.sorted_neural_net[q][1])


        # writting function here that stors the matrix
        open("weights_best.txt", "w").close()
        open("biases_best.txt", "w").close()
        file_best_weights.write(str(self.weights_of_neural_nets[0]))
        file_best_biases.write(str(self.biases_of_neural_nets[0]))

        self.child_neural_net_weights = []
        self.child_neural_net_bias = []
        self.weights_of_neural_nets = []
        self.biases_of_neural_nets = []

        self.generation_number += 1
        self.generation = self.child_generation
        self.genome_counter = 0

        print('Evolution Info:')
        print('Generation Scores: '+str(self.sorted_neural_net))
        print('Generation Length: '+ str(len(self.sorted_neural_net)))
        print('Generation Number: '+str(self.generation_number))

        self.scores_nets = [0] * len(self.generation)



def evolution_controller(number_of_iterations):
    evolution_obj = Evolution()

    # x_data_point = []
    # y_data_point = []
    #
    # style.use('fivethirtyeight')
    #
    # for i in range(number_of_iterations):
    #     plt.clf()
    #
    #     print('Generation: ' + str(i))
    #     print('Generation Info: ')
    #     print('number_of_genomes: ' + str(len(evolution_obj.child_generation)))
    #     print('average score of generation: ' + str(
    #         (np.sum(evolution_obj.scores_nets) / len(evolution_obj.child_generation))))
    #     print('max score in generation: ' + str(np.max(evolution_obj.scores_nets)))
    #     print('Min score in generation: ' + str(np.min(evolution_obj.scores_nets)))
    #
    #     x_data_point.append(i)
    #     y_data_point.append((np.sum(evolution_obj.scores_nets) / len(evolution_obj.child_generation)))
    #
    #     # if np.max(evolution_obj.scores_nets) == data_points_for_each_group:
    #     #     print("came here")
    #
    #
    #     evolution_obj.start_evolution(child_gen_old)
    #     child_gen_new = evolution_obj.child_generation
    #     child_gen_old = child_gen_new
    #
    # plt.plot(x_data_point, y_data_point)
    # plt.show()

evolution_controller(100)