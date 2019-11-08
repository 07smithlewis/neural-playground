import numpy as np
import tensorflow.compat.v1 as tf
import copy


class Genome:

    dt = np.int16

    def __init__(self, basic_structure):
        structure = [*basic_structure, 0]

        self.nodes = [
            structure,                                                         # Structure[ In, Out, Hidden ]
            np.arange(0, structure[0] + structure[1], dtype=self.dt),     # Nodes
        ]

        self.connections = [
            0,                                      # number of connections
            np.empty(shape=(0, 4), dtype=self.dt),  # connections[innovation number, in node, out node, active?]
            np.empty(shape=0, dtype=np.float32)     # weights
        ]

    # Creates a Tensorflow Graph that can be used to calculate the output nodes of the network
    def calculate_output(self, input_nodes):
        nodes = []
        node_num_map = {}   # returns node position in 'nodes' given innovation number

        # Adding the input nodes to the nodes list
        for i in range(self.nodes[0][0]):
            nodes.append(input_nodes[i])
            node_num_map[self.nodes[1][i]] = i

        calculated = [False] * self.nodes[0][2]

        # Keeps track of the nodes that have been calculated
        available_nodes = set(self.nodes[1][:self.nodes[0][0]])

        while False in calculated:
            for i in range(self.nodes[0][2]):
                node = self.nodes[1][sum(self.nodes[0][0:2]) + i]

                # If node hasn't been calculated and all nodes required to calculate it are available:
                if not calculated[i] and self.nodes_required_to_calculate(node).issubset(available_nodes):

                    calculated[i] = True
                    available_nodes.add(node)

                    selector = np.logical_and(self.connections[1][:, 2] == node, self.connections[1][:, 3] == 1)
                    connected_nodes = [self.connections[1][selector, 1], self.connections[2][selector]]
                    if connected_nodes[0].size != 0:
                        connected_nodes[0] = np.vectorize(node_num_map.get)(connected_nodes[0])

                    nodes.append(tf.math.tanh(tf.reduce_sum(
                        tf.multiply(tf.constant(connected_nodes[1], dtype=tf.float32),
                                    tf.stack([nodes[j] for j in connected_nodes[0]])))))

                    node_num_map[node] = len(node_num_map)

        for i in range(self.nodes[0][1]):
            node = self.nodes[1][self.nodes[0][0] + i]

            selector = np.logical_and(self.connections[1][:, 2] == node, self.connections[1][:, 3] == 1)
            connected_nodes = [self.connections[1][selector, 1], self.connections[2][selector]]
            if connected_nodes[0].size != 0:
                connected_nodes[0] = np.vectorize(node_num_map.get)(connected_nodes[0])

            nodes.append(tf.math.tanh(tf.reduce_sum(
                tf.multiply(tf.constant(connected_nodes[1], dtype=tf.float32),
                            tf.stack([nodes[j] for j in connected_nodes[0]])))))

        return tf.stack(nodes[self.nodes[0][0] + self.nodes[0][2]:])
        # initializer is returned as the node variables must be initiated in order

    # Returns a set of nodes which 'node' is allowed to make a connection to
    def calculate_available_connections(self, node):
        # All non input nodes (Input nodes cannot be output for a connection)
        available_connections = set(self.nodes[1][self.nodes[0][0]:])

        if self.connections[0] != 0:

            # Exclude all nodes required in the calculation of 'node'
            # (Connections to these would result in infinite loop)
            available_connections.difference_update(self.nodes_required_to_calculate(node))
            available_connections.discard(node)

            # Exclude all nodes 'node' is already connected to
            # (Doubling up on connections is pointless as the weight can change)
            previously_connected_nodes = set(self.connections[1][self.connections[1][:, 1] == node, 2])
            available_connections.difference_update(previously_connected_nodes)

        return available_connections

    # Returns the set of nodes required to calculate 'node'
    def nodes_required_to_calculate(self, node):
        cascade = {node}
        while True:
            # Add any nodes directly connected to the nodes in the 'cascade' set to the 'cascade' set
            node_dependencies = set(self.connections[1][np.isin(self.connections[1][:, 2], list(cascade)), 1])
            node_dependencies.difference_update(cascade)
            if len(node_dependencies) == 0:
                break
            cascade.update(node_dependencies)
        cascade.remove(node)
        return cascade

    # Creates a description of the network structure to be decoded by Graphics.py
    def visualiser(self, draw_size, draw_location=[0, 0], weight_maximum=1):

        border = [np.min(draw_size) / 10.] * 2
        node_size = 6
        hidden_offset = draw_size[0] / 10.
        hidden_horizontal_locations = 1

        node_list = []
        connection_list = []
        object_list = []
        node_locations = []

        for i in range(self.nodes[0][0]):
            if self.nodes[0][0] == 1:
                location = [draw_size[0] / 2., border[1]]
            else:
                location = [border[0] + i * (draw_size[0] - border[0] * 2) / (self.nodes[0][0] - 1.),
                            border[1]]
            node_locations.append(location)
            node_list.append(['Rectangle', *list(np.add(location, draw_location)), 0, [node_size, node_size]])

        for i in range(self.nodes[0][1]):
            if self.nodes[0][1] == 1:
                location = [draw_size[0] / 2., draw_size[1] - border[1]]
            else:
                location = [border[0] + i * (draw_size[0] - border[0] * 2) / (self.nodes[0][1] - 1.),
                            draw_size[1] - border[1]]
            node_locations.append(location)
            node_list.append(['Rectangle', *list(np.add(location, draw_location)), 0, [node_size, node_size]])

        layer = 0
        layering = [-1] * self.nodes[0][2]
        available_nodes = set(self.nodes[1][:self.nodes[0][0]])
        while -1 in layering:
            new_nodes = set()
            for i in range(self.nodes[0][2]):
                node = self.nodes[1][sum(self.nodes[0][0:2]) + i]
                if (layering[i] == -1) and self.nodes_required_to_calculate(node).issubset(available_nodes):
                    layering[i] = layer
                    new_nodes.add(node)
            available_nodes.update(new_nodes)
            layer += 1

        layer_lengths = []
        for i in range(layer):
            layer_lengths.append(layering.count(i))

        count = [0] * layer

        max_layer = layer

        location_available = [True] * hidden_horizontal_locations * self.nodes[0][2]

        for i in range(self.nodes[0][2]):
            layer = layering[i]
            closest_location = [1., 0]
            for j in range(len(location_available)):
                if location_available[j]:

                    if layer_lengths[layer] == 1:
                        distance = abs((count[layer] + 0.5) / layer_lengths[layer] -
                                       j / (len(location_available) - 0.99))
                    else:
                        distance = abs((count[layer]) / (layer_lengths[layer] - 1.) -
                                       j / (len(location_available) - 0.99))

                    if distance < closest_location[0]:
                        closest_location = [distance, j]
            location_available[closest_location[1]] = False

            location = [border[0] + hidden_offset + closest_location[1] *
                        (draw_size[0] - border[0] * 2 - hidden_offset * 2) / (len(location_available) - 0.99),
                        border[1] + (layer + 1) * (draw_size[1] - border[1] * 2) / (max_layer + 1)]
            count[layer] += 1
            node_locations.append(location)
            node_list.append(['Rectangle', *list(np.add(location, draw_location)), 0, [node_size, node_size]])

        for i in range(self.connections[0]):
            if self.connections[1][i, 3] == 1:
                location1 = node_locations[np.where(self.nodes[1] == self.connections[1][i, 1])[0][0]]
                location2 = node_locations[np.where(self.nodes[1] == self.connections[1][i, 2])[0][0]]
                if self.connections[2][i] >= 0:
                    color = tuple([0, 0, 0, int(np.tanh(self.connections[2][i] * 4 / weight_maximum) * 255)])
                else:
                    color = tuple([255, 0, 0, int(np.tanh(-self.connections[2][i] * 4 / weight_maximum) * 255)])

                connection_list.append(['Line', *list(np.add(location1, draw_location)),
                                        *list(np.add(location2, draw_location)), 2, color])

        object_list.extend(connection_list)
        object_list.extend(node_list)
        return object_list

    # Orders the connections by their innovation number
    def order_connections(self):
        sort_order = np.argsort(self.connections[1][:, 0])
        self.connections[1] = self.connections[1][sort_order, :]
        self.connections[2] = self.connections[2][sort_order]

    # Returns a numerical measure of difference between this genome and another
    # (Genome must be ordered before calling this)
    def compare(self, genome):
        parameters = [1., 1., 1.]
        disjoint = 0
        weight_difference = 0

        index = [0, 0]
        connections = [self.connections, genome.connections]
        while index[0] < connections[0][0] and index[1] < connections[1][0]:
            if connections[0][1][index[0], 0] < connections[1][1][index[1], 0]:
                index[0] += 1
                disjoint += 1
            elif connections[0][1][index[0], 0] > connections[1][1][index[1], 0]:
                index[1] += 1
                disjoint += 1
            else:
                weight_difference += abs(connections[0][2][index[0]] - connections[1][2][index[1]])
                for i in range(2):
                    index[i] += 1

        excess = connections[0][0] + connections[1][0] - sum(index)
        overlap = (connections[0][0] + connections[1][0] - disjoint - excess) / 2
        max_connections = max(connections[0][0], connections[1][0])

        diff = (parameters[0] * disjoint + parameters[1] * excess) / (max_connections + 1.) + parameters[
            2] * weight_difference / (overlap + 1.)

        return diff


class Population:

    # The 'distance' between genomes required to consider them different species
    max_species_diversity = 0.2

    def __init__(self, basic_structure, size, mutation_fraction=0.1, mutation_factor=0.1, structure_mutation_chance=0.2,
                 ratio_add_to_split=0.6):
        self.generation = 0
        self.basic_structure = basic_structure
        self.size = size
        self.species_structure = []
        self.members = []
        self.member_fitness = np.ones(size, dtype=np.float32)
        self.mutation_fraction = mutation_fraction  # The fraction of weights in the network that change each generation
        self.mutation_factor = mutation_factor                    # The maximum amount a weight can change when mutating
        self.structure_mutation_chance = structure_mutation_chance         # the probability new structure is added to a
                                                                           # network each generation
        self.ratio_add_to_split = ratio_add_to_split          # the ratio between that new structure being an additional
                                                              # connection, to an additional node

        for i in range(size):
            self.members.append(Genome(self.basic_structure))

        self.update_species_structure()

    # Finds the number of species, and the number of agents classified into each species
    def update_species_structure(self):
        species_groups = [[0]]

        for member in self.members:
            member.order_connections()

        for i in range(1, self.size):
            found_species = False

            for j in range(len(species_groups)):
                rand = np.random.randint(0, len(species_groups[j]))
                if self.members[i].compare(self.members[species_groups[j][rand]]) < Population.max_species_diversity:
                    species_groups[j].append(i)
                    found_species = True
                    break

            if not found_species:
                species_groups.append([i])

        members_order = []
        for group in species_groups:
            members_order.extend(group)

        self.species_structure = [len(group) for group in species_groups]

        self.members = [self.members[i] for i in members_order]

    # Create a new population by breeding together members of the current population
    def next_generation(self, history):

        self.generation += 1

        new_members = []

        average_fitness = np.sum(self.member_fitness) / self.size

        i = 0
        for j in self.species_structure:

            total_species_fitness = sum(self.member_fitness[i:i + j])

            # The new population of each species is determined by how well the species performed
            species_population = int(total_species_fitness / average_fitness)

            for _ in range(species_population):

                # Choose two members of the species ar random, weighted by their fitness
                index = [0] * 2
                for ind in range(2):
                    rand = np.random.random_sample() * total_species_fitness
                    for j_ in range(j):
                        if rand < self.member_fitness[i + j_]:
                            index[ind] = i + j_
                            break
                        rand -= self.member_fitness[i + j_]

                # Breed them together, and add the result to the new population
                new_members.append(Mutate.reproduce(self.members[index[0]], self.members[index[1]]))

            i += j

        for i in range(self.size - len(new_members)):
            new_members.append(Mutate.reproduce(self.members[np.random.randint(0, self.size)],
                                                self.members[np.random.randint(0, self.size)]))

        # Mutate the weights of the new population
        for member in new_members:
            Mutate.weight_mutation(member, int(np.ceil(member.connections[0] * self.mutation_fraction)),
                                   mutation_factor=self.mutation_factor)

            # Randomly add additional network structure to some members of the population
            if np.random.random_sample() < self.structure_mutation_chance:
                if np.random.random_sample() < self.ratio_add_to_split:
                    Mutate.add_random_connection(member, history)
                else:
                    Mutate.split_connection(member, history)

        self.members = new_members

        self.update_species_structure()


class Mutate:
    # A number of functions to edit the structure of a network

    # Add a connection at a random location in the network
    @staticmethod
    def add_random_connection(genome, history):

        # Finds all places in the network where a connection can be added

        available_connections = []
        for i in range(genome.nodes[0][0] + genome.nodes[0][2]):
            if i >= genome.nodes[0][0]:
                node1 = genome.nodes[1][i + genome.nodes[0][1]]
            else:
                node1 = genome.nodes[1][i]
            node2_list = list(genome.calculate_available_connections(node1))
            for node2 in node2_list:
                available_connections.append([node1, node2])

        # Add a random connection from the set of all possible connections

        if len(available_connections) != 0:
            connection = available_connections[np.random.randint(len(available_connections))]

            Mutate.add_connection(genome, history, *connection)

    # Add a connection between two specific nodes in the network
    @staticmethod
    def add_connection(genome, history, node1, node2, weight=0):
        if not History.is_in(node1, node2, history.connection_history):
            history.append_connection_history(node1, node2)
        genome.connections[1] = np.append(genome.connections[1],
                                          [[history.connection_history[node1][node2], node1, node2, 1]], axis=0)
        genome.connections[2] = np.append(genome.connections[2], weight)
        genome.connections[0] += 1

    # Add a node along a random connection in the network, splitting it into two connections
    @staticmethod
    def split_connection(genome, history):

        if genome.connections[0] != 0:

            # Select a random connection that hasn't previous;y been split
            rand = np.random.randint(sum(genome.connections[1][:, 3]))
            for i in range(genome.connections[0]):
                if genome.connections[1][i, 3] == 1:
                    if rand == 0:

                        connection = genome.connections[1][i, 1:3]

                        # Disable the connection
                        genome.connections[1][i, 3] = 0

                        # Add a node between the two nodes
                        if not History.is_in(*connection, history.node_history):
                            history.append_node_history(*connection)
                        new_node = history.node_history[connection[0]][connection[1]]
                        genome.nodes[1] = np.append(genome.nodes[1], new_node)
                        genome.nodes[0][2] += 1

                        # Add the two replacement connections
                        if genome.connections[2][i] <= 0:
                            Mutate.add_connection(genome, history, connection[0], new_node,
                                                  weight=-np.sqrt(-genome.connections[2][i]))
                            Mutate.add_connection(genome, history, new_node, connection[1],
                                                  weight=np.sqrt(-genome.connections[2][i]))
                        else:
                            Mutate.add_connection(genome, history, connection[0], new_node,
                                                  weight=np.sqrt(genome.connections[2][i]))
                            Mutate.add_connection(genome, history, new_node, connection[1],
                                                  weight=np.sqrt(genome.connections[2][i]))
                        break
                    rand -= 1

    # Combine together two parent networks to create a child network that inherits structure from both
    @staticmethod
    def reproduce(genome1, genome2):

        genome3 = copy.deepcopy(genome1)

        additional_nodes = np.setdiff1d(genome2.nodes[1], genome1.nodes[1])
        genome3.nodes[1] = np.append(genome3.nodes[1], additional_nodes)
        genome3.nodes[0][2] += additional_nodes.size

        for i in range(genome2.connections[0]):
            if genome2.connections[1][i, 0] in genome1.connections[1][:, 0]:
                if genome2.connections[1][i, 3] == 0:
                    genome3.connections[1][np.where(genome3.connections[1][:, 0]
                                                    == genome2.connections[1][i, 0])[0][0], 3] = 0
            else:
                if not genome2.connections[1][i, 2] in genome1.nodes_required_to_calculate(
                        genome2.connections[1][i, 1]):
                    genome3.connections[0] += 1
                    genome3.connections[1] = np.append(genome3.connections[1], [genome2.connections[1][i, :]], axis=0)
                    genome3.connections[2] = np.append(genome3.connections[2], genome2.connections[2][i])

        return genome3

    # Change the weights of n connections in the network by a small amount
    @staticmethod
    def weight_mutation(genome, n=1, mutation_factor=0.1, weight_max=1):
        if genome.connections[0] != 0:
            prob = float(n) / genome.connections[0]
            if genome.connections[0] != 0:
                for i in range(genome.connections[0]):
                    if genome.connections[1][i, 3] == 1 and np.random.random_sample() < prob:
                        genome.connections[2][i] = np.clip(genome.connections[2][i] + mutation_factor * 2 *
                                                           (np.random.random_sample() - 0.5), -weight_max, weight_max)


class History:

    # The dictionaries in this class keep track of the innovation numbers assigned to each change in structure. This
    # allows for structural comparison between networks.

    def __init__(self, initial_innovation_number):
        self.innovation_numbers = [initial_innovation_number, 0]    # [nodes, connections]
        self.node_history = {}
        self.connection_history = {}

    def append_node_history(self, node1, node2):
        self.innovation_numbers[0] += 1
        History.append(node1, node2, self.innovation_numbers[0], self.node_history)

    def append_connection_history(self, node1, node2):
        self.innovation_numbers[1] += 1
        History.append(node1, node2, self.innovation_numbers[1], self.connection_history)

    @staticmethod
    def is_in(x, y, hist):
        is_in = False
        if x in hist:
            if y in hist[x]:
                is_in = True
        return is_in

    @staticmethod
    def append(x, y, value, hist):
        if x in hist:
            hist[x][y] = value
        else:
            hist[x] = {y: value}
