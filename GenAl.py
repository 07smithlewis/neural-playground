import numpy as np
import tensorflow.compat.v1 as tf
import copy

data_type = np.float32


class Genome:

    def __init__(self, structure):

        self.structure = structure
        self.number_of_connections = np.sum(np.multiply(structure[1:], structure[:len(structure) - 1]))

        self.connections = []
        for i in range(len(structure) - 1):
            self.connections.append(np.empty((structure[i + 1], structure[i]), dtype=data_type))

    def randomise_connections(self):

        for i in range(len(self.connections)):
            self.connections[i] = np.random.random_sample(self.connections[i].shape) * 2 - 1

    def calculate_output(self, input_nodes):

        connections = []
        layers = [tf.reshape(input_nodes, [self.structure[0], 1])]
        for i in range(len(self.connections)):
            connections.append(tf.constant(self.connections[i], dtype=data_type))
            if i == len(self.connections) - 1:
                layers.append(tf.math.tanh(tf.matmul(connections[i], layers[i])))
            else:
                layers.append(tf.sigmoid(tf.matmul(connections[i], layers[i])))

        return tf.reshape(layers[len(layers) - 1], [-1])

    def compare(self, genome):

        weight_difference = 0

        for i in range(len(self.connections)):
            weight_difference += np.sum(np.abs(self.connections[i] - genome.connections[i]))

        return weight_difference / self.number_of_connections

    def weight_mutation(self, mutation_fraction, mutation_factor):

        if mutation_fraction == 1:

            for i in range(len(self.connections)):
                self.connections[i] += (np.random.random_sample(self.connections[i].shape) - 0.5) * mutation_factor

        else:

            for i in range(len(self.connections)):
                self.connections[i] += np.multiply(
                    np.random.choice(2, p=[1 - mutation_fraction, mutation_fraction], size=self.connections[i].shape),
                    (np.random.random_sample(self.connections[i].shape) - 0.5) * mutation_factor
                )

    @staticmethod
    def reproduce(genome1, genome2, fitness1, fitness2):

        genome3 = copy.deepcopy(genome1)

        for i in range(len(genome1.connections)):
            genome3.connections[i] += np.multiply(np.random.choice(
                2, p=[fitness1/(fitness1 + fitness2), 1 - fitness1/(fitness1 + fitness2)],
                size=genome1.connections[i].shape),
                genome2.connections[i] - genome1.connections[i])

        return genome3

    def visualiser(self, draw_size, draw_location=[0, 0], weight_maximum=1):

        border = [np.min(draw_size) / 10.] * 2
        node_size = 6
        hidden_offset = draw_size[0] / 10.

        node_list = []
        connection_list = []
        object_list = []
        node_locations = []

        node_locations.append([])
        for i in range(self.structure[0]):
            if self.structure[0] == 1:
                location = [draw_size[0] / 2., border[1]]
            else:
                location = [border[0] + i * (draw_size[0] - border[0] * 2) / (self.structure[0] - 1.),
                            border[1]]
            node_locations[0].append(location)
            node_list.append(['Rectangle', *list(np.add(location, draw_location)), 0, [node_size, node_size]])

        for i in range(len(self.structure[1:-1])):
            node_locations.append([])
            for j in range(self.structure[i + 1]):

                if self.structure[i + 1] != 1:
                    location = [border[0] + hidden_offset + j *
                                (draw_size[0] - border[0] * 2 - hidden_offset * 2) / (self.structure[i + 1] - 1),
                                border[1] + (i + 1) * (draw_size[1] - border[1] * 2) / (len(self.structure[1:-1]) + 1)]
                else:
                    location = [border[0] + draw_size[0] / 2,
                                border[1] + (i + 1) * (draw_size[1] - border[1] * 2) / (len(self.structure[1:-1]) + 1)]

                node_locations[i + 1].append(location)
                node_list.append(['Rectangle', *list(np.add(location, draw_location)), 0, [node_size, node_size]])

        node_locations.append([])
        for i in range(self.structure[-1]):
            if self.structure[-1] == 1:
                location = [draw_size[0] / 2., draw_size[1] - border[1]]
            else:
                location = [border[0] + i * (draw_size[0] - border[0] * 2) / (self.structure[-1] - 1.),
                            draw_size[1] - border[1]]
            node_locations[-1].append(location)
            node_list.append(['Rectangle', *list(np.add(location, draw_location)), 0, [node_size, node_size]])

        for i in range(len(self.connections)):
            for j in range(self.connections[i].shape[0]):
                for k in range(self.connections[i].shape[1]):

                    location1 = node_locations[i][k]
                    location2 = node_locations[i + 1][j]
                    if self.connections[i][j, k] >= 0:
                        color = tuple([0, 0, 0, int(self.connections[i][j, k] * 255)])
                    else:
                        color = tuple([255, 0, 0, -int(self.connections[i][j, k] * 255)])

                    connection_list.append(['Line', *list(np.add(location1, draw_location)),
                                            *list(np.add(location2, draw_location)), 2, color])

        object_list.extend(connection_list)
        object_list.extend(node_list)
        return object_list


class Population:

    def __init__(self, structure, size, max_species_diversity=0.01, mutation_fraction=1, mutation_factor=0.01):

        self.structure = structure
        self.size = size
        self.max_species_diversity = max_species_diversity
        self.mutation_fraction = mutation_fraction  # The fraction of weights in the network that change each generation
        self.mutation_factor = mutation_factor  # The maximum amount a weight can change when mutating

        self.generation = 0
        self.species_structure = []
        self.member_fitness = np.ones(size, dtype=data_type)

        self.members = []
        for i in range(size):
            self.members.append(Genome(structure))
            self.members[i].randomise_connections()

    def update_species_structure(self):
        species_groups = [[0]]

        for i in range(1, self.size):
            found_species = False

            for j in range(len(species_groups)):
                rand = np.random.randint(0, len(species_groups[j]))
                if self.members[i].compare(self.members[species_groups[j][rand]]) < self.max_species_diversity:
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

    def next_generation(self):

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
                new_members.append(Genome.reproduce(self.members[index[0]], self.members[index[1]],
                                                    self.member_fitness[index[0]], self.member_fitness[index[1]]))

            i += j

        for i in range(self.size - len(new_members)):
            genome = (Genome(self.structure))
            genome.randomise_connections()
            new_members.append(genome)

        # Mutate the weights of the new population
        for member in new_members:
            member.weight_mutation(self.mutation_fraction, self.mutation_factor)

        self.members = new_members

        self.update_species_structure()
