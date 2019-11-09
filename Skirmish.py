import pyglet
import Neat
import Physics
import numpy as np
import tensorflow.compat.v1 as tf
import Graphics
import time
import pickle
import GenAl


class Simulation:
    def __init__(self, population_sizes, window_dimensions, num_photoreceptors=3, visual_acuity=8, velocity_decay=0.2,
                 dt=0.1, bounds='loop', maximum_acceleration=200., rotation_multiplier=4., strafe_multiplier=0.5,
                 energy_regeneration=0.02, acceleration_energy_use=2., attack_energy_use=2., attack_range=50,
                 attack_angle=np.pi / 8., friendly_fire_multiplier=1., show_vision=False, vision_draw_length=100.,
                 took_dmg_from_friend=-0.5, damaged_friend=-1., took_dmg_from_enemy=-1., damaged_enemy=2., died=-1.,
                 killed_friend=-0.25, killed_enemy=0.5, frame_rate=20, graphics=True, max_score_history=50,
                 max_damage=0.1, mutation_fraction=0.1, mutation_factor=0.1, structure_mutation_chance=0.2,
                 ratio_add_to_split=0.6, vision_range=200, use_neat=True, hidden_layers=[10, 10, 10, 10],
                 draw_network=True):

        self.population_sizes = population_sizes
        self.window_dimensions = window_dimensions

        self.num_photoreceptors = num_photoreceptors       # The number of angle regions the agents vision is split into
        self.visual_acuity = visual_acuity             # How concentrated the regions are about the centre of its vision
        self.velocity_decay = velocity_decay          # each second the velocity is multiplied by this (simple friction)
        self.dt = dt                    # The simulation time that passes for each iteration of the main simulation loop
        self.bounds = bounds       # Type of simulation boundary used: 'loop' = space is looped. 'wall' = solid boundary
        self.maximum_acceleration = maximum_acceleration                     # Maximum forward acceleration of the agent
        self.rotation_multiplier = rotation_multiplier                     # Max torque with respect to max acceleration
        self.strafe_multiplier = strafe_multiplier            # Max strafe acceleration with respect to max acceleration
        self.energy_regeneration = energy_regeneration                              # Amount of energy gained per second
        self.acceleration_energy_use = acceleration_energy_use   # Energy used per second at max acceleration wrt energy
        # regeneration
        self.attack_energy_use = attack_energy_use     # Energy used per second when attacking at full energy wrt energy
        # regeneration
        self.attack_range = attack_range                                           # Range at which attacks cause damage
        self.attack_angle = attack_angle                                   # Maximum angle at which attacks cause damage
        self.max_damage = max_damage                                   # Damage per second when attacking at full energy
        self.friendly_fire_multiplier = friendly_fire_multiplier          # Damage to own team wrt damage to other teams
        self.frame_rate = frame_rate            # Target framerate for rendering (Actual framerate will stay below this)
        self.died = died                                                                    # Score value of agent death
        self.killed_friend = killed_friend                        # Score value of agent killing member of it's own team
        self.killed_enemy = killed_enemy                           # Score value of agent killing member of another team
        self.took_dmg_from_friend = took_dmg_from_friend   # Score value per damage point of taking damage from own team
        self.damaged_friend = damaged_friend                        # Score value per damage point of attacking own team
        self.took_dmg_from_enemy = took_dmg_from_enemy       # Score value per dmg point of taking dmg from another team
        self.damaged_enemy = damaged_enemy                      # Score value per damage point of attacking another team
        self.max_score_history = np.zeros(max_score_history, dtype=np.float32)  # Number of generations plotted on score
        # history graph
        self.graphics = graphics                                       # True: Draw graphics, False: Don't draw graphics
        self.show_vision = show_vision                                   # Draw lines showing agent vision angle regions
        self.vision_draw_length = vision_draw_length                                            # Length of vision lines
        self.mutation_fraction = mutation_fraction  # The fraction of weights in the network that change each generation
        self.mutation_factor = mutation_factor                    # The maximum amount a weight can change when mutating
        self.structure_mutation_chance = structure_mutation_chance         # the probability new structure is added to a
        # network each generation
        self.ratio_add_to_split = ratio_add_to_split          # the ratio between that new structure being an additional
        # connection, to an additional node
        self.vision_range = vision_range                     # The maximum range at which the agent can see other agents
        self.use_neat = use_neat         # True: Use NEAT,  False: Use genetic evolution of feed forward network weights
        self.hidden_layers = hidden_layers                      # Lengths of hidden layers if using feed forward network
        self.draw_network = draw_network                              # True: Show the network,  False: Hide the network

        self.population_total = sum(population_sizes)
        self.velocity_decay_dt = (1 - (1 - velocity_decay) * dt)
        self.max_velocity = self.maximum_acceleration / (1 - velocity_decay)

        wd = window_dimensions
        params = [0.3, 0.2, 0.6]
        self.layout = [[[params[0] * wd[0], 0], [(1 - params[0]) * wd[0], wd[1]]],
                       [[0, params[1] * wd[1]], [params[0] * wd[0], params[2] * wd[1]]],
                       [[0, 0], [params[0] * wd[0], params[1] * wd[1]]],
                       [[0, (params[1] + params[2]) * wd[1]], [params[0] * wd[0], (1 - params[1] - params[2]) * wd[1]]]]

        self.vision_bins = np.arange(-self.num_photoreceptors / 2., self.num_photoreceptors / 2. + 1)
        self.vision_bins = np.exp(self.vision_bins * self.visual_acuity / self.num_photoreceptors) - np.exp(
            -self.vision_bins * self.visual_acuity / self.num_photoreceptors)
        self.vision_bins *= np.pi / 2. / (np.exp(self.visual_acuity / 2.) - np.exp(-self.visual_acuity / 2.))

        inputs = num_photoreceptors * 2 + 3 + 2 + 1    # [Vision, Velocity, Agent Stats, Bias]
        outputs = 3 + 1                             # [Acceleration, Agent Actions]
        self.basic_structure = [inputs, outputs]

        if self.use_neat:
            self.population = Neat.Population(
                self.basic_structure, sum(self.population_sizes), mutation_fraction=self.mutation_fraction,
                mutation_factor=self.mutation_factor, structure_mutation_chance=self.structure_mutation_chance,
                ratio_add_to_split=self.ratio_add_to_split)
            self.history = Neat.History(sum(self.basic_structure) - 1)
        else:
            self.population = GenAl.Population([self.basic_structure[0], *self.hidden_layers, self.basic_structure[1]],
                                               sum(self.population_sizes), mutation_fraction=self.mutation_fraction,
                                               mutation_factor=self.mutation_factor)

        self.add_extra_member_info()

        self.object_list = []
        self.fill_object_list()

    def add_extra_member_info(self):
        for member in self.population.members:
            member.phys = Physics.Phys(displacement=np.concatenate((np.random.random_sample(2) * self.layout[0][1],
                                                                   np.array([np.random.random_sample() * 360]))))
            member.stats = np.ones(3, dtype=np.float32)     # [Health, Energy, attacking?]
            member.input = np.zeros(self.basic_structure[0], dtype=np.float32)
            member.rotation_mat = np.array([[1, 0], [0, 1]], dtype=np.float32)
            member.score = 0.

    # Calculate the network input of each member of the population
    def calculate_input(self):

        vision_range_squared = np.power(self.vision_range, 2)
        coordinates = np.empty((2, sum(self.population_sizes)), dtype=np.float32)

        for i in range(self.population_total):
            coordinates[:, i] = self.population.members[i].phys.vars[0:2]

        for i in range(len(self.population_sizes)):

            friendly_coordinates = np.copy(coordinates[:, sum(
                self.population_sizes[:i]):sum(self.population_sizes[:i + 1])])
            unfriendly_coordinates = np.copy(np.concatenate((coordinates[:, :sum(
                self.population_sizes[:i])], coordinates[:, sum(self.population_sizes[:i + 1]):]), axis=1))

            for j in range(self.population_sizes[i]):
                k = sum(self.population_sizes[:i]) + j

                c = np.cos(self.population.members[k].phys.vars[2] * np.pi / 180.)
                s = np.sin(self.population.members[k].phys.vars[2] * np.pi / 180.)
                self.population.members[k].rotation_mat = np.array([[c, -s], [s, c]])

                friendly_coordinates_ = np.matmul(self.population.members[k].rotation_mat,
                                                  friendly_coordinates - coordinates[:, k][:, None])
                unfriendly_coordinates_ = np.matmul(self.population.members[k].rotation_mat,
                                                    unfriendly_coordinates - coordinates[:, k][:, None])

                friendly_coordinates_ = friendly_coordinates_[
                                        :, np.add(np.power(friendly_coordinates_[0, :], 2),
                                                  np.power(friendly_coordinates_[1, :],
                                                           2)) < vision_range_squared]

                unfriendly_coordinates_ = unfriendly_coordinates_[
                                          :, np.add(np.power(unfriendly_coordinates_[0, :], 2),
                                                    np.power(unfriendly_coordinates_[1, :],
                                                             2)) < vision_range_squared]

                friendly_photoreceptors = np.angle(1j * friendly_coordinates_[0, :] + friendly_coordinates_[1, :])
                unfriendly_photoreceptors = np.angle(1j * unfriendly_coordinates_[0, :] + unfriendly_coordinates_[1, :])

                friendly_photoreceptors = np.histogram(friendly_photoreceptors, bins=self.vision_bins)[0]
                friendly_photoreceptors[int(self.num_photoreceptors/2)] -= 1
                friendly_photoreceptors = np.tanh(friendly_photoreceptors)
                unfriendly_photoreceptors = np.tanh(np.histogram(unfriendly_photoreceptors, bins=self.vision_bins)[0])

                normalised_velocity = np.divide(np.concatenate((np.matmul(
                    self.population.members[k].rotation_mat, self.population.members[k].phys.vars[3:5]
                ), self.population.members[k].phys.vars[5:6])),
                    np.array([self.strafe_multiplier, 1., self.rotation_multiplier]) * self.max_velocity / 10)

                self.population.members[k].input = np.concatenate((
                    friendly_photoreceptors, unfriendly_photoreceptors,
                    normalised_velocity, self.population.members[k].stats[:2], np.ones(1)))

    # Create the list of objects needed by Graphics.py, to draw the population
    def fill_object_list(self):

        color_dict = {0: (50, 50, 50, 255), 1: (200, 200, 200, 255), 2: (100, 100, 0, 255), 3: (0, 200, 0, 255),
                      4: (0, 0, 200, 255), 5: (100, 0, 100, 255), 6: (0, 100, 100, 255)}

        for i in range(len(self.population_sizes)):
            for j in range(self.population_sizes[i]):
                if self.show_vision:
                    self.object_list.append(['Fob With Vision', self.vision_bins, np.zeros(self.num_photoreceptors), 0,
                                             0, 0, color_dict[i], 0, 0, self.vision_draw_length])
                else:
                    self.object_list.append(['Fob', 0, 0, 0, color_dict[i], 0, 0])

    # Update the list of objects with current positions, and agent stats
    def update_object_list(self, offset=[0, 0]):
        offset_ = [0, 0, 0]
        offset_[:2] = offset

        if self.show_vision:
            for i in range(self.population_total):
                self.object_list[i][2] = np.add(self.population.members[i].input[:self.num_photoreceptors],
                                                self.population.members[i].input[
                                                self.num_photoreceptors:2 * self.num_photoreceptors])
                self.object_list[i][3:6] = np.add(self.population.members[i].phys.vars[:3], offset_)
                self.object_list[i][7:9] = self.population.members[i].stats[:2]
        else:
            for i in range(self.population_total):
                self.object_list[i][1:4] = np.add(self.population.members[i].phys.vars[:3], offset_)
                self.object_list[i][5:7] = self.population.members[i].stats[:2]

    # Use the network outputs to update the agent's acceleration, stats, and energy
    def update_member(self, member, network_output):

        member.stats[1] += self.energy_regeneration * self.dt

        energy_function = np.tanh(4 * member.stats[1])
        a_multiplier = np.array(
            [self.strafe_multiplier, 1.], dtype=np.float32) * energy_function * self.maximum_acceleration * self.dt

        member.phys.vars[6:8] = np.matmul(member.rotation_mat.transpose(),
                                          np.multiply(network_output[:2], a_multiplier))
        member.phys.vars[8] = network_output[2] * energy_function * self.maximum_acceleration * \
            self.rotation_multiplier * self.dt
        if network_output[3] > 0:
            member.stats[2] = energy_function * self.max_damage * self.dt
        else:
            member.stats[2] = 0

        member.stats[1] = np.clip(member.stats[1] - self.energy_regeneration * self.dt * (
            energy_function * self.acceleration_energy_use * np.sqrt(network_output[:3].dot(
                network_output[:3])) + self.attack_energy_use * member.stats[2] / (self.max_damage * self.dt)), 0, 1)

    # Determine the agents that have taken damage, and update health and score accordingly
    def update_health(self):

        attack_object_list = []
        attack_range_squared = np.power(self.attack_range, 2)
        coordinates = np.empty((3, sum(self.population_sizes)), dtype=np.float32)

        for i in range(self.population_total):
            coordinates[:, i] = np.append(np.array([i]), self.population.members[i].phys.vars[0:2])

        for i in range(len(self.population_sizes)):

            friendly_coordinates = np.copy(coordinates[:, sum(
                self.population_sizes[:i]):sum(self.population_sizes[:i + 1])])
            unfriendly_coordinates = np.copy(np.concatenate((coordinates[:, :sum(
                self.population_sizes[:i])], coordinates[:, sum(self.population_sizes[:i + 1]):]), axis=1))

            for j in range(self.population_sizes[i]):
                k = sum(self.population_sizes[:i]) + j

                if self.population.members[k].stats[2] != 0:

                    attack_object_list.append(['Circle', *np.add(coordinates[1:, k], self.layout[0][0]), 4,
                                               (200, 0, 0, 255)])

                    friendly_coordinates_ = np.copy(friendly_coordinates)
                    unfriendly_coordinates_ = np.copy(unfriendly_coordinates)
                    friendly_coordinates_[1:, :] -= coordinates[:, k][1:, None]
                    unfriendly_coordinates_[1:, :] -= coordinates[:, k][1:, None]

                    friendly_coordinates_ = friendly_coordinates_[
                                            :, np.add(np.power(friendly_coordinates_[1, :], 2),
                                                      np.power(friendly_coordinates_[2, :],
                                                               2)) < attack_range_squared]

                    unfriendly_coordinates_ = unfriendly_coordinates_[
                                              :, np.add(np.power(unfriendly_coordinates_[1, :], 2),
                                                        np.power(unfriendly_coordinates_[2, :],
                                                                 2)) < attack_range_squared]

                    c = np.cos(self.population.members[k].phys.vars[2] * np.pi / 180.)
                    s = np.sin(self.population.members[k].phys.vars[2] * np.pi / 180.)
                    self.population.members[k].rotation_mat = np.array([[c, -s], [s, c]])

                    friendly_coordinates_[1:, :] = np.matmul(self.population.members[k].rotation_mat,
                                                             friendly_coordinates_[1:, :])
                    unfriendly_coordinates_[1:, :] = np.matmul(self.population.members[k].rotation_mat,
                                                               unfriendly_coordinates_[1:, :])

                    friendly_angle = np.abs(np.angle(1j * friendly_coordinates_[1, :] + friendly_coordinates_[2, :]))
                    unfriendly_angle = np.abs(np.angle(1j * unfriendly_coordinates_[1, :] +
                                                       unfriendly_coordinates_[2, :]))

                    for ind in friendly_coordinates_[0, friendly_angle < self.attack_angle]:
                        if int(ind) != k:
                            damage = self.friendly_fire_multiplier * self.population.members[k].stats[2]
                            self.population.members[int(ind)].stats[0] = np.clip(
                                self.population.members[int(ind)].stats[0] - damage, 0, 1)
                            if self.population.members[int(ind)].stats[0] == 0:
                                self.population.members[int(ind)].score += self.died
                                self.population.members[k].score += self.killed_friend
                            self.population.members[int(ind)].score += damage * self.took_dmg_from_friend
                            self.population.members[k].score += damage * self.damaged_friend

                            attack_object_list.append(['Line', *np.add(coordinates[1:, k], self.layout[0][0]),
                                                       *np.add(coordinates[1:, int(ind)], self.layout[0][0]), 2,
                                                       (200, 0, 0, 255)])

                    for ind in unfriendly_coordinates_[0, unfriendly_angle < self.attack_angle]:
                        damage = self.population.members[k].stats[2]
                        self.population.members[int(ind)].stats[0] = np.clip(
                            self.population.members[int(ind)].stats[0] - damage, 0, 1)
                        if self.population.members[int(ind)].stats[0] == 0:
                            self.population.members[int(ind)].score += self.died
                            self.population.members[k].score += self.killed_enemy
                        self.population.members[int(ind)].score += damage * self.took_dmg_from_enemy
                        self.population.members[k].score += damage * self.damaged_enemy

                        attack_object_list.append(['Line', *np.add(coordinates[1:, k], self.layout[0][0]),
                                                   *np.add(coordinates[1:, int(ind)], self.layout[0][0]), 2,
                                                   (200, 0, 0, 255)])

        for member in self.population.members:
            if member.stats[0] == 0:
                member.phys = Physics.Phys(displacement=np.concatenate((np.random.random_sample(2) * self.layout[0][1],
                                                                        np.array([np.random.random_sample() * 360]))))
                member.stats = np.ones(3, dtype=np.float32)

        return attack_object_list

    # Create the object lists used by Graphics.py to draw the maximum score over generation graph
    def draw_max_score_history(self, draw_size, draw_location=[0, 0]):
        border = np.multiply(draw_size, [0.15, 0.1])

        object_list = [['Line', *np.add(draw_location, border).tolist(), draw_location[0] + draw_size[0] - border[0],
                        draw_location[1] + border[1], 2],
                       ['Line', *np.add(draw_location, border).tolist(), draw_location[0] + border[0],
                        draw_location[1] + draw_size[1] - border[1], 2]]

        maximum = np.max(self.max_score_history)
        graph_maximum = 0
        for i in range(100):
            graph_maximum = np.power(10, i + 1)
            if maximum < graph_maximum:
                graph_maximum = int(np.power(10, i) * (int(maximum / np.power(10, i)) + 1))
                break

        text_list = [['Text', 2 * (draw_location[0] + border[0] / 2), 2 * (draw_location[1] + border[1]),
                      '{}'.format(0)],
                     ['Text', 2 * (draw_location[0] + border[0] / 2), 2 * (draw_location[1] + draw_size[1] - border[1]),
                      '{}'.format(graph_maximum)]]

        for i in range(len(self.max_score_history) - 1):
            object_list.append(['Line', draw_location[0] + border[0] + (draw_size[0] - 2 * border[0]) * i / (
                        len(self.max_score_history) - 1),
                                draw_location[1] + border[1] + (draw_size[1] - 2 * border[1]) * self.max_score_history[
                                    i] / graph_maximum,
                                draw_location[0] + border[0] + (draw_size[0] - 2 * border[0]) * (i + 1) / (
                                            len(self.max_score_history) - 1),
                                draw_location[1] + border[1] + (draw_size[1] - 2 * border[1]) * self.max_score_history[
                                    i + 1] / graph_maximum, 1, (0, 0, 0, 200)])

        return object_list, text_list

    # Run the simulation
    def run(self, run_time, environment):

        # Create the TensorFlow computation graphs
        network_graph = tf.Graph()
        last_frame_time = time.time()

        # Create networks computation graph
        with network_graph.as_default():
            network_in = [tf.placeholder(tf.float32, [self.basic_structure[0]])] * self.population_total
            network_in_load = [np.empty(self.basic_structure[0], np.float32)] * self.population_total
            network_out = [None] * self.population_total

            for i in range(self.population_total):
                network_out[i] = self.population.members[i].calculate_output(network_in[i])

                # Loading bar
                if self.graphics and time.time() - last_frame_time > 1. / self.frame_rate:
                    last_frame_time = time.time()
                    object_lists = environment.object_lists
                    if environment.object_lists:
                        environment.window.dispatch_events()
                        environment.on_draw()
                    size = [200, 30, 4]
                    environment.object_lists = [
                        [['Rectangle', *np.multiply(self.window_dimensions, 0.5).tolist(), 0, self.window_dimensions,
                          (255, 255, 255, 150)],
                         ['Rectangle', *np.multiply(self.window_dimensions, 0.5).tolist(), 0, size[:2], (0, 0, 0, 255)],
                         ['Rectangle', *np.multiply(self.window_dimensions, 0.5).tolist(), 0,
                          [size[0] - 2 * size[2], size[1] - 2 * size[2]], (255, 255, 255, 255)],
                         ['Rectangle', *np.add(np.multiply(self.window_dimensions, 0.5),
                          [(size[0] - 2 * size[2]) * 0.5 * ((i + 1.) / self.population_total - 1), 0]).tolist(), 0,
                          [(size[0] - 2 * size[2]) * ((i + 1.) / self.population_total),
                          size[1] - 2 * size[2]], (100, 100, 100, 255)]],
                        [['Text', self.window_dimensions[0], self.window_dimensions[1] + size[1] * 2,
                          'Creating TensorFlow computation graph'],
                         ['Text', *self.window_dimensions,
                          '%.0f' % ((i + 1.) / self.population_total * 100) + '%']]
                    ]
                    if not object_lists:
                        environment.window.dispatch_events()
                    environment.on_draw()
                    environment.window.flip()
                    environment.object_lists = object_lists

        # Create physics computation graph
        physics_graph = tf.Graph()
        with physics_graph.as_default():
            phys_in = tf.placeholder(tf.float32, [self.population_total, 9])
            phys_in_load = np.empty((self.population_total, 9), dtype=np.float32)
            phys_out = Physics.Phys.calculate(phys_in, self.dt, self.layout[0][1], self.velocity_decay_dt,
                                              self.bounds)

        network_session = tf.Session(graph=network_graph)
        phys_session = tf.Session(graph=physics_graph)

        # Graphics description for network visualisation

        visualiser_object_list = [self.population.members[i].visualiser(self.layout[1][1], self.layout[1][0])
                                  for i in range(self.population_total)]

        # Graphics description for layout separation borders
        lines = []
        x = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        for ly in self.layout:
            points = []
            for i in range(4):
                points.append(np.add(ly[0], np.multiply(x[i, :], ly[1])).tolist())
            for i in range(4):
                lines.append(points[i] + points[(i + 1) % 4])
        layout_object_list = [['Line', *np.add(lines[0], [-13, 0, -13, 0]).tolist(),
                               26, (255, 255, 255, 255)]]
        for line in lines:
            layout_object_list.append(['Line', *line, 8])

        # Graphics description for Text
        x = [self.window_dimensions[1], self.layout[3][0][1], self.layout[3][1][0]]
        text_object_list = []
        lines_of_text = 4
        for i in range(lines_of_text):
            text_object_list.append(['Text', x[2], 2 * (x[0] - (x[0] - x[1]) * (i + 1) / (lines_of_text + 1)), ''])
        text_object_list[0][3] = 'Generation {}'.format(self.population.generation)

        # Graphics description for score history graph
        graph_object_list, graph_text = self.draw_max_score_history(self.layout[2][1], self.layout[2][0])
        text_object_list.extend(graph_text)

        # Graphics description for clearing the parts of the screen that update
        clear = [[['Rectangle', *np.array(self.window_dimensions) * 0.5, 0, self.window_dimensions,
                   (255, 255, 255, 255)]],
                 [['Rectangle', *np.add(np.array(self.layout[0][1]) * 0.5, self.layout[0][0]), 0,
                   self.layout[0][1], (255, 255, 255, 255)],
                  ['Rectangle', *np.add(np.array(self.layout[3][1]) * 0.5, self.layout[3][0]), 0,
                   self.layout[3][1], (255, 255, 255, 255)]
                  ]]

        scores = np.empty(self.population_total, dtype=np.float32)
        last_frame_time = time.time()
        highest_scoring_agent = np.array([-1, -1])

        # Main simulation loop
        for i in range(int(run_time / self.dt)):

            self.calculate_input()
            for j in range(self.population_total):
                network_in_load[j] = self.population.members[j].input

            # Calculate agent networks
            network_out_load = network_session.run(network_out, feed_dict={
                a: b for a, b in zip(network_in, network_in_load)})

            for j in range(self.population_total):
                self.update_member(self.population.members[j], network_out_load[j])
                phys_in_load[j, :] = self.population.members[j].phys.vars

            # Calculate agent physics
            phys_out_load = phys_session.run(phys_out, feed_dict={phys_in: phys_in_load})

            for j in range(self.population_total):
                self.population.members[j].phys.vars = phys_out_load[j, :]

            attack_object_list = self.update_health()

            # Graphics loop (Run frequency determined by framerate)
            if self.graphics and time.time() - last_frame_time > 1. / self.frame_rate:

                last_frame_time = time.time()

                self.update_object_list(self.layout[0][0])

                # Fill in text with current information
                text_object_list[1][3] = 'Simulation time: {}h {}m {}s'.format(
                    int(i * self.dt / 3600), int(((i * self.dt) % 3600) / 60), int((i * self.dt) % 60))
                for j in range(self.population_total):
                    scores[j] = self.population.members[j].score
                text_object_list[2][3] = 'Highest Score: %.2f' % np.max(scores)
                text_object_list[3][3] = 'Num of species: {}'.format(len(self.population.species_structure))

                # Pass current graphics information to the environment object
                if np.argmax(scores) != highest_scoring_agent[0] and self.draw_network:
                    print(1)
                    highest_scoring_agent = np.roll(highest_scoring_agent, -1)
                    highest_scoring_agent[-1] = np.argmax(scores)

                    environment.object_lists = [clear[0], self.object_list, attack_object_list, layout_object_list,
                                                graph_object_list, visualiser_object_list[np.argmax(scores)],
                                                text_object_list]
                else:
                    print(2)
                    environment.object_lists = [clear[1], self.object_list, attack_object_list, layout_object_list,
                                                graph_object_list, text_object_list]

                pyglet.clock.tick()

                # Draw the frame
                environment.window.dispatch_events()
                environment.window.dispatch_event('on_draw')
                environment.window.flip()

        for i in range(self.population_total):
            self.population.member_fitness[i] = np.exp(self.population.members[i].score)

        self.max_score_history = np.roll(self.max_score_history, -1)
        self.max_score_history[len(self.max_score_history) - 1] = np.max(scores)

# =====----===== # =====----===== # =====----===== # =====----===== # =====----===== # =====----===== # =====----===== #
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
# =====----===== # =====----===== # =====----===== # =====----===== # =====----===== # =====----===== # =====----===== #
# Only make changes to the parameters in here


window_dimensions_ = [1000, 700]

# A list of agent team sizes e.g. [3, 3, 3, 3] would create 4 teams of 3 (Max 6 teams)
team_sizes = [15, 15]

# The time in seconds each generation is simulated for
run_time_ = 180

# Look at the definition of the Simulation class for a full list of optional arguments
sim = Simulation(team_sizes, window_dimensions_, use_neat=False)

# Uncomment this line if you want to continue from your last session, rather than starting a new one
# sim = pickle.load(open("save.p", "rb"))

# =====----===== # =====----===== # =====----===== # =====----===== # =====----===== # =====----===== # =====----===== #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# =====----===== # =====----===== # =====----===== # =====----===== # =====----===== # =====----===== # =====----===== #

env = Graphics.Environment(window_dimensions_, clear=False)

while True:
    pickle.dump(sim, open("save.p", "wb"))
    if sim.use_neat:
        sim.population.next_generation(sim.history)
    else:
        sim.population.next_generation()
    sim.add_extra_member_info()
    sim.run(run_time_, env)
