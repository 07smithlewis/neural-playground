import pyglet
import Neat
import Physics
import numpy as np
import tensorflow.compat.v1 as tf
import Graphics
import LoadingBar
import time
import pickle


class Simulation:
    def __init__(self, population_sizes, window_dimensions, num_photoreceptors=10, visual_acuity=3, velocity_decay=0.9,
                 dt=0.1, bounds='wall', maximum_acceleration=100., rotation_multiplier=2., max_damage=0.01,
                 energy_regeneration=0.001, acceleration_energy_use=2., attack_energy_use=4., attack_range=50,
                 attack_angle=np.pi / 6., friendly_fire_multiplier=1., show_vision=False, vision_draw_length=100.,
                 took_dmg_from_friend=-0.5, damaged_friend=-1., took_dmg_from_enemy=-1., damaged_enemy=2., died=0.5,
                 frame_rate=30):

        self.population_sizes = population_sizes
        self.population_total = sum(population_sizes)
        self.num_photoreceptors = num_photoreceptors
        self.visual_acuity = visual_acuity
        self.velocity_decay = velocity_decay
        self.dt = dt
        self.bounds = bounds
        self.maximum_acceleration = maximum_acceleration
        self.rotation_multiplier = rotation_multiplier
        self.max_damage = max_damage
        self.energy_regeneration = energy_regeneration
        self.acceleration_energy_use = acceleration_energy_use
        self.attack_energy_use = attack_energy_use
        self.attack_range = attack_range
        self.attack_angle = attack_angle
        self.friendly_fire_multiplier = friendly_fire_multiplier
        self.show_vision = show_vision
        self.vision_draw_length = vision_draw_length
        self.frame_rate = frame_rate
        self.died = died

        self.window_dimensions = window_dimensions
        wd = window_dimensions
        params = [0.3, 0.2, 0.6]
        self.layout = [[[params[0] * wd[0], 0], [(1 - params[0]) * wd[0], wd[1]]],
                       [[0, params[1] * wd[1]], [params[0] * wd[0], params[2] * wd[1]]],
                       [[0, 0], [params[0] * wd[0], params[1] * wd[1]]],
                       [[0, (params[1] + params[2]) * wd[1]], [params[0] * wd[0], (1 - params[1] - params[2]) * wd[1]]]]

        # Score multipliers
        self.took_dmg_from_friend = took_dmg_from_friend
        self.damaged_friend = damaged_friend
        self.took_dmg_from_enemy = took_dmg_from_enemy
        self.damaged_enemy = damaged_enemy

        self.vision_bins = np.arange(-self.num_photoreceptors / 2., self.num_photoreceptors / 2. + 1)
        self.vision_bins = np.exp(self.vision_bins * self.visual_acuity / self.num_photoreceptors) - np.exp(
            -self.vision_bins * self.visual_acuity / self.num_photoreceptors)
        self.vision_bins *= np.pi / 2. / (np.exp(self.visual_acuity / 2.) - np.exp(-self.visual_acuity / 2.))

        inputs = num_photoreceptors * 2 + 3 + 2     # [Vision, Velocity, Agent Stats]
        outputs = 3 + 1                             # [Acceleration, Agent Actions]
        self.basic_structure = [inputs, outputs]

        self.population = Neat.Population(self.basic_structure, sum(self.population_sizes))
        self.history = Neat.History(sum(self.basic_structure) - 1)

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

    def calculate_input(self):

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

                friendly_photoreceptors = np.angle(1j * friendly_coordinates_[0, :] + friendly_coordinates_[1, :])
                unfriendly_photoreceptors = np.angle(1j * unfriendly_coordinates_[0, :] + unfriendly_coordinates_[1, :])

                friendly_photoreceptors = np.histogram(friendly_photoreceptors, bins=self.vision_bins)[0]
                friendly_photoreceptors[int(self.num_photoreceptors/2)] -= 1
                unfriendly_photoreceptors = np.histogram(unfriendly_photoreceptors, bins=self.vision_bins)[0]

                self.population.members[k].input = np.concatenate((
                    friendly_photoreceptors, unfriendly_photoreceptors,
                    np.matmul(self.population.members[k].rotation_mat, self.population.members[k].phys.vars[3:5]),
                    self.population.members[k].phys.vars[5:6], self.population.members[k].stats[:2]))

    def fill_object_list(self):

        color_dict = {0: (50, 50, 50), 1: (200, 200, 200)}

        for i in range(len(self.population_sizes)):
            for j in range(self.population_sizes[i]):
                if self.show_vision:
                    self.object_list.append(['Fob With Vision', self.vision_bins, np.zeros(self.num_photoreceptors), 0,
                                             0, 0, color_dict[i], 0, 0, self.vision_draw_length])
                else:
                    self.object_list.append(['Fob', 0, 0, 0, color_dict[i], 0, 0])

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

    def update_member(self, member, network_output):

        member.stats[1] += self.energy_regeneration

        energy_function = np.tanh(4 * member.stats[1])
        a_multiplier = np.array([1., 0.5], dtype=np.float32) * energy_function * self.maximum_acceleration * self.dt

        member.phys.vars[6:8] = np.matmul(member.rotation_mat.transpose(),
                                          np.multiply(network_output[:2], a_multiplier))
        member.phys.vars[8] = network_output[2] * energy_function * self.maximum_acceleration * \
            self.rotation_multiplier * self.dt
        if network_output[3] > 0:
            member.stats[2] = energy_function * self.max_damage
        else:
            member.stats[2] = 0

        member.stats[1] = np.clip(member.stats[1] - self.energy_regeneration * (
            energy_function * self.acceleration_energy_use * np.sqrt(network_output[:3].dot(
                network_output[:3])) - self.attack_energy_use * member.stats[2] / self.max_damage), 0, 1)

    def update_health(self):

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

                    friendly_coordinates_ = np.copy(friendly_coordinates)
                    unfriendly_coordinates_ = np.copy(unfriendly_coordinates)
                    friendly_coordinates_[1:, :] -= coordinates[:, k][1:, None]
                    unfriendly_coordinates_[1:, :] -= coordinates[:, k][1:, None]

                    friendly_coordinates_ = friendly_coordinates_[:, np.abs(
                        friendly_coordinates_[1, :]) < self.attack_range]
                    friendly_coordinates_ = friendly_coordinates_[:, np.abs(
                        friendly_coordinates_[2, :]) < self.attack_range]
                    unfriendly_coordinates_ = unfriendly_coordinates_[:, np.abs(
                        unfriendly_coordinates_[1, :]) < self.attack_range]
                    unfriendly_coordinates_ = unfriendly_coordinates_[:, np.abs(
                        unfriendly_coordinates_[2, :]) < self.attack_range]

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
                            self.population.members[int(ind)].score += damage * self.took_dmg_from_friend
                            self.population.members[k].score += damage * self.damaged_friend

                    for ind in unfriendly_coordinates_[0, unfriendly_angle < self.attack_angle]:
                        damage = self.population.members[k].stats[2]
                        self.population.members[int(ind)].stats[0] = np.clip(
                            self.population.members[int(ind)].stats[0] - damage, 0, 1)
                        self.population.members[int(ind)].score += damage * self.took_dmg_from_enemy
                        self.population.members[k].score += damage * self.damaged_enemy

        for member in self.population.members:
            if member.stats[0] == 0:
                member.phys = Physics.Phys(displacement=np.concatenate((np.random.random_sample(2) * self.layout[0][1],
                                                                        np.array([np.random.random_sample() * 360]))))
                member.stats = np.ones(3, dtype=np.float32)
                member.score -= self.died

    def run(self, run_time, environment):
        network_graph = tf.Graph()
        lb = LoadingBar.Bar("Creating network computation graph...")
        with network_graph.as_default():
            network_in = [tf.placeholder(tf.float32, [self.basic_structure[0]])] * self.population_total
            network_in_load = [np.empty(self.basic_structure[0], np.float32)] * self.population_total
            network_out = [None] * self.population_total

            for i in range(self.population_total):
                network_out[i] = self.population.members[i].calculate_output(network_in[i])
                lb.show((i + 1.) / self.population_total)

        physics_graph = tf.Graph()
        with physics_graph.as_default():
            phys_in = tf.placeholder(tf.float32, [self.population_total, 9])
            phys_in_load = np.empty((self.population_total, 9), dtype=np.float32)
            phys_out = Physics.Phys.calculate(phys_in, self.dt, self.layout[0][1], self.velocity_decay,
                                              self.bounds)

        network_session = tf.Session(graph=network_graph)
        phys_session = tf.Session(graph=physics_graph)

        visualiser_object_list = [self.population.members[i].visualiser(self.layout[1][1], self.layout[1][0])
                                  for i in range(self.population_total)]

        lines = []
        x = [self.layout[1][1][0], self.window_dimensions[1]]
        layout_object_list = [['Rectangle', x[0] / 2., x[1] / 2., 0, x, (255, 255, 255)]]
        x = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        for ly in self.layout:
            points = []
            for i in range(4):
                points.append(np.add(ly[0], np.multiply(x[i, :], ly[1])).tolist())
            for i in range(4):
                lines.append(points[i] + points[(i + 1) % 4])
        for line in lines:
            layout_object_list.append(['Line', *line, 8])

        x = [self.window_dimensions[1], self.layout[3][0][1], self.layout[3][1][0]]
        text_object_list = []
        lines_of_text = 4
        for i in range(lines_of_text):
            text_object_list.append(['Text', x[2], 2 * (x[0] - (x[0] - x[1]) * (i + 1) / (lines_of_text + 1)), ''])
        text_object_list[0][3] = 'Generation {}'.format(self.population.generation)

        scores = np.empty(self.population_total, dtype=np.float32)

        last_frame_time = time.time()
        for i in range(int(run_time / self.dt)):

            self.calculate_input()

            for j in range(self.population_total):
                network_in_load[j] = self.population.members[j].input

            network_out_load = network_session.run(network_out, feed_dict={
                a: b for a, b in zip(network_in, network_in_load)})

            for j in range(self.population_total):
                self.update_member(self.population.members[j], network_out_load[j])
                phys_in_load[j, :] = self.population.members[j].phys.vars

            phys_out_load = phys_session.run(phys_out, feed_dict={phys_in: phys_in_load})

            for j in range(self.population_total):
                self.population.members[j].phys.vars = phys_out_load[j, :]

            self.update_health()

            if time.time() - last_frame_time > 1. / self.frame_rate:
                last_frame_time = time.time()

                self.update_object_list(self.layout[0][0])

                text_object_list[1][3] = 'Simulation time: {}h {}m {}s'.format(
                    int(i * self.dt / 3600), int(((i * self.dt) % 3600) / 60), int((i * self.dt) % 60))
                for i in range(self.population_total):
                    scores[i] = self.population.members[i].score
                text_object_list[2][3] = 'Highest Score: %.2f' % np.max(scores)
                text_object_list[3][3] = 'Num of species: {}'.format(len(self.population.species_structure))

                environment.object_lists = [self.object_list, layout_object_list,
                                            visualiser_object_list[np.argmax(scores)], text_object_list]

                pyglet.clock.tick()

                for window in pyglet.app.windows:
                    window.switch_to()
                    window.dispatch_events()
                    window.dispatch_event('on_draw')
                    window.flip()

        for i in range(self.population_total):
            self.population.member_fitness[i] = np.exp(self.population.members[i].score)


window_dim = [1000, 600]
sim = Simulation([20, 20], window_dim, num_photoreceptors=5, bounds='loop', attack_range=40, visual_acuity=8,
                 frame_rate=20.)
# sim = pickle.load(open("save.p", "rb"))
env = Graphics.Environment(window_dim)

while True:
    pickle.dump(sim, open("save.p", "wb"))
    sim.population.next_generation(sim.history)
    sim.add_extra_member_info()

    sim.run(300, env)
