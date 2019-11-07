import numpy as np
import tensorflow.compat.v1 as tf

datatype = tf.float32


class Phys:

    def __init__(self, displacement=np.zeros(3), velocity=np.zeros(3), acceleration=np.zeros(3)):

        self.vars = np.zeros(9)
        self.vars[0:3] = displacement
        self.vars[3:6] = velocity
        self.vars[6:9] = acceleration

    @staticmethod
    def calculate(vars_0, dt, window_dimensions, velocity_decay, bounds='wall'):
        dt_ = tf.constant(dt, dtype=datatype, name='dt')
        velocity_decay = tf.constant(velocity_decay, dtype=datatype, name='velocity_decay')

        velocity_1 = tf.add(vars_0[:, 3:6], tf.multiply(vars_0[:, 6:9], dt_), name='velocity')
        velocity_2 = tf.multiply(velocity_1, velocity_decay, name='velocity')

        displacement_1 = tf.add(vars_0[:, 0:3], tf.multiply(velocity_2, dt_), name='displacement')
        displacement_2 = Bounds.apply_boundary(displacement_1, window_dimensions, bounds)

        return tf.concat([displacement_2, tf.concat([velocity_2, vars_0[:, 6:9]], 1)], 1, name='variables')


class Bounds:

    @staticmethod
    def apply_boundary(displacement_0, window_dimensions, bounds):

        upper_bound = tf.constant(window_dimensions, dtype=datatype, name='upper_bound')
        lower_bound = tf.constant([0, 0], dtype=datatype, name='lower_bound')

        angle_1 = tf.mod(displacement_0[:, 2], 360, name='angle')
        position_1 = ({'wall': lambda x: tf.clip_by_value(x, lower_bound, upper_bound, name='position'),
                       'loop': lambda x: tf.mod(x, upper_bound, name='position')
                       }[bounds](displacement_0[:, :2]))

        return tf.concat([position_1, tf.transpose([angle_1])], 1, name='displacement')
