import tensorflow as tf
import PAlexNet as AlexNet
import numpy as np
from math import ceil

w_index = 0
b_index = 0
def new_weights(shape, name):
    global w_index
    w_index += 1
    return tf.get_variable(shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                           name=name, dtype=tf.float32)
    # return tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
    #         name=name, dtype=tf.float32)


def new_bias(shape, name):
    global b_index
    b_index += 1
    return tf.get_variable(initializer=tf.constant(0.1, shape=shape, dtype=tf.float32),
                           name=name)


def new_fc_layer(input,
                 num_inputs,
                 num_outputs,
                 fc_num,
                 use_relu=True):

    shape = [num_inputs, num_outputs]
    weights = new_weights(shape, name=('weights_fcLayerm_%d') % fc_num)
    biases = new_bias([num_outputs], name=('bias_fcLayerm_%d') % fc_num)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer, name=('relu_fcm_%d') % fc_num)

    return layer


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    # num_of_features = layer_shape[0] * layer_shape[2] * layer_shape[3]
    num_of_features = np.array(layer_shape[1:], dtype=int).prod()
    return tf.reshape(layer, shape=[-1, num_of_features]), int(num_of_features)


class VDetector:

    def __init__(self,frames_per_sample, inputWidth, inputHeight, inputChannels, batch_size, sess, frame_dif, learning_rate=0.00005, dropout_rate=0.7, logs_path='logs/log1.txt', alexnet_path=''):

        self.logs_path = logs_path
        self.alexnet_path = alexnet_path
        # video details
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
        self.frames_per_sample = frames_per_sample
        self.inputChannels = inputChannels
        self.batch_size = batch_size

        # placeholders and training process
        self.range_of_dif = frame_dif
        self.x_input = tf.placeholder(dtype=tf.float32,
                                      shape=[None, (ceil(self.frames_per_sample/self.range_of_dif)-1),
                                             inputWidth, inputHeight, inputChannels], name = 'x_inputm')
        self.y_input = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y_inputm')
        self.dropout_rate = tf.placeholder(dtype=tf.float32, name='dropoutm')
        self.train_dropout = dropout_rate

        self.res = self.prepare_training()
        self.softed_res = tf.nn.softmax(self.res, name='softed_resm')
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.res, labels=self.y_input), name='costm')
        self.correct_pred = tf.equal(tf.argmax(self.softed_res, 1), tf.argmax(self.y_input, 1), name='correct_predm')
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, 'float32'), name='accuracym')

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='AdamOptimizerm')
        self.minimize = self.optimizer.minimize(self.cost)

        # initializeing variables
        init = tf.global_variables_initializer()
        sess.run(init)
        self.sess = sess


    def prepare_training(self):
        convLstm = tf.contrib.rnn.ConvLSTMCell( conv_ndims=2,  # ConvLSTMCell definition
                                                input_shape=[13, 13, 256],
                                                output_channels=256,
                                                kernel_shape=[3, 3],
                                                skip_connection=False,
                                                name='ConvLSTMm')
 
        self.alexnet_res = AlexNet.apply_alexNet(self.x_input, self.alexnet_path)
        self.alexnet_res = tf.reshape(self.alexnet_res, shape=(-1, (ceil(self.frames_per_sample/self.range_of_dif)-1), 13, 13, 256))
        (outputs, state) = tf.nn.dynamic_rnn(convLstm, self.alexnet_res, time_major=False, dtype=tf.float32)
        # print('output -1 size', outputs.shape)
        # print('state -1 shape', len(state))
        
        self.flattened, n_flattened = flatten_layer(state[0])
        # flattened = tf.nn.batch_normalization(flattened, tf.constant(0.0), tf.constant(1.0),
        #                                       offset=None, scale=None, variance_epsilon=tf.constant(0.0001),
        #                                       name='batch_norm')
        self.flattened = tf.layers.batch_normalization(self.flattened)
        fc1 = new_fc_layer(self.flattened, n_flattened, 1000, 1)
        fc1 = tf.nn.dropout(fc1, self.dropout_rate)
        fc2 = new_fc_layer(fc1, 1000, 128, 2)
        fc2 = tf.nn.dropout(fc2, self.dropout_rate)
        fc3 = new_fc_layer(fc2, 128, 10, 3)
        fc3 = tf.nn.dropout(fc3, self.dropout_rate)
        fc4 = new_fc_layer(fc3, 10, 2, 4, use_relu=False)

        return fc4

    def standardize_img(self, inputs, axis=None):
        # axis param denotes axes along which mean & std reductions are to be performed
        mean = np.mean(inputs, axis=axis, keepdims=True)
        std = np.sqrt(((inputs - mean)**2).mean(axis=axis, keepdims=True))
        return (inputs - mean) / std

    def train(self, x_batch, y_batch):
        x_input_batch = []
        # for j, x_frames in enumerate(x_batch):
        #     dif = np.empty(shape=((ceil(self.frames_per_sample/self.range_of_dif)-1), self.inputHeight,
        #                           self.inputWidth, self.inputChannels))
        #     i = 0
        #     k = 0
        #     while (k+self.range_of_dif)<len(x_frames):
        #         dif[i] = np.subtract(x_frames[k], x_frames[k+self.range_of_dif])
        #         k += self.range_of_dif
        #         i += 1
        #     standardized_images = self.standardize_img(dif, axis=(1,2))
        #     x_input_batch.append(standardized_images)
        # del x_batch

        feed = {self.x_input: x_batch,
                self.y_input: y_batch,
                self.dropout_rate: self.train_dropout}
        res, _, c, acc, alex, flat = self.sess.run([self.softed_res, self.minimize, self.cost, self.acc, self.alexnet_res, self.flattened], feed_dict=feed)
        
        print(res, file=open(self.logs_path, 'a'))
        print('cost of batch is ', c)
        return c, acc

    def test(self, x_batch, y_batch):
        # x_input_batch = np.empty(shape=(self.batch_size, int(self.frames_per_sample/self.range_of_dif),
        #                                 self.inputHeight, self.inputWidth, self.inputChannels))
        # x_input_batch = []
        # for j, x_frames in enumerate(x_batch):
        #     dif = np.empty(shape=((ceil(self.frames_per_sample/self.range_of_dif)-1), self.inputHeight,
        #                           self.inputWidth, self.inputChannels))
        #     for k in range(dif.shape[0]):
        #         dif[k] = np.subtract(x_frames[k], x_frames[k+self.range_of_dif])
        #     # x_input_batch[j] = dif
        #     x_input_batch.append(dif)

        # del x_batch

        feed = {self.x_input: x_batch,
                self.y_input: y_batch,
                self.dropout_rate: 1.0}
        res, acc = self.sess.run([self.softed_res, self.acc], feed_dict=feed)
        print(res, file=open(self.logs_path, 'a'))
        return acc

