import tensorflow as tf
from reader import VideoReader
import copy
import cv2
import numpy as np
import sklearn.metrics as met
import json
import os
import PViolentDetector_new as VD
import random
import matplotlib
from math import ceil
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def standardize_img(inputs, axis=None):
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(inputs, axis=axis, keepdims=True)
    std = np.sqrt(((inputs - mean)**2).mean(axis=axis, keepdims=True))
    return (inputs - mean) / std

def do_shuffle(samples, labels):
    samples_shuf = []
    labels_shuf = []
    index_shuf = list(range(len(labels)))
    random.shuffle(index_shuf)
    return samples[index_shuf], labels[index_shuf]
    
def validate(path, create=False):
    if not path[-1] == '/':
        path += '/'
    if create:
        if not os.path.exists(path):
            os.mkdir(path)
    return path

class ViolDet:
    def __init__(self, model_path, meta_file, num_frames, frame_dif):
        self.model_path = model_path
        self.resolution = (224,224)
        self.frame_dif = frame_dif	
        self._sess = tf.InteractiveSession()
        self.x_input = tf.placeholder(dtype=tf.float32,
                                      shape=[None, (ceil(num_frames/self.frame_dif)-1),
                                             self.resolution[0], self.resolution[1], 3], name = 'x_inputm')
        self.dropout_rate = tf.placeholder(dtype=tf.float32, name='dropoutm')
        saver = tf.train.import_meta_graph(meta_file, input_map={"x_inputm:0": self.x_input, "dropoutm:0": self.dropout_rate})
        saver.restore(self._sess, self.model_path)
        self.graph = tf.get_default_graph()
        self.softed_res = self.graph.get_tensor_by_name('softed_resm:0')

    def test(self, sample):
        sample = standardize_img(sample, axis=(1,2))
        sample = np.array(sample).astype(np.float32)
        feed = {self.x_input: np.expand_dims(sample, 0), self.dropout_rate: 1.0}
        pred = self._sess.run(self.softed_res, feed_dict=feed)
        return pred[0]

class DataExtractor:
    def __init__(self, inputWidth, inputHeight, num_frames, num_dif, gen_keyword, size_of_samples=192):
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.num_frames = num_frames
        self.num_dif = num_dif
        self.keyword = gen_keyword
        self.size_of_samples = size_of_samples

    def subtract_frames(self, sample):
        difs = []
        for i in range(1, len(sample)):
            difs.append(sample[i] - sample[i-1])
        #TODO - check np array
        difs = standardize_img(np.array(difs), axis=(1,2))
        return np.array(difs)

    def get_data(self, pth):
        samples = []
        labels = []
        full_filenames = []

        for root, directories, filenames in os.walk(pth):
            random.shuffle(filenames)
            full_filenames.extend([os.path.join(root, filename) for filename in filenames])

        print('filenames -> ', full_filenames)
        for filename in full_filenames:
            full_filename = os.path.basename(filename)
            file_name_base, file_extension = os.path.splitext(full_filename)
            if len(file_extension) and file_extension in '.avi .mp4 .mpg .mov':
                video_capture = cv2.VideoCapture(filename)
                frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

                print('frame_count is ', frame_count)
                if frame_count == 0:
                    print(filename)
                samples_per_video = int(frame_count/self.num_frames)
                print('samples_per_video is ', samples_per_video)

                violent = False if self.keyword in filename else True

                for i in range(samples_per_video):
                    sample = []
                    read_frames = 0
                    while read_frames < self.num_frames:
                        ret, frame = video_capture.read()
                        if (read_frames%self.num_dif==0) and ret:
                            frame = cv2.resize(frame, (self.inputWidth, self.inputHeight))
                            sample.append(frame)
                        else:
                            if not ret:
                                sample.append(sample[-1])
                                print('pay attention for %d' % i)
                        read_frames += 1

                    labels.append([0, 1.0] if violent else [1.0, 0])
                    sample = self.subtract_frames(sample)
                    samples.append(sample)
                    print('length -> ', len(samples))
                    if len(samples) == self.size_of_samples:
                        samples = np.array(samples)
                        labels = np. array(labels)
                        samples, labels = do_shuffle(samples, labels)
                        print('yielding')
                        yield samples, labels
                        samples = []
                        labels = []
           
        print('last unfilled size -> ', len(samples))
        samples = np.array(samples)
        labels = np. array(labels)
        samples, labels = do_shuffle(samples, labels)
        yield samples, labels
                      

class Runner:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path

        with open(self.config_file_path) as datafile:
            config = json.load(datafile)

        self.model_path = config['VD_model_path']
        self.model_meta_path = config['model_meta_path']
        self.fps = config['fps']
        self.num_frames_VD = config['num_frames_VD']
        self.frame_dif_VD = config['frame_dif_VD']
        self.learning_rate = config['learning_rate_VD']
        self.batch_size = config['batch_size_VD']
        self.max_epochs = config['max_epochs']
        self.violence_threshold = config['violence_threshold']
        self.weight_path_AN = config['AlexNet_weights_path']
        self.result_dir = validate(config['result_dir'], create=True)
        self.video_paths_train = config['video_paths_train']
        self.video_paths_test = config['video_paths_test']
        self.video_paths_validation = config['video_paths_validation']
        self.keyword = config['general_keyword']
        self.logs_path = validate(config['logs_path'], create=True)
        self.logs_path = os.path.join(self.logs_path, 'log1.txt')
        self.plot = config['plot']
        self.gpu_fraction = config['gpu_fraction']
        self.inputWidth = 224
        self.inputHeight = 224

        if config['run_train']:
            print('training the model')
            self.train()

        if config['run_test']:
            print('testing the model')
            self.test()


    def get_batch(self, samples, labels):
        batch_index = 0
        while batch_index + self.batch_size < len(samples):
            x_batch = samples[batch_index:batch_index+self.batch_size]
            y_batch = labels[batch_index:batch_index+self.batch_size]

            batch_index += self.batch_size
            yield x_batch, y_batch
            
    def _evaluate(self, label, y_pred):
        threshold = self.violence_threshold
        for i in range(len(y_pred)):
            y_pred[i] = 0 if y_pred[i]<threshold else 1
        accuracy = round(met.accuracy_score(label, y_pred), 3)
        recall = round(met.recall_score(label, y_pred), 3)
        precision = round(met.precision_score(label, y_pred), 3)
        f1 = round(met.f1_score(label, y_pred), 3)
        print('accuracy = {}'.format(accuracy))
        print('recall = {}'.format(recall))
        print('precision = {}'.format(precision))
        print('f1 score = {}'.format(f1))
        print('')
        print('accuracy = {}'.format(accuracy), file=open(self.logs_path, 'a'))
        print('recall = {}'.format(recall), file=open(self.logs_path, 'a'))
        print('precision = {}'.format(precision), file=open(self.logs_path, 'a'))
        print('f1 score = {}'.format(f1), file=open(self.logs_path, 'a'))
        print('', file=open(self.logs_path, 'a'))

    def train(self):
        early_stop_threshold = 7
        stopping_step = 0
        sess = tf.InteractiveSession()
        v_detector = VD.VDetector(self.num_frames_VD, self.inputWidth, self.inputHeight, 3, self.batch_size, sess, frame_dif=self.frame_dif_VD, logs_path=self.logs_path, learning_rate=self.learning_rate, alexnet_path=self.weight_path_AN)
        saver = tf.train.Saver(max_to_keep=8)
        data_extractor = DataExtractor(self.inputWidth, self.inputHeight, self.num_frames_VD, self.frame_dif_VD, self.keyword, size_of_samples=256)
        for e in range(self.max_epochs):
            total_train_cost = 0
            total_train_accuracy = 0
            train_iter = 0
            total_valid_accuracy = 0
            valid_iter = 0
            print('epoch %d' % e)
            print('epoch %d' % e, file=open(self.logs_path, 'a'))
            
            for training_data, training_labels in data_extractor.get_data(self.video_paths_train):
                for x_batch, y_batch in self.get_batch(training_data, training_labels):
                    c, acc = v_detector.train(x_batch, y_batch)
                    total_train_cost += c
                    total_train_accuracy += acc
                    train_iter += 1
                    print(y_batch, file=open(self.logs_path, 'a'))

            for valid_data, valid_labels in data_extractor.get_data(self.video_paths_validation):
                for x_batch, y_batch in self.get_batch(valid_data, valid_labels):
                    acc = v_detector.test(x_batch, y_batch)
                    total_valid_accuracy += acc
                    valid_iter += 1
                    print('v -> ', y_batch, file=open(self.logs_path, 'a'))

            if train_iter != 0: 
                print('train cost of epoch %d is %.5f' % (e, total_train_cost/train_iter))
                print('train cost of epoch %d is %.5f' % (e, total_train_cost/train_iter), file=open(self.logs_path, 'a'))
    
                print('train accuracy of epoch %d is %.5f' % (e, total_train_accuracy/train_iter), file=open(self.logs_path, 'a'))
                print('train accuracy of epoch %d is %.5f' % (e, total_train_accuracy/train_iter))
            else:
                print('NOTE!!! no training iteration')

            if valid_iter != 0:
                print('accuracy = ', total_valid_accuracy/valid_iter)
                print('accuracy = ', total_valid_accuracy/valid_iter, file=open(self.logs_path, 'a'))

            save_model_loc = os.path.join(self.model_path, ('VD_model_%s.nn' % e))
            saver.save(sess, save_model_loc)
            print('model saved in %s' % save_model_loc)
            print('model saved in %s' % save_model_loc, file=open(self.logs_path, 'a'))

            if e == 0:
                best_loss = (total_train_cost/train_iter)
            else:
                if ((total_train_cost/train_iter) < best_loss):
                    stopping_step = 0
                    best_loss = total_train_cost/train_iter
                else:
                    stopping_step += 1
                if stopping_step >= early_stop_threshold:
                    self.model_path = os.path.join(self.model_path, ('VD_model_%s.nn' % (e-early_stop_threshold)))
                    self.model_meta_path = os.path.join(self.model_meta_path, ('VD_model_%s.nn.meta' % (e-early_stop_threshold)))
                    print("Early stopping is trigger at step:", file=open(self.logs_path, 'a'))
                    break
            print('training ended !!')
        sess.close()

    def test(self):
        true_violence = []
        predicted_violence = []
        test_iter = 0
        total_test_acc = 0

        detector = ViolDet(self.model_path, self.model_meta_path, self.num_frames_VD, self.frame_dif_VD)
        data_extractor = DataExtractor(self.inputWidth, self.inputHeight, self.num_frames_VD, self.frame_dif_VD, self.keyword, size_of_samples=256)

        for testing_data, testing_labels in data_extractor.get_data(self.video_paths_test):
            for x_batch, y_batch in self.get_batch(testing_data, testing_labels):
                true_violence.extend(np.array(y_batch)[:, 1])
                for j, clip in enumerate(x_batch):
                    res = detector.test(clip)
                    predicted_violence.append(res[1])
                    print(res[1], file=open(self.logs_path, 'a'), end=' ')
                    print(y_batch[j][1], file=open(self.logs_path, 'a'))
                test_iter += 1
                print('', file=open(self.logs_path, 'a'))
                print(y_batch, file=open(self.logs_path, 'a'))

        self._evaluate(true_violence, predicted_violence.copy())
        predicted_violence = np.array(predicted_violence)
        true_violence = np.array(true_violence)
        idx = np.argsort(true_violence)
        if self.plot:
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(predicted_violence)), predicted_violence[idx], c='blue', alpha=0.5)
            plt.plot(range(len(true_violence)), true_violence[idx], c='red', alpha=0.5)
            plt.savefig(os.path.join(self.result_dir, 'probs_test.png'))
            plt.close()








