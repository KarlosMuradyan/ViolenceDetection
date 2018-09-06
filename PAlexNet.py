import tensorflow as tf
import numpy as np
w_index = 0
b_index = 0

def new_weights(shape):
    global w_index
    w_index += 1
    return tf.get_variable(shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
                           name='weight_%d' % w_index, dtype=tf.float32)
    # return tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
    #         name='weight_%d' % w_index, dtype=tf.float32)

def new_biases(shape):
    global b_index
    b_index += 1
    return tf.get_variable(initializer=tf.constant(0.001, shape=shape, dtype=tf.float32),
                           name='bias_%d' % b_index)

def load_weights(pth):
    net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
    return net_data



def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



def with_weights(img, net_data):
    print('with weighttttttttttttttttttttttttttttttttssssssssssssssssssssssssss')
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(img, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)
    
    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    
    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    
    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)
    
    
    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    
    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)
    
    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)
    
    
    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)
    print('alexnet ret shape is  ', conv5.shape)
    return conv5


def apply_alexNet(img, pth_to_weights=''):
    # reshape the input image vector to 227 x 227 x 3 dimensions
    img = tf.reshape(img, [-1, 224, 224, 3],)
    # print(type(img))
    loaded = False
    if pth_to_weights:
        loaded=True
        net_data = load_weights(pth_to_weights)
        res = with_weights(img, net_data)
        return res
    # 1st convolutional layer
    if loaded:
        w1 = tf.Variable(net_data["conv1"][0])
        b1 = tf.Variable(net_data["conv1"][1])
    else:
        w1 = new_weights([11, 11, 3, 96])
        b1 = new_biases(shape=[96])
    conv1 = tf.nn.conv2d(img, w1, strides=[1, 4, 4, 1], padding="SAME", name="conv1")
    conv1 = tf.nn.bias_add(conv1, b1)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.local_response_normalization(conv1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # 2nd convolutional layer
    if loaded:
        w2 = tf.Variable(net_data["conv2"][0])
        b2 = tf.Variable(net_data["conv2"][1])
    else:
        w2 = new_weights([5, 5, 96, 256])
        b2 = new_biases(shape=[256])
    print(conv1.shape)
    print(w2.shape)
    conv2 = tf.nn.conv2d(conv1, w2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
    conv2 = tf.nn.bias_add(conv2, b2)
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.local_response_normalization(conv2, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # 3rd convolutional layer
    if loaded:
        w3 = tf.Variable(net_data["conv3"][0])
        b3 = tf.Variable(net_data["conv3"][1])
    else:
        w3 = new_weights([3, 3, 256, 384])
        b3 = new_biases(shape=[384])
    conv3 = tf.nn.conv2d(conv2, w3, strides=[1, 1, 1, 1], padding="SAME", name="conv3")
    conv3 = tf.nn.bias_add(conv3, b3)
    conv3 = tf.nn.relu(conv3)

    # 4th convolutional layer
    if loaded:
        w4 = tf.Variable(net_data["conv4"][0])
        b4 = tf.Variable(net_data["conv4"][1])
    else:
        w4 = new_weights([3, 3, 384, 384])
        b4 = new_biases(shape=[384])
    conv4 = tf.nn.conv2d(conv3, w4, strides=[1, 1, 1, 1], padding="SAME", name="conv4")
    conv4 = tf.nn.bias_add(conv4, b4)
    conv4 = tf.nn.relu(conv4)

    # 5th convolutional layer
    if loaded:
        w5 = tf.Variable(net_data["conv5"][0])
        b5 = tf.Variable(net_data["conv5"][1])
    else:
        w5 = new_weights([3, 3, 384, 256])
        b5 = new_biases(shape=[256])
    conv5 = tf.nn.conv2d(conv4, w5, strides=[1, 1, 1, 1], padding="SAME", name="conv5")
    conv5 = tf.nn.bias_add(conv5, b5)
    conv5 = tf.nn.relu(conv5)
    conv5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    return conv5
