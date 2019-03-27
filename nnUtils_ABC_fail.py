from __future__ import division, print_function
import tensorflow as tf
import numpy as np

def get_mean_stddev(input_tensor):
    with tf.name_scope('mean_stddev_cal'):
        mean, variance = tf.nn.moments(input_tensor, axes=list(range(len(input_tensor.get_shape()))))
        stddev = tf.sqrt(variance, name="standard_deviation")
        return mean, stddev

# TODO: Allow shift parameters to be learnable
def get_shifted_stddev(stddev, no_filters):
    with tf.name_scope('shifted_stddev'):
        spreaded_deviation = -1. + (2./(no_filters - 1)) * tf.convert_to_tensor(list(range(no_filters)),
                                                                                dtype=tf.float32)
        return spreaded_deviation * stddev

def get_binary_filters(convolution_filters, no_filters, name=None):
    with tf.name_scope(name, default_name="get_binary_filters"):
        mean, stddev = get_mean_stddev(convolution_filters)
        shifted_stddev = get_shifted_stddev(stddev, no_filters)

        # Normalize the filters by subtracting mean from them
        mean_adjusted_filters = convolution_filters - mean

        # Tiling filters to match the number of filters
        expanded_filters = tf.expand_dims(mean_adjusted_filters, axis=0, name="expanded_filters")
        tiled_filters = tf.tile(expanded_filters, [no_filters] + [1] * len(convolution_filters.get_shape()),
                                name="tiled_filters")

        # Similarly tiling spreaded stddev to match the shape of tiled_filters
        expanded_stddev = tf.reshape(shifted_stddev, [no_filters] + [1] * len(convolution_filters.get_shape()),
                                     name="expanded_stddev")

        binarized_filters = tf.sign(tiled_filters + expanded_stddev, name="binarized_filters")
        return binarized_filters

def get_alphas(convolution_filters, binary_filters, no_filters, name=None):
    with tf.name_scope(name, "get_alphas"):
        # Reshaping convolution filters to be one dimensional and binary filters to be of [no_filters, -1] dimension
        reshaped_convolution_filters = tf.reshape(convolution_filters, [-1], name="reshaped_convolution_filters")
        reshaped_binary_filters = tf.reshape(binary_filters, [no_filters, -1],
                                             name="reshaped_binary_filters")

        # Creating variable for alphas
        alphas = tf.Variable(tf.random_normal(shape=(no_filters, 1), mean=1.0, stddev=0.1), name="alphas")

        # Calculating W*alpha
        weighted_sum_filters = tf.reduce_sum(tf.multiply(alphas, reshaped_binary_filters),
                                             axis=0, name="weighted_sum_filters")

        # Defining loss
        error = tf.square(reshaped_convolution_filters - weighted_sum_filters, name="alphas_error")
        loss = tf.reduce_mean(error, axis=0, name="alphas_loss")

        # Defining optimizer
        training_op = tf.train.AdamOptimizer().minimize(loss, var_list=[alphas],
                                                        name="alphas_training_op")

        return alphas, training_op, loss

def ApproxConv(no_filters, convolution_filters, convolution_biases=None,
               strides=(1, 1), padding="VALID", name=None):
    with tf.name_scope(name, "ApproxConv"):
        # Creating variables from input convolution filters and convolution biases
        filters = tf.Variable(convolution_filters, dtype=tf.float32, name="filters")
        if convolution_biases is None:
            biases = 0.
        else:
            biases = tf.Variable(convolution_biases, dtype=tf.float32, name="biases")

        # Creating binary filters
        binary_filters = get_binary_filters(filters, no_filters)

        # Getting alphas
        alphas, alphas_training_op, alphas_loss = get_alphas(filters, binary_filters,
                                                             no_filters)

        # Defining function for closure to accept multiple inputs with same filters
        def ApproxConvLayer(input_tensor, name=None):
            with tf.name_scope(name, "ApproxConv_Layer"):
                # Reshaping alphas to match the input tensor
                reshaped_alphas = tf.reshape(alphas,
                                             shape=[no_filters] + [1] * len(input_tensor.get_shape()),
                                             name="reshaped_alphas")

                # Calculating convolution for each binary filter
                approxConv_outputs = []
                for index in range(no_filters):
                    # Binary convolution
                    this_conv = tf.nn.conv2d(input_tensor, binary_filters[index],
                                             strides=(1,) + strides + (1,),
                                             padding=padding)
                    approxConv_outputs.append(this_conv + biases)
                conv_outputs = tf.convert_to_tensor(approxConv_outputs, dtype=tf.float32,
                                                    name="conv_outputs")

                # Summing up each of the binary convolution
                ApproxConv_output = tf.reduce_sum(tf.multiply(conv_outputs, reshaped_alphas), axis=0)

                return ApproxConv_output

        return alphas_training_op, ApproxConvLayer, alphas_loss

def ABC(convolution_filters, convolution_biases=None, no_binary_filters=5, no_ApproxConvLayers=5,
        strides=(1, 1), padding="VALID", name=None):
    with tf.name_scope(name, "ABC"):
        # Creating variables shift parameters and weighted sum parameters (betas)
        shift_parameters = tf.Variable(tf.constant(0., shape=(no_ApproxConvLayers, 1)), dtype=tf.float32,
                                       name="shift_parameters")
        betas = tf.Variable(tf.constant(1., shape=(no_ApproxConvLayers, 1)), dtype=tf.float32,
                            name="betas")

        # Instantiating the ApproxConv Layer
        alphas_training_op, ApproxConvLayer, alphas_loss = ApproxConv(no_binary_filters,
                                                                      convolution_filters, convolution_biases,
                                                                      strides, padding)

        def ABCLayer(input_tensor, name=None):
            with tf.name_scope(name, "ABCLayer"):
                # Reshaping betas to match the input tensor
                reshaped_betas = tf.reshape(betas,
                                            shape=[no_ApproxConvLayers] + [1] * len(input_tensor.get_shape()),
                                            name="reshaped_betas")

                # Calculating ApproxConv for each shifted input
                ApproxConv_layers = []
                for index in range(no_ApproxConvLayers):
                    # Shifting and binarizing input
                    shifted_input = tf.clip_by_value(input_tensor + shift_parameters[index], 0., 1.,
                                                     name="shifted_input_" + str(index))
                    binarized_activation = tf.sign(shifted_input - 0.5)

                    # Passing through the ApproxConv layer
                    ApproxConv_layers.append(ApproxConvLayer(binarized_activation))
                ApproxConv_output = tf.convert_to_tensor(ApproxConv_layers, dtype=tf.float32,
                                                         name="ApproxConv_output")

                # Taking the weighted sum using the betas
                ABC_output = tf.reduce_sum(tf.multiply(ApproxConv_output, reshaped_betas), axis=0)
                return ABC_output

        return alphas_training_op, ABCLayer, alphas_loss

# Defining utils function
def weight_variable(shape, name="weight"):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def ABC_model():
    alphas_training_operations = []
    def model(x, is_training=True):
        with tf.variable_op_scope([x], None, "ABCConv_1", reuse=None):
            w1 = weight_variable(shape=([3, 3, 3, 128]), name="weight_1")
            alphas_training_op1, ABCLayer1, alphas_loss1 = ABC(w1, padding="SAME")
            alphas_training_operations.append(alphas_training_op1)
            conv1 = ABCLayer1(x)
            bn_conv1 = tf.layers.batch_normalization(conv1, axis=-1)
            h_conv1 = tf.nn.relu(bn_conv1)

            w2 = weight_variable(shape=([3, 3, 128, 128]), name="weight_2")
            alphas_training_op2, ABCLayer2, alphas_loss2 = ABC(w2, padding="SAME")
            alphas_training_operations.append(alphas_training_op2)
            conv2 = ABCLayer2(h_conv1)
            pool2 = max_pool_2x2(conv2)
            bn_conv2 = tf.layers.batch_normalization(pool2, axis=-1)
            h_conv2= tf.nn.relu(bn_conv2)

            w3 = weight_variable(shape=([3, 3, 128, 256]), name="weight_3")
            alphas_training_op3, ABCLayer3, alphas_loss3 = ABC(w3, padding="SAME")
            alphas_training_operations.append(alphas_training_op3)
            conv3 = ABCLayer3(h_conv2)
            bn_conv3 = tf.layers.batch_normalization(conv3, axis=-1)
            h_conv3= tf.nn.relu(bn_conv3)

            w4 = weight_variable(shape=([3, 3, 256, 256]), name="weight_4")
            alphas_training_op4, ABCLayer4, alphas_loss4 = ABC(w4, padding="SAME")
            alphas_training_operations.append(alphas_training_op4)
            conv4 = ABCLayer4(h_conv3)
            pool4 = max_pool_2x2(conv4)
            bn_conv4 = tf.layers.batch_normalization(pool4, axis=-1)
            h_conv4= tf.nn.relu(bn_conv4)

            w5 = weight_variable(shape=([3, 3, 256,512]), name="weight_5")
            alphas_training_op5, ABCLayer5, alphas_loss5 = ABC(w5, padding="SAME")
            alphas_training_operations.append(alphas_training_op5)
            conv5 = ABCLayer5(h_conv4)
            bn_conv5 = tf.layers.batch_normalization(conv5, axis=-1)
            h_conv5= tf.nn.relu(bn_conv5)

            w6 = weight_variable(shape=([3, 3, 512,512]), name="weight_6")
            alphas_training_op6, ABCLayer6, alphas_loss6 = ABC(w6, padding="SAME")
            alphas_training_operations.append(alphas_training_op6)
            conv6 = ABCLayer6(h_conv5)
            pool6 = max_pool_2x2(conv6)
            bn_conv6 = tf.layers.batch_normalization(pool6, axis=-1)
            h_conv6= tf.nn.relu(bn_conv6)

            reshaped = tf.reshape(h_conv6, [h_conv6.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w_fc1 = tf.get_variable('weight_fc1', [nInputPlane, 1024], initializer=tf.contrib.layers.xavier_initializer())
            fc1 = tf.nn.relu(tf.matmul(reshaped, w_fc1))
            bn_fc1 = tf.layers.batch_normalization(fc1, axis=-1)
            h_fc1 = tf.nn.relu(bn_fc1)

            w_fc2 = tf.get_variable('weight_fc2', [1024, 10], initializer=tf.contrib.layers.xavier_initializer())
            output = tf.nn.relu(tf.matmul(h_fc1, w_fc2))

        return output
    return model, alphas_training_operations
