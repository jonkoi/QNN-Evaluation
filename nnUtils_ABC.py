import tensorflow as tf
import math
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops

def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            x=tf.clip_by_value(x,-1,1)
            return tf.sign(x)

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
        with tf.get_default_graph().gradient_override_map({"Sign": "Identity"}):
            binarized_filters = tf.sign(tiled_filters + expanded_stddev, name="binarized_filters")
        return binarized_filters

def alpha_training(convolution_filters, binary_filters, alphas, no_filters):
    with tf.name_scope("alpha_training"):
        reshaped_convolution_filters = tf.reshape(convolution_filters, [-1], name="reshaped_convolution_filters")
        reshaped_binary_filters = tf.reshape(binary_filters, [no_filters, -1],
                                             name="reshaped_binary_filters")

        weighted_sum_filters = tf.reduce_sum(tf.multiply(alphas, reshaped_binary_filters),
                                             axis=0, name="weighted_sum_filters")

        # Defining loss
        error = tf.square(reshaped_convolution_filters - weighted_sum_filters, name="alphas_error")
        loss = tf.reduce_mean(error, axis=0, name="alphas_loss")

        # Defining optimizer
        training_op = tf.train.AdamOptimizer().minimize(loss, var_list=[alphas],
                                                        name="alphas_training_op")

        return training_op, loss

def ApproxConvLayer(input_tensor, binary_filters, alphas, no_filters, strides=(1, 1), padding="SAME", name=None):
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
            approxConv_outputs.append(this_conv)
        conv_outputs = tf.convert_to_tensor(approxConv_outputs, dtype=tf.float32,
                                            name="conv_outputs")

        # Summing up each of the binary convolution
        ApproxConv_output = tf.reduce_sum(tf.multiply(conv_outputs, reshaped_alphas), axis=0)

        return ApproxConv_output

def ABCSpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='BinarizedSpatialConvolution', no_filters_conv=5):
    def abc_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_op_scope([x], None, name, reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            alphas_conv = tf.Variable(tf.random_normal(shape=(no_filters_conv, 1), mean=1.0, stddev=0.1),dtype=tf.float32, name="alphas_conv")
            binary_filters_conv = get_binary_filters(w, no_filters_conv)
            alpha_training_conv, alpha_loss_conv = alpha_training(tf.stop_gradient(w, "no_gradient_W_conv"),
                                                            tf.stop_gradient(binary_filters_conv, "no_gradient_binary_filters_conv"), alphas_conv, no_filters_conv)
            # bin_w = binarize(w)
            bin_x = binarize(x)

            ApproxLayer = ApproxConvLayer(bin_x, binary_filters_conv, alphas_conv, no_filters_conv)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                ApproxLayer = tf.nn.bias_add(ApproxLayer, b)
            # for i in range(no_filters_conv):
            #
            #     tf.summary.histogram(name + '_bWeights_' + str(i), binary_filters_conv[i, :])
            # tf.summary.histogram(name + '_bActivation_', bin_x)
            return ApproxLayer, alpha_training_conv
    return abc_conv2d

def ABCSpatialConvolutionFirst(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='BinarizedSpatialConvolution', no_filters_conv=5):
    def abc_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_op_scope([x], None, name, reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            alphas_conv = tf.Variable(tf.random_normal(shape=(no_filters_conv, 1), mean=1.0, stddev=0.1),dtype=tf.float32, name="alphas_conv")
            binary_filters_conv = get_binary_filters(w, no_filters_conv)
            alpha_training_conv, alpha_loss_conv = alpha_training(tf.stop_gradient(w, "no_gradient_W_conv"),
                                                            tf.stop_gradient(binary_filters_conv, "no_gradient_binary_filters_conv"), alphas_conv, no_filters_conv)
            # bin_w = binarize(w)

            ApproxLayer = ApproxConvLayer(x, binary_filters_conv, alphas_conv, no_filters_conv)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                ApproxLayer = tf.nn.bias_add(ApproxLayer, b)
            # for i in range(no_filters_conv):
            #
            #     tf.summary.histogram(name + '_bWeights_' + str(i), binary_filters_conv[i, :])
            # tf.summary.histogram(name + '_bActivation_', bin_x)
            return ApproxLayer, alpha_training_conv
    return abc_conv2d

def BinarizedWeightOnlySpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='BinarizedWeightOnlySpatialConvolution'):
    '''
    This function is used only at the first layer of the model as we dont want to binarized the RGB images
    '''
    def bc_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_op_scope([x], None, name, reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bin_w = binarize(w)
            out = tf.nn.conv2d(x, bin_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return bc_conv2d

def SpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='SpatialConvolution'):
    def conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_op_scope([x], None, name, reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            out = tf.nn.conv2d(x, w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return conv2d

def Affine(nOutputPlane, bias=True, name=None, reuse=None):
    def affineLayer(x, is_training=True):
        with tf.variable_op_scope([x], name, 'Affine', reuse=reuse):
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            output = tf.matmul(reshaped, w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return affineLayer

def BinarizedAffine(nOutputPlane, bias=True, name=None, reuse=None):
    def b_affineLayer(x, is_training=True):
        with tf.variable_op_scope([x], name, 'Affine', reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            bin_x = binarize(x)
            reshaped = tf.reshape(bin_x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            bin_w = binarize(w)
            output = tf.matmul(reshaped, bin_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return b_affineLayer

def BinarizedWeightOnlyAffine(nOutputPlane, bias=True, name=None, reuse=None):
    def bwo_affineLayer(x, is_training=True):
        with tf.variable_op_scope([x], name, 'Affine', reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            bin_w = binarize(w)
            output = tf.matmul(reshaped, bin_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return bwo_affineLayer

def Linear(nInputPlane, nOutputPlane):
    return Affine(nInputPlane, nOutputPlane, add_bias=False)


def wrapNN(f,*args,**kwargs):
    def layer(x, scope='', is_training=True):
        return f(x,*args,**kwargs)
    return layer

def Dropout(p, name='Dropout'):
    def dropout_layer(x, is_training=True):
        with tf.variable_op_scope([x], None, name):
            # def drop(): return tf.nn.dropout(x,p)
            # def no_drop(): return x
            # return tf.cond(is_training, drop, no_drop)
            if is_training:
                return tf.nn.dropout(x,p)
            else:
                return x
    return dropout_layer

def ReLU(name='ReLU'):
    def layer(x, is_training=True):
        with tf.variable_op_scope([x], None, name):
            return tf.nn.relu(x)
    return layer

def HardTanh(name='HardTanh'):
    def layer(x, is_training=True):
        with tf.variable_op_scope([x], None, name):
            return tf.clip_by_value(x,-1,1)
    return layer


def View(shape, name='View'):
    with tf.variable_op_scope([x], None, name, reuse=reuse):
        return wrapNN(tf.reshape,shape=shape)

def SpatialMaxPooling(kW, kH=None, dW=None, dH=None, padding='VALID',
            name='SpatialMaxPooling'):
    kH = kH or kW
    dW = dW or kW
    dH = dH or kH
    def max_pool(x,is_training=True):
        with tf.variable_op_scope([x], None, name):
              return tf.nn.max_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding=padding)
    return max_pool

def SpatialAveragePooling(kW, kH=None, dW=None, dH=None, padding='VALID',
        name='SpatialAveragePooling'):
    kH = kH or kW
    dW = dW or kW
    dH = dH or kH
    def avg_pool(x,is_training=True):
        with tf.variable_op_scope([x], None, name):
              return tf.nn.avg_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding=padding)
    return avg_pool

def BatchNormalization(*kargs, **kwargs):
    return wrapNN(tf.contrib.layers.batch_norm, *kargs, **kwargs)


def Sequential(moduleList):
    def model(x, is_training=True):
    # Create model
        output = x
        alphas_training_operations = []
        #with tf.variable_op_scope([x], None, name):
        for i,m in enumerate(moduleList):
            output = m(output, is_training=is_training)
            if type(output) is tuple:
                output = output[0]
                alphas_training_operations.append(output[1])
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
        return output, alphas_training_operations
    return model

def Concat(moduleList, dim=3):
    def model(x, is_training=True):
    # Create model
        outputs = []
        for i,m in enumerate(moduleList):
            name = 'layer_'+str(i)
            with tf.variable_op_scope([x], name, 'Layer', reuse=reuse):
                outputs[i] = m(x, is_training=is_training)
            output = tf.concat(dim, outputs)
        return output
    return model

def Residual(moduleList, name='Residual',fixShape=['pad',1,1]):
    #fixShape:fixShape if input filters !=output filters
    #params=['pad' or 'conv',stride,stride]
    #'pad' or 'conv':fixshape method:conv1x1 or pooling1x1+padiing;
    #stride:stride for pooling
    m = Sequential(moduleList)
    def model(x, is_training=True):
    # Create model
        with tf.variable_op_scope([x], None, name):
            output=m(x,is_training=is_training)
            with tf.variable_op_scope(None, 'fixShape', reuse=None):
                filterIn=x.get_shape()[3]
                filterOut=output.get_shape()[3]
                if filterIn !=filterOut:
                    if fixShape[0]=='pad':
                        x=tf.nn.avg_pool(x, ksize=[1, 1, 1, 1], strides=[1, fixShape[1],fixShape[2],1],padding='VALID')
                        x=tf.pad(x,[[0, 0], [0, 0], [0, 0],[(filterOut-filterIn)//2, (filterOut-filterIn)//2]])
                    else:#conv method
                        w = tf.get_variable('weight', [1, 1, filterIn, filterOut],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
                        x = tf.nn.conv2d(x, w, strides=[1, fixShape[1],fixShape[2], 1], padding='SAME')
            output = tf.add(output, x)
            return output
    return model

def Residual_func(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, name='Residual_func',reuse=None,fixShapeMethod='pad',type='basic',bottleWidth=2):
        with tf.variable_op_scope(None,None, name, reuse=reuse):
            if type=='basic':
                curr_layers = [
                    SpatialConvolution(nOutputPlane,kW,kH,dW,dH, padding=padding,bias=bias),
                    BatchNormalization(),
                    ReLU(),
                    SpatialConvolution(nOutputPlane,kW,kH,1,1, padding=padding,bias=bias),
                    BatchNormalization()
                ]
            elif type=='pre':
                curr_layers = [
                    BatchNormalization(),
                    ReLU(),
                    SpatialConvolution(nOutputPlane,kW,kH,dW,dH, padding=padding,bias=bias),
                    BatchNormalization(),
                    ReLU(),
                    SpatialConvolution(nOutputPlane,kW,kH,1,1, padding=padding,bias=bias)
                ]
            elif type=='bottleneck':
                curr_layers = [
                    SpatialConvolution(nOutputPlane,1,1,1,1, padding='valid',bias=bias),
                    BatchNormalization(),
                    ReLU(),
                    SpatialConvolution(nOutputPlane,kW,kH,dW,dH, padding=padding,bias=bias),
                    BatchNormalization(),
                    ReLU(),
                    SpatialConvolution(nOutputPlane*bottleWidth,1,1,1,1, padding='valid',bias=bias),
                    BatchNormalization()
                ]
            if type=='dropout':
                curr_layers = [
                    SpatialConvolution(nOutputPlane,kW,kH,dW,dH, padding=padding,bias=bias),
                    ReLU(),
                    Dropout(0.5),
                    SpatialConvolution(nOutputPlane,kW,kH,1,1, padding=padding,bias=bias)
                ]
            elif type=='prebottleneck':
                curr_layers = [
                    BatchNormalization(),
                    ReLU(),
                    SpatialConvolution(nOutputPlane,1,1,1,1, padding='valid',bias=bias),
                    BatchNormalization(),
                    ReLU(),
                    SpatialConvolution(nOutputPlane,kW,kH,dW,dH, padding=padding,bias=bias),
                    BatchNormalization(),
                    ReLU(),
                    SpatialConvolution(nOutputPlane*bottleWidth,1,1,1,1, padding='valid',bias=bias)
                ]
            modules = []
            if 'pre' in type:
                modules=[Residual(curr_layers,fixShape=[fixShapeMethod,dW,dH])]
            else:
                modules=[Residual(curr_layers,fixShape=[fixShapeMethod,dW,dH])]+[ReLU()]
        return modules

def Block(nOutputPlane, kW, kH, dW=1, dH=1,K=10,N=4,padding='VALID', bias=True, name='Block',reuse=None,
        fixShapeMethod='pad',bottleWidth=2):# K:Network Width;N:GroupNum
    def model(x, is_training=True):
        with tf.variable_op_scope([x],None,name,reuse=reuse):
            modules = []
            for i in xrange(0,N):
                if i==0:
                    modules +=Residual_func(nOutputPlane*K,kW,kH,dW,dH,padding=padding,bias=bias,
                        reuse=reuse,fixShapeMethod=fixShapeMethod,bottleWidth=bottleWidth)
                else:
                    modules += Residual_func(nOutputPlane*K,kW,kH,1,1,padding=padding,bias=bias,
                        reuse=reuse,fixShapeMethod=fixShapeMethod,bottleWidth=bottleWidth)
            m=Sequential(modules)
            output=m(x,is_training=is_training)
            return output
    return model
