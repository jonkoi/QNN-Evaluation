import tensorflow as tf
import math
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import function
import numpy as np

def cabs(x):
    return tf.minimum(1.0, tf.abs(x), name = 'cabs')


def get_dorefa(bitW, bitA, bitG):
    """
    return the three quantization functions fw, fa, fg, for weights, activations and gradients respectively
    It's unsafe to call this function multiple times with different parameters
    """
    G = tf.get_default_graph()

    def quantize(x, k):
        n = float(2**k - 1)
        with G.gradient_override_map({"Round": "Identity"}):
            return tf.round(x * n) / n

    def fw(x):
        if bitW == 32:
            return x
        if bitW == 1:   # BWN
            print('BinaryWeight!')
            with G.gradient_override_map({"Sign": "Identity"}):
                E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
                return tf.sign(x / E) * E
        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        return 2 * quantize(x, bitW) - 1

    def fa(x):
        if bitA == 32:
            return x
        return quantize(x, bitA)

    @tf.RegisterGradient("FGGrad")
    def grad_fg(op, x):
        rank = x.get_shape().ndims
        assert rank is not None
        maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
        x = x / maxx
        n = float(2**bitG - 1)
        x = x * 0.5 + 0.5 + tf.random_uniform(
            tf.shape(x), minval=-0.5 / n, maxval=0.5 / n)
        x = tf.clip_by_value(x, 0.0, 1.0)
        x = quantize(x, bitG) - 0.5
        out = x * maxx * 2

        tf.summary.histogram(name + '_bGradient', out)
        return out

    def fg(x):
        if bitG == 32:
            return x
        with G.gradient_override_map({"Identity": "FGGrad"}):
            return tf.identity(x)
    return fw, fa, fg

BITW = 1
BITA = 1
BITG = 32
fw, fa, fg = get_dorefa(BITW, BITA, BITG)

def DorefaOnlyWeightSpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='DoReFa_Convolution_w'):
    def b_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_op_scope([x], None, name, reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bin_w = fw(w)
            '''
            Note that we use binarized version of the input and the weights. Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            out = tf.nn.conv2d(x, bin_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
                # out = ReLU(out)
            out = fg(out)
            tf.summary.histogram(name + '_bWeights', bin_w)
            return out
    return b_conv2d


def DorefaSpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1, padding='VALID', bias=True, reuse=None, name='DoReFa_Convolution'):
    def b_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_op_scope([x], None, name, reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bin_w = fw(w)
            bin_x = fa(x)
            '''
            Note that we use binarized version of the input and the weights. Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            out = tf.nn.conv2d(bin_x, bin_w, strides=[1, dH, dW, 1], padding=padding)
            out = fg(out)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
                # out = ReLU(out)
            tf.summary.histogram(name + '_bWeights', bin_w)
            tf.summary.histogram(name + '_bActivation', bin_x)
            return out
    return b_conv2d

def DoReFaAffine(nOutputPlane, bias=True, name='DoReFaAffine', reuse=None):
    def b_affineLayer(x, is_training=True):
        with tf.variable_op_scope([x], None, name, reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            bin_x = fa(x)
            reshaped = tf.reshape(bin_x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            bin_w = fw(w)
            # reshaped = fa(cabs(reshaped))
            output = tf.matmul(reshaped, bin_w)
            output = fg(output)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
                # output = ReLU(output)
            tf.summary.histogram(name + '_bWeights', bin_w)
            tf.summary.histogram(name + '_bActivation', bin_x)
        return output
    return b_affineLayer


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
        #with tf.variable_op_scope([x], None, name):
        for i,m in enumerate(moduleList):
            output = m(output, is_training=is_training)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
        return output
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
