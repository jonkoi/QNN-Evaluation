import tensorflow as tf
import WAGE_Option as Option
import WAGE_Initializer
import math

LR    = Option.lr
bitsW = Option.bitsW
bitsA = Option.bitsA
bitsG = Option.bitsG
# bitsG = 32
bitsE = Option.bitsE
bitsR = Option.bitsR
L2 = Option.L2

Graph = tf.get_default_graph()

def S(bits):
  return 2.0 ** (bits - 1)

def Shift(x):
  return 2 ** tf.round(tf.log(x) / tf.log(2.0))

def C(x, bits=32):
  if bits > 15 or bits == 1:
    delta = 0.
  else:
    delta = 1. / S(bits)
  MAX = +1 - delta
  MIN = -1 + delta
  x = tf.clip_by_value(x, MIN, MAX, name='saturate')
  return x

def Q(x, bits):
  if bits > 15:
    return x
  elif bits == 1:  # BNN
    return tf.sign(x)
  else:
    SCALE = S(bits)
    return tf.round(x * SCALE) / SCALE

def W(x,scale = 1.0):
  with tf.name_scope('QW'):
    with Graph.gradient_override_map({'Identity': 'CustomGrad'}):
        x = tf.identity(x)

    y = Q(C(x, bitsW), bitsW)
    # we scale W in QW rather than QA for simplicity

    if scale > 1.8:
      y = y/scale
    # if bitsG > 15:
      # when not quantize gradient, we should rescale the scale factor in backprop
      # otherwise the learning rate will have decay factor scale
      # x = x * scale
    return x + tf.stop_gradient(y - x)  # skip derivation of Quantize and C

def A(x):
  with tf.name_scope('QA'):
    x = C(x, bitsA)
    y = Q(x, bitsA)
    return x + tf.stop_gradient(y - x)  # skip derivation of Quantize, but keep Clip

@tf.RegisterGradient('CustomGrad')
def grads(op,x):
  with tf.name_scope('QG'):
    if bitsG > 15:
      return x
    else:
      if x.name.lower().find('batchnorm') > -1:
        return x  # batch norm parameters, not quantize now

      xmax = tf.reduce_max(tf.abs(x))
      x = x / Shift(xmax)

      norm = Q(LR * x, bitsR)

      norm_sign = tf.sign(norm)
      norm_abs = tf.abs(norm)
      norm_int = tf.floor(norm_abs)
      norm_float = norm_abs - norm_int
      rand_float = tf.random_uniform(x.get_shape(), 0, 1)
      norm = norm_sign * ( norm_int + 0.5 * (tf.sign(norm_float - rand_float) + 1) )

      return norm / S(bitsG)

@tf.RegisterGradient('Error')
def error(op, x):
  if bitsE > 15:
    return x
  else:
    xmax = tf.reduce_max(tf.abs(x))
    xmax_shift = Shift(xmax)
    return Q(C( x /xmax_shift, bitsE), bitsE)

def E(x):
  with tf.name_scope('QE'):
    with Graph.gradient_override_map({'Identity': 'Error'}):
      return tf.identity(x)


### layers

def WAGESpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='WAGE_Convolution', scale=1.0):
    def b_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(None, name, [x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.random_uniform_initializer(-Option.fixed_limit, Option.fixed_limit))
            w = W(w, scale)
            if bitsA <= 16:
                x = A(x)
            if bitsE <= 16:
                x = E(x)

            out = tf.nn.conv2d(x, w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out

    return b_conv2d

def WAGEWeightOnlySpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='WAGE_Convolution',scale = 1.0):
    def b_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(None, name, [x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.random_uniform_initializer(-Option.fixed_limit, Option.fixed_limit))
            w = W(w, scale)

            out = tf.nn.conv2d(x, w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return b_conv2d

def WAGEAffine(nOutputPlane, bias=True, name=None, reuse=None, scale = 1.0):
    def three_affineLayer(x, is_training=True):
        with tf.variable_op_scope([x], name, 'Affine', reuse=reuse):
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.random_uniform_initializer(-Option.fixed_limit, Option.fixed_limit))
            w = W(w, scale)

            if bitsA <= 16:
                x = A(reshaped)
            if bitsE <= 16:
                x = E(x)
            output = tf.matmul(x, w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return three_affineLayer

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
