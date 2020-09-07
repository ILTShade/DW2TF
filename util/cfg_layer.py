# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# # Syntax
# def cfg_layerName(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
#     pass


def cfg_net(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    width = int(param["width"])
    height = int(param["height"])
    channels = int(param["channels"])
    with tf.compat.v1.variable_scope(scope):
        output = tf.compat.v1.placeholder(tf.float32, [1, width, height, channels])
    return output

def cfg_convolutional(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    batch_normalize = ('batch_normalize' in param.keys())
    size = int(param['size'])
    filters = int(param['filters'])
    stride = int(param['stride'])
    if param['pad'] == '0':
        pad = 'VALID'
    elif int(param['pad']) == 1 or int(param['pad']) * 2 + 1 == size:
        pad = 'SAME'
    else:
        raise Exception(f'can not support {param["pad"]}')
    activation = param['activation']
    # load weights
    weight_size = C * filters * size * size
    biases, scales, rolling_mean, rolling_variance, weights = \
        weights_walker.get_weight(
            param['name'],
            filters = filters,
            weight_size = weight_size,
            batch_normalize = batch_normalize
        )
    weights = weights.reshape(filters, C, size, size).transpose([2, 3, 1, 0])
    # add conv
    with tf.compat.v1.variable_scope(scope):
        weight_variable = tf.compat.v1.get_variable(
            name = 'weight', dtype = tf.float32, trainable = True,
            shape = weights.shape, initializer = tf.initializers.constant(weights, verify_shape = True),
        )
        conv = tf.nn.conv2d(input = net, filter = weight_variable, strides = [1, stride, stride, 1], padding = pad)
        # add batchnorm or add bias
        if batch_normalize:
            # without bias and with bn
            conv = tf.layers.batch_normalization(
                conv,
                beta_initializer = tf.initializers.constant(biases, verify_shape = True),
                gamma_initializer = tf.initializers.constant(scales, verify_shape = True),
                moving_mean_initializer = tf.initializers.constant(rolling_mean, verify_shape = True),
                moving_variance_initializer = tf.initializers.constant(rolling_variance, verify_shape = True),
                trainable = True,
            )
        else:
            bias_variable = tf.compat.v1.get_variable(
                name = 'bias',  dtype = tf.float32, trainable = True,
                shape = weights.shape[-1],  initializer =  tf.initializers.constant(biases, verify_shape = True),
            )
            conv = tf.nn.bias_add(conv, bias_variable)
        if activation == 'leaky':
            conv = tf.nn.leaky_relu(conv, alpha = 0.125)
        elif activation == 'relu':
            conv = tf.nn.relu(conv)
    # return
    return conv

def cfg_route(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    if not isinstance(param["layers"], list):
        param["layers"] = [param["layers"]]
    input_index = [int(x) for x in param["layers"]]
    inputs = [stack[x] for x in input_index]
    with tf.compat.v1.variable_scope(scope):
        output = tf.concat(inputs, axis = -1)
    return output

def cfg_shortcut(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    index = int(param['from'])
    if param['activation'] == 'linear':
        with tf.compat.v1.variable_scope(scope):
            from_layer = stack[index]
            output = from_layer + net
    else:
        raise Exception(f'does not support {param["activation"]}')
    return output

def cfg_yolo(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    with tf.compat.v1.variable_scope(scope):
        output_index.append(len(stack) - 1)
    return net


def cfg_upsample(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    stride = int(param['stride'])
    if stride == 2:
        with tf.compat.v1.variable_scope(scope):
            output = tf.compat.v1.image.resize_nearest_neighbor(net, (H * stride, W * stride))
    else:
        raise Exception(f'does not support stride not 2')
    return output

def cfg_ignore(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    if verbose:
        print("=> Ignore: ", param)
    return net

_cfg_layer_dict = {
    "net": cfg_net,
    "convolutional": cfg_convolutional,
    "route": cfg_route,
    "shortcut": cfg_shortcut,
    "yolo": cfg_yolo,
    "upsample": cfg_upsample,
}


def get_cfg_layer(net, layer_name, param, weights_walker, stack, output_index,
    scope = None, training = True, const_inits = True, verbose = True):
    B, H, W, C = [None, None, None, None] if net is None else net.shape.as_list()
    layer = _cfg_layer_dict.get(layer_name, cfg_ignore)(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose)
    return layer
