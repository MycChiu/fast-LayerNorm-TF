#!/usr/bin/env python
from layer_norm_fused_layer import layer_norm_custom
from tensorflow.contrib.layers import layer_norm
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import utils

from timeit import default_timer

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

nb_layers = 5
nb_epoch = 10
itrs = 100


def custom_vanilla(inputs, scale=False, center=False):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    dtype = inputs.dtype.base_dtype
    out = inputs
    if scale:
        gamma_collections = utils.get_variable_collections(None,
                                                           'gamma')
        gamma = variables.model_variable(
            'gamma',
            shape=params_shape,
            dtype=dtype,
            initializer=init_ops.ones_initializer,
            collections=gamma_collections,
            trainable=True)
        out = out * gamma
    if center:
        beta_collections = utils.get_variable_collections(None,
                                                          'beta')
        beta = variables.model_variable('beta',
                                        shape=params_shape,
                                        dtype=dtype,
                                        initializer=init_ops.zeros_initializer,
                                        collections=beta_collections,
                                        trainable=True)
        out = out + beta
    return out

# def custom_layer_norm(inputs, scale=False, center=False):
#     return cMod.layer_norm_custom(inputs, epsilon=1e-12)


def benchmark(norm_fn, batch_size=128, nb_units=128):
    epo_times = np.zeros([nb_epoch])
    with tf.Graph().as_default():
        # fake data
        x = tf.random_uniform([batch_size, 30])
        y = tf.random_uniform([batch_size], maxval=10)
        labels = tf.to_int32(y)
        # define graph, need to pass scale argument since batch_norm's default
        # option for scale is False while that of layer_norm is True.
        out = slim.repeat(x, nb_layers, slim.fully_connected, nb_units,
                          # normalizer_fn=layer_norm,
                          normalizer_fn=norm_fn,
                          normalizer_params={"scale": False, "center": True},
                          biases_initializer=None,
                          scope="fc")
        # calculate loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(out, labels)
        # define optimizer
        optimizer = tf.train.AdamOptimizer(1e-3)
        # create train_op
        train_op = optimizer.minimize(loss)
        # initialization op
        # init = tf.global_variables_initializer()
        init = tf.initialize_all_variables()

        with tf.Session("") as sess:
            sess.run(init)
            # warm up
            for i in range(itrs):
                sess.run(train_op)
            # start benchmark
            for e in range(nb_epoch):
                epo_start = default_timer()
                for i in range(itrs):
                    sess.run(train_op)
                eT = (default_timer() - epo_start)
                epo_times[e] = eT
            return epo_times


params = [
    ["NoNorm", custom_vanilla],
    ["BatchNorm", slim.batch_norm],
    ["LayerNormBuiltIn", layer_norm],
    ["LayerNormCustom", layer_norm_custom]
]
# for _nb_units in [32,64,128,256,512,1024]:

all_dfs = []
for _nb_units in [32, 64, 128, 256, 512, 1024]:
    # data = []
    _data = benchmark(params[0][1], nb_units=_nb_units)
    mean = _data.mean()
    for param in params:
        _data = benchmark(param[1], nb_units=_nb_units)
        df = pd.DataFrame(_data / mean, columns=["runtime_ratio"])
        df["Norm_fn"] = param[0]
        df["nb_units"] = _nb_units
        all_dfs.append(df)
pan_df = pd.concat(all_dfs)
ax = sns.pointplot(x="nb_units", y="runtime_ratio",
                   hue="Norm_fn", data=pan_df)
plt.savefig("benchmark_ratio_nb_unit.png")

# all_dfs = []
# # for _batch_size in [256, 512]:
# for _batch_size in [256, 512, 1024, 4096, 8192, 16384]:
#     _data = benchmark(params[0][1], batch_size=_batch_size)
#     mean = _data.mean()
#     for param in params:
#         _data = benchmark(param[1], batch_size=_batch_size)
#         df = pd.DataFrame(_data / mean, columns=["runtime_ratio"])
#         df["Norm_fn"] = param[0]
#         df["batch_size"] = _batch_size
#         all_dfs.append(df)
# pan_df = pd.concat(all_dfs)
# ax = sns.pointplot(x="batch_size", y="runtime_ratio",
#                    hue="Norm_fn", data=pan_df)
# plt.savefig("benchmark_ratio_batch_size.png")
# vanilla_t = benchmark(custom_vanilla)
# batchNorm_t = benchmark(slim.batch_norm)
# layerNorm_t = benchmark(layer_norm)

# print("vanilla times in seconds:")
# print(vanilla_t)
# print("batch_norm times in seconds:")
# print(batchNorm_t)
# print("layer_norm times in seconds:")
# print(layerNorm_t)
# print("custom_layer_norm times in seconds:")
# print(customLayerNorm_t)
# print("batch_norm is %1.2fX slower" %
#       (batchNorm_t.sum() / vanilla_t.sum()))
# print("layer_norm is %1.2fX slower" %
#       (layerNorm_t.sum() / vanilla_t.sum()))
# print("custom_layer_norm is %1.2fX slower" %
#       (customLayerNorm_t.sum() / vanilla_t.sum()))
