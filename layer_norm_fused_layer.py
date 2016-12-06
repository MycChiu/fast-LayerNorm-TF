from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import common_shapes

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope

import tensorflow as tf

cMod = tf.load_op_library(
    'layer_norm_fused_op.so')

# disabled these if using newer version of Tensorflow. (You can keep this
# if no error raised)
tf.RegisterShape("LayerNormCustom")(common_shapes.call_cpp_shape_fn)
tf.RegisterShape("LayerNormBiasAddCustom")(common_shapes.call_cpp_shape_fn)
tf.RegisterShape("LayerNormFusedCustom")(common_shapes.call_cpp_shape_fn)


@ops.RegisterGradient("LayerNormCustom")
def _LayerNormCustomGrad(op, grad):
    return [cMod.layer_norm_backprop_custom(
        op.inputs[0], grad, op.get_attr("epsilon"))]


@ops.RegisterGradient("LayerNormBiasAddCustom")
def _LayerNormBiasAddCustomGrad(op, grad):
    in_back, beta_back = cMod.layer_norm_bias_add_backprop_custom(
        op.inputs[0], grad, op.inputs[1],
        op.get_attr("epsilon"))
    return [in_back, beta_back]


@ops.RegisterGradient("LayerNormFusedCustom")
def _LayerNormFusedCustomGrad(op, grad):
    in_back, gamma_back, beta_back = cMod.layer_norm_fused_backprop_custom(
        op.inputs[0], grad, op.inputs[1],
        op.get_attr("epsilon"))
    return [in_back, gamma_back, beta_back]


# adapted from tf.contrib.layers.layer_norm
@add_arg_scope
def layer_norm_custom(inputs,
                      center=True,
                      scale=True,
                      activation_fn=None,
                      reuse=None,
                      variables_collections=None,
                      outputs_collections=None,
                      trainable=True,
                      epsilon=1E-12,
                      scope=None):
    """Adds a Layer Normalization layer from https://arxiv.org/abs/1607.06450.
      "Layer Normalization"
      Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    Can be used as a normalizer function for conv2d and fully_connected.
    Args:
      inputs: a tensor with 2 or more dimensions. The normalization
              occurs over all but the first dimension.
      center: If True, subtract `beta`. If False, `beta` is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is
        not used. When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling can be done by the next layer.
      activation_fn: activation function, default set to None to skip it and
        maintain a linear activation.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: optional collections for the variables.
      outputs_collections: collections to add the outputs.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      epsilon: small value added to prevent NaN outputs.
      scope: Optional scope for `variable_scope`.
    Returns:
      A `Tensor` representing the output of the operation.
    Raises:
      ValueError: if rank or last dimension of `inputs` is undefined.
    """
    with variable_scope.variable_scope(scope, 'LayerNorm', [inputs],
                                       reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        params_shape = inputs_shape[-1:]
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' % (
                inputs.name, params_shape))
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta_collections = utils.get_variable_collections(variables_collections,
                                                              'beta')
            beta = variables.model_variable('beta',
                                            shape=params_shape,
                                            dtype=dtype,
                                            initializer=init_ops.zeros_initializer,
                                            collections=beta_collections,
                                            trainable=trainable)
        if scale:
            gamma_collections = utils.get_variable_collections(variables_collections,
                                                               'gamma')
            gamma = variables.model_variable(
                'gamma',
                shape=params_shape,
                dtype=dtype,
                initializer=init_ops.ones_initializer,
                collections=gamma_collections,
                trainable=trainable)

        variance_epsilon = epsilon
        if epsilon <= 0:
            print("WARNING: epsilon <=0, may result in NaN outputs.")
        if center and scale:
            outputs = cMod.layer_norm_fused_custom(
                inputs, gamma, beta, epsilon=variance_epsilon)
        elif center:
            outputs = cMod.layer_norm_bias_add_custom(
                inputs, beta, epsilon=variance_epsilon)
        elif scale:
            # dummy constant beta for layer_norm_fused_custom()
            beta = tf.zeros(params_shape, dtype=dtype,
                            name="dummy_beta")
            outputs = cMod.layer_norm_fused_custom(
                inputs, gamma, beta, epsilon=variance_epsilon)
        else:
            outputs = cMod.layer_norm_custom(inputs, epsilon=variance_epsilon)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope,
                                           outputs)
