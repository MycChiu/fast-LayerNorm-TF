Fast Tensorflow Layer Normalization GPU kernel
====
![comparing built-in and custom]
(https://github.com/MycChiu/fast-LayerNorm-TF/blob/master/images/nvvp_comparison.png)
*Kernel profile produced in [NVIDIA Visual Profiler](https://developer.nvidia.com/nvidia-visual-profiler), with input shape of [16,1024,256].*

**Layer normalization** ([Jimmy Lei Ba et al.](https://arxiv.org/abs/1607.06450))is a technique used to prevent "covariate-shift" which in terms reduces the number of batches needed to reach convergence, and in some cases improves the performance of a model.However, the current implementation of layer_norm in Tensorflow will increase the clock-time required per batch dramatically. This is a result of computing mean and variance seperately through multiple steps, with the current architecture of NVIDIA's GPU, reading and writing to global memory (on the GPU device) is quite costly. This is unavoidable for batch normalization, since we would have to keep the running mean and variance for the test time inference. However, layer normalization does not have this constraint, we can lump all the computations together with single read and write to the global memory, which is why this custom kernel is so much faster than the current implementation.

Here are some benchmarks for 5 layers of fully-connected layers using different normalization methods. *Generated with `layer_norm_fused_bench_mark.py`*
![benchmark with different nb_units](https://github.com/MycChiu/fast-LayerNorm-TF/blob/master/images/benchmark_ratio_nb_unit.png)
Batch size fixed to 128 with different nb_units.
![benchmark with different batch_size](https://github.com/MycChiu/fast-LayerNorm-TF/blob/master/images/benchmark_ratio_batch_size.png)
Number of units fixed to 128 with different batch size.

Instructions
====
In most cases, you can just run `make` in the root directory, and the make file will produce `layer_norm_fused_op.so` in the same folder. 

####Notes
1. The `makefile` assumes your cuda library is installed in `/usr/local/cuda`, if you installed it somewhere else, you can change the part `-L /usr/local/cuda/lib64/` in the last line of the `makefile` to `-L [your cuda install path]/lib64/`
2. By default, `nvcc` will compile the kernel for compute capability 6.1, you should change the `-arch=sm_61` at the end of line 5 in `makefile` to match the compute capability of your card.For example, GTX980's compute capability is 5.2, so the argument should be `-arch=sm_52`. You can check the compute capability of your card [here](https://developer.nvidia.com/cuda-gpus).

after the custom library is successfully compiled,You can just copy `layer_norm_fused_op.so` to where you want to use it and load it like the following:
```python3
import tensorflow as tf
from tensorflow.python.framework import common_shapes

#loading the custom op library
custom_module = tf.load_op_library('layer_norm_fused_op.so')

#This line is needed so Tensorflow can infer the shape of the output.
tf.RegisterShape("LayerNormCustom")(common_shapes.call_cpp_shape_fn)

#register gradients for auto-differentiation.
@ops.RegisterGradient("LayerNormCustom")
def _LayerNormCustomGrad(op, grad):
    return [cMod.layer_norm_backprop_custom(
        op.inputs[0], grad, op.get_attr("epsilon"))]

input_shape = [32,512,128]
inputs = tf.random_normal(input_shape)
normalized_output = custom_module.layer_norm_custom(inputs, epsilon=variance_epsilon)
#do what ever you want next...
```
Or you can use the `layer_norm_custom` layer I adapted from the built-in `tf.contrib.layers.layer_norm` within `layer_norm_fused_layer.py`.

There are three diffrenet kernels in this code, they are `layer_norm_custom`,`layer_norm_bias_add_custom`, and `layer_norm_fused_custom`. Take a look at `layer_norm_fused_layer.py` to see how they can be used.

Caveats
----
1. This implementation uses warp shuffle instructions to reduce shared memory access, which (I think) only exists after Kepler (Geforce 600 series), so you will need to modify the code to use on Fermi or older cards. 

2. The performance may differ with different hardware, I only optimized the code for the card I am using (GTX1070). You can use `layer_norm_bench_mark.py` to check if it really is faster with your hardware, and `layer_norm_fused_test.py` to test for validity of the outputs.

3. *This implementation is not exactly the same as `tf.contrib.layers.layer_norm`*. This custom kernel normalize along the last dimension, while the built-in implementation normalize along all dimensions except the first. This will probably not affect standard usage for rnn and fully-connected layers, but it will be diffrenet for 1D or 2D convolutions.

4. The current implementation of this kernel has a limit on the size of your last dimension. More specifically,it can't be more than 5120, which should be more than enough for most use cases, but if you need to increase this limit, please submit an issue, and I will write additional instruction on how to increase this limit.

5. I am really new to CUDA and C++, so the code is far from optimized. Any suggestion on how to improve the kernel is deeply appreciated.

6. If you have any question regarding this kernel, feel free to submit an issue. I will do my best to answer them.