TF_INC :=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
INC=-I${TF_INC}
layer_norm_fused_op: register_ops.cc layer_norm_fused_op.h layer_norm_fused_op.cc layer_norm_fused_grad_op.cc layer_norm_fused_op_gpu.cu.cc
	nvcc -std=c++11 -c -o layer_norm_fused_op_gpu.cu.o layer_norm_fused_op_gpu.cu.cc \
	$(INC) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -arch=sm_61
	g++ -std=c++11 -shared -o layer_norm_fused_op.so register_ops.cc layer_norm_fused_op.h \
	layer_norm_fused_grad_op.cc layer_norm_fused_op.cc layer_norm_fused_op_gpu.cu.o \
	$(INC) -L /usr/local/cuda/lib64/ -fPIC -lcudart -O2 -DNDEBUG