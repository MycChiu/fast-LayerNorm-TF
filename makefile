TF_INC :=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
INC=-I${TF_INC}
layer_norm_fused_op: register_ops.cc layer_norm_fused_op.h layer_norm_fused_op.cc layer_norm_fused_grad_op.cc layer_norm_fused_op_gpu.cu.cc
	nvcc -std=c++11 -c -o layer_norm_fused_op_gpu.cu.o layer_norm_fused_op_gpu.cu.cc \
	$(INC) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -gencode=arch=compute_35,code=sm_35 \
	-gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 \
	-gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61
	g++ -std=c++11 -shared \
	-D_GLIBCXX_USE_CXX11_ABI=0 \
	-o layer_norm_fused_op.so register_ops.cc layer_norm_fused_op.h \
	layer_norm_fused_grad_op.cc layer_norm_fused_op.cc layer_norm_fused_op_gpu.cu.o \
	$(INC) -L /usr/local/cuda/lib64/ -fPIC -lcudart -O2 -DNDEBUG
