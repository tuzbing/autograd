# autograd

autograd is a simple framework for training LLM models(like GPT2/3/4, or MoE supported LLM etc). It will try to implemented with low memory usage and trainning time. It also try to support multiple backend devices, like CPU, Metal, CUDA.

Design Philosophy
    To maximize performance, during the compilation of the computation graph(DAG) constructed from tensors, all tensor operations are bound to kernel functions provided by the target backend device.
	At the same time, all memory required for the training process is pre-allocated in advance, aoviding frequent memory allocation and deallocation during training.

Requirements
	tiktoken
	pyarrow
	pybind11

Run
a) compile metal
	python setup.py build_ext --inplace
b) train gpt2
	python test_train_gpt2.py
