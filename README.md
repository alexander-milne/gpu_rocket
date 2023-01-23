# gpu_rocket

This repo provides a replica of the multi-channel ROCKET model transformation implemented in PyTorch. The example notebook gives an example of its use and shows that the outputs are identical when the same kernels are loaded. Where GPU is available this can be used to speed up the model by a significant margin depending on the amount of data (batchsize and data size).

Since we have implemented in PyTorch we have provided a soft valiant of PPV using sigmoid. 

More details available: https://arxiv.org/abs/2301.08527
