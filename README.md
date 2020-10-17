# Transformer neural networks for the visual analysis of spatiotemporal ensembles - official implementation
![alt text]()

## Requirements

* Both Linux and Windows are supported.
* 64-bit Python 3.7 installation.
* TensorFlow 2.1, which we used for all experiments in the paper.
* One or more high-end NVIDIA GPUs, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5. To reproduce the results reported in the paper, you need an NVIDIA GPU with at least 12 GB of DRAM.

## Examples

Get started by running the gauss kernel example [here](https://github.com/Kecksbox/TransformerVIS/blob/master/src/Examples/GaussKernelTest.py). 
You will need to download the corresponding dataset from [here](https://drive.google.com/file/d/1u0IXJhkeLjhZRXanoJMrcCJPsDTw5x2l/view) and copy it to ./src/Examples/Datasets. The first run should then lie under ./src/Examples/Datasets/ensemble1_np/Sim_1.npy.
