# Hardware-Accelerated Communication Model-Serving System
This repo presents a hardware-accelerated communication framework for model serving systems. 

## Summary
Hardware-accelerated communication enables lower latency and higher throughput by skipping the kernel TCP/IP stack. In this case, remote clients can directly read and write server's memory addresses. This repo contains a reference implementation for hardware-accelerated model serving. The current implementation supports two types of hardware-accelerated communication (RDMA and GPUDirect RDMA) and a TCP-based model for reference. We implement the model serving pipeline using OpenCV and TensortRT. The implementation uses:
* NVIDIA OFED v5.6
* CUDA v11.6.2
* OpenCV v4.5.5
* TensorRT v8.4
* ZeroMQ V2.1

## Paper
To refer to the paper or the results. Please use the following citation.

```
@inproceedings{hanafy_iwqos23,
    author={Hanafy, Walid and Wang, Limin and Chang, Hyunseok and Mukherjee, Sarit  and Lakshman, T. V.  and Shenoy, Prashant},
    booktitle={2023 IEEE/ACM 31st International Symposium on Quality of Service (IWQoS)},
    title={Understanding the Benefits of Hardware-Accelerated Communication in Model-Serving Applications},
    year={2023},
    pdf = {https://lass.cs.umass.edu/papers/pdf/Walidiwqos2023.pdf},
}
```

## Project Structure
```
tree .
.
├── LICENCE
├── README.md # Documents
├── opencv_build.sh # Build Open CV
└── src # Source Code
    └── models # Code and Folder fo models
        ├── get_onnx.py # Download ONNX
        └── imagenet_classes.txt # names for classes
    ...
```

## Hardware Requirements
This project requires RDMA-capable NIC (RNIC) and GPUDirect-capable GPUs. The code was tested on ConnextX-5 25GbE RNIC and NVIDIA A2 GPU.
## Installation
### Install Dependencies
The code depends on many libraries as follows:
#### CUDA
Instructions are available [here][CUDA].
#### OFED
Instructions are available [here][OFED]. Also you should make sure to load the `nvidia-peermem` kernel module.

```bash
sudo modprobe nvidia-peermem
```

#### CUDNN
 Instructions are available [here][cudnn].

#### TensorRT
TensorRT: Instructions are available [here][tensorRT].
#### OpenCV
Server side needs OpenCV with CUDA. Use the following script to build OpenCV with CUDA.

```bash
./opencv_build.sh
```
#### ZeroMQ
TensorRT: Instructions are available [here][zeroMQ].
### Build Code
```bash
cd src
cmake build . -D BUILD_TARGET=ALL
make
```
The current model supports the following build flags
* `BUILD_TARGET=ALL`: (Default) BUILDs all project executables
* `BUILD_TARGET=RDMA_ALL`: BUILD RDMA Client, Server, and Proxy
* `BUILD_TARGET=RDMA_SERVER`: BUILD RDMA Server
* `BUILD_TARGET=RDMA_CLIENT`: BUILD RDMA Client
* `BUILD_TARGET=PROXY_RDMA`: BUILD RDMA Proxy
* `BUILD_TARGET=ZMQ_ALL`: BUILD ZMQ Client, Server, and Proxy
* `BUILD_TARGET=ZMQ_SERVER`: BUILD ZMQ Server
* `BUILD_TARGET=ZMQ_CLIENT`: BUILD ZMQ Client
* `BUILD_TARGET=PROXY_ZMQ`: BUILD ZMQ PROXY, which requires support for RDMA and ZMQ

## Applications
In all cases you will need a DNN module. We use TensorRT. The following script downloads a pretrained model in onnx format. Then we use `trtexec` to compile it into the runnable format.
```bash
cd src/models
python3 get_onnx.py # download a sample model
/usr/src/tensorrt/bin/trtexec --onnx=resnet50.onnx --saveEngine=resnet50.trt --explicitBatch --useCudaGraph # convert it to TRT engine
```

### RDMA Client
```bash
./rdma_client -h

RDMA Client
Usage: ./rdma_client [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  --host TEXT [7.7.7.2]       RDMA Host
  -p,--port INT [20079]       RDMA Server port
  -d,--debug [0]              USE DEBUG
  -r,--requests INT [1]       Total Requests
  -t,--task ENUM [0]          Task
  -i,--image TEXT [dog.jpg]   Image
  -s,--save-logs [0]          Save Logs
  -f,--folder TEXT [results]  Logs Folder
  --client-id INT [0]         Client ID
  --model-name TEXT [resnet50] 
                              Model Name
  -c,--total-clients INT [1]  Total Clients
```
### RDMA Server
```bash
./rdma_server -h
RDMA Server
Usage: ./rdma_server [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -p,--port INT [20079]       RDMA Listen port
  -a,--addressing ENUM [0]    Memory Addressing
  -d,--debug [0]              USE DEBUG
  -k,--keep-alive BOOLEAN [1] 
                              Keep Alive
  -i,--monitor-interval INT [20] 
                              Monitor Interval
  -f,--folder TEXT [results]  Logs Folder
  --classes-file TEXT [models/imagenet_classes.txt] 
                              Classes File
  --engine-file TEXT [models/resnet50.trt] 
                              Tensor RT Engine File
  --streams INT [10]          Tensor RT Streams
  --priority-clients INT [0]  High Priority Client
```
### ZMQ Client
```bash
./zmq_client -h
ZMQ Client
Usage: ./zmq_client [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  --host TEXT [7.7.7.2]       RDMA Host
  --port INT [5555]           ZMQ Server port
  -i,--image TEXT [dog.jpg]   Image
  -d,--debug [0]              USE DEBUG
  -r,--requests INT [1]       Total Requests
  --client-id INT [0]         Client ID
  -t,--task ENUM [0]          Task
  -s,--save-logs [0]          Save Logs
  -f,--folder TEXT [results]  Logs Folder
  --model-name TEXT [resnet50] 
                              Model Name
```
#### ZMQ Server
```bash
/zmq_server -h
ZMQ Server
Usage: ./zmq_server [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  --port INT [5555]           ZMQ Listen port
  -d,--debug [0]              USE DEBUG
  --classes-file TEXT [models/imagenet_classes.txt] 
                              Classes File
  --engine-file TEXT [models/resnet50.trt] 
                              Tensor RT Engine File
  --streams INT [0]           Number of Cuda Streams and TRTContexts
  --workers INT [1]           Number of ZMQ Workers
  -f,--folder TEXT [results]  Logs Folder
  -i,--monitor-interval INT [20] 
                              Monitor Interval
  -s,--save-logs [0]          Save Logs
  --priority-clients BOOLEAN [0] 
                              High Priority Client
  --total-requests INT [1000] 
                              Total requests
```

#### RDMA Proxy
```bash
./rdma_proxy -h
RDMA Proxy
Usage: ./rdma_proxy [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  --server-address TEXT [7.7.7.2] 
                              RDMA Server Address
  --target-port INT [20079]   RDMA Target port
  -r,--requests INT [1000000] 
                              Total Requests
  -t,--task ENUM [3]          Task
  -s,--save-logs [0]          Save Logs
  -p,--port INT [20079]       RDMA Listen port
  -d,--debug [0]              USE DEBUG
  -k,--keep-alive BOOLEAN [1] 
                              Keep Alive
  -c,--connections INT [1]    RDMA Connections to Backend
  --model-name TEXT [resnet50] 
                              Model Name

```
#### ZMQ Balancer
The ZMQ balancer forwards TCP request to ZMQ Server
```bash
./zmq_balancer -h
ZMQ Plain Proxy
Usage: ./zmq_balancer [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  --server-address TEXT [7.7.7.2] 
                              Target Server Address
  --server-port INT [5555]     Target Server port
  --local-port INT [5555]     ZMQ Listen port
  -d,--debug [0]              USE DEBUG
  -c,--connections INT [1]    Number of Cuda Streams and TRTContexts
  --model-name TEXT [resnet50] 
                              Model Name

```
#### ZMQ Proxy
The ZMQ balancer forwards TCP request to RDMA Server.
```bash
./zmq_proxy -h
ZMQ Plain Proxy
Usage: ./zmq_proxy [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  --server-address TEXT [7.7.7.2] 
                              RDMA Server Address
  --rdma-port INT [20079]     RDMA Server port
  --zmq-port INT [5555]       ZMQ Listen port
  -d,--debug [0]              USE DEBUG
  -r,--requests INT [1000000] 
                              Total Requests
  -t,--task ENUM [3]          Task
  -s,--save-logs [0]          Save Logs
  -f,--folder TEXT [results]  Logs Folder
  -c,--clients INT [1]        RDMA Clients
  --model-name TEXT [resnet50] 
                              Model Name
```
#### Executor
A python script to run multiple clients.
```bash
python3 multi-client.py -h
usage: multi-client.py [-h] [--host HOST_ADDRESS] [-p HOST_PORT] [-c CLIENTS] [-a APPLICATION_NAME] -t TASK [-r REQUESTS] [-s] [-f RESULTS_FOLDER]
                       [-d] [-m {single,range}] [--model-name MODEL_NAME]

Running Multiple Clients

optional arguments:
  -h, --help            show this help message and exit
  --host HOST_ADDRESS   Host Address
  -p HOST_PORT, --port HOST_PORT
                        Host port
  -c CLIENTS, --clients CLIENTS
                        Number of clients
  -a APPLICATION_NAME, --application-name APPLICATION_NAME
                        Client Application
  -t TASK, --task TASK  Task
  -r REQUESTS, --requests REQUESTS
                        Requests
  -s, --save-logs       Save Logs
  -f RESULTS_FOLDER, --folder RESULTS_FOLDER
                        Results Folder
  -d, --debug-mode      Debug Mode
  -m {single,range}     execution mode
  --model-name MODEL_NAME
                        Model Name
```
## Examples
### RDMA client and Server:
Server:
```bash
./rdma_server -a 0 -d
RDMA device is ready 
....
```
Client:
```bash
./rdma_client --host 7.7.7.1 -t 0 -d
RDMA client is ready 
....
```
### GPUDirect client and Server:
Server:
```bash
./rdma_server -a 1 -d
RDMA device is ready 
....
```
Client:
```bash
./rdma_client --host 7.7.7.1 -t 0 -d
RDMA client is ready 
....
```

### ZMQ client and Server:
Server:
```bash
./zmq_server -d
```
Client:
```bash
./zmq_client --host 7.7.7.1 -t 0
```

### RDMA Proxy
Server:
```bash
./rdma_server -a 0 -d
```
Proxy:
```bash
./rdma_proxy --server-address 7.7.7.1 -p 20078 -d
```
Client:
```bash
./rdma_client --host 7.7.7.1 -t 0 -d
```
### ZMQ Proxy
Server:
```bash
./rdma_server -a 0 -d
```
Proxy:
```bash
./zmq_proxy --server-address 7.7.7.1 -d
```
Client:
```bash
./zmq_client --host 7.7.7.1 -t 1 -d
```
### ZMQ Balancer
Server:
```bash
./zmq_server -d
```
Proxy:
```bash
./zmq_balancer --server-address 7.7.7.1 -d --local-port 4444
```
Client:
```bash
./zmq_client --host 7.7.7.1 -t 1 -d --port 4444
```

## Known Issues
* RDMA requires knowledge about number of messages. We negotiate number of requests before they start.

## Contact Information
This code was part of [Walid A. Hanafy](https://people.cs.umass.edu/~whanafy/) Internship in summer 2022. Please contact him if you have any questions or issues.

email: whanafy(AT)cs(DOT)umass(DOT)edu

[OFED]: https://docs.nvidia.com/networking/display/MLNXOFEDv562090/
[CUDA]: https://developer.nvidia.com/cuda-downloads
[cudnn]: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb
[tensorRT]: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian
[OpenCV]: https://github.com/keaneflynn/RazerBlade14_OpenCVBuild/blob/main/opencv_build.sh
[zeroMQ]: https://zeromq.org/download/
