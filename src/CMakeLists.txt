cmake_minimum_required(VERSION 3.16)
project(RDMAServe VERSION 0.1)

# Variables
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g3")
set(CMAKE_CXX_COMPILER g++)

option(BUILD_TARGET "Compile applications" ALL)


message(STATUS "BUILD_TARGET ${BUILD_TARGET}")

if(BUILD_TARGET STREQUAL "ALL" OR BUILD_TARGET MATCHES "(RDMA_SERVER|RDMA_ALL|ZMQ_SERVER|ZMQ_ALL)")
        message(STATUS "Including CUDA")
        enable_language(CUDA)
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
        set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
        find_package(CUDA REQUIRED)
endif()

if(BUILD_TARGET STREQUAL "ALL" OR BUILD_TARGET MATCHES "(RDMA_SERVER|RDMA_ALL|ZMQ_SERVER|ZMQ_ALL)")
        set(TensorRT_DIR /usr/src/tensorrt/)
        set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
        find_package(TensorRT REQUIRED)
endif()

find_package(OpenCV REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
include_directories(.)

# DNN Computation
if(BUILD_TARGET STREQUAL "ALL" OR BUILD_TARGET MATCHES "(SERVER|ALL)$")
        message(STATUS "Including compute engines")
        add_library(dnn_compute SHARED computation.cpp trt_engine.cpp)
        target_include_directories(dnn_compute PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
        target_link_libraries(dnn_compute PUBLIC cuda ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
endif()

# RDMA Commmon
if(BUILD_TARGET STREQUAL "ALL" OR BUILD_TARGET MATCHES "(^RDMA|PROXY|ALL$)")
        message(STATUS "Including rdma common")
        find_library(IBVERBS_LIBRARY ibverbs)
        find_library(RDMACM_LIBRARY rdmacm)
        add_library(rdma_common SHARED rdma_common.cpp)
        target_link_libraries(rdma_common PUBLIC rdmacm ibverbs)
endif()

if(BUILD_TARGET STREQUAL "ALL" OR BUILD_TARGET MATCHES "(RDMA_ALL|RDMA_SERVER)")
        message(STATUS "Including rdma server")

        # RDMA Server
        add_executable(rdma_server rdma_server.cpp)
        target_link_libraries(rdma_server rdma_common dnn_compute rdmacm ibverbs cuda Threads::Threads)
endif()

# RDMA Client
if(BUILD_TARGET STREQUAL "ALL" OR BUILD_TARGET MATCHES "(RDMA_ALL|RDMA_CLIENT)")
        message(STATUS "Including rdma client")
        add_executable(rdma_client rdma_client.cpp)
        target_link_libraries(rdma_client rdma_common rdmacm ibverbs ${OpenCV_LIBS})
endif()

# RDMA Proxy
if(BUILD_TARGET STREQUAL "ALL" OR BUILD_TARGET MATCHES "(RDMA_ALL|RDMA_PROXY)")
        message(STATUS "Including rdma proxy")
        add_executable(rdma_proxy rdma_proxy.cpp)
        target_link_libraries(rdma_proxy rdma_common rdmacm ibverbs Threads::Threads)
endif()

# ZMQ Server
if(BUILD_TARGET STREQUAL "ALL" OR BUILD_TARGET MATCHES "(ZMQ_ALL|ZMQ_SERVER)")
        message(STATUS "Including zmq server")
        add_executable(zmq_server zmq_server.cpp ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(zmq_server dnn_compute zmq
                ${OpenCV_LIBS}
                Threads::Threads)
endif()

# ZMQ Client
if(BUILD_TARGET STREQUAL "ALL" OR BUILD_TARGET MATCHES "(ZMQ_ALL|ZMQ_CLIENT)")
        message(STATUS "Including zmq client")
        add_executable(zmq_client zmq_client.cpp ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(zmq_client zmq
        ${OpenCV_LIBS})
endif()

# ZMQ TCP/RDMA
if(BUILD_TARGET STREQUAL "ALL" OR BUILD_TARGET MATCHES "(PROXY_ZMQ|ZMQ_ALL)")
        message(STATUS "Including zmq TCP/RDMA proxy")
        add_executable(zmq_proxy zmq_proxy.cpp)
        target_link_libraries(zmq_proxy zmq rdma_common rdmacm ibverbs Threads::Threads)
endif()

# ZMQ TCP/TCP
if(BUILD_TARGET STREQUAL "ALL" OR BUILD_TARGET MATCHES "(PROXY_ZMQ|ZMQ_ALL)")
        message(STATUS "Including zmq TCP/TCP proxy")
        add_executable(zmq_balancer zmq_balancer.cpp)
        target_link_libraries(zmq_balancer zmq Threads::Threads)
endif()
