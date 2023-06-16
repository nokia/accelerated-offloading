/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Tensor RT Execution Engine
 *
 **/
#pragma once
#include "cuda.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include "NvInfer.h"
#include "utils.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <mutex>
#include <condition_variable>


/// @brief Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override;
};

/// @brief Engine Stream Class
/// A wrapper over cuda stream which holds extra monitoring properties
class EngineStream
{
public:
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    cv::cuda::Stream cv_stream = cv::cuda::Stream::Null();
    cudaStream_t m_cudaStream = nullptr;
    std::vector<void *> stream_buffers;
    cudaEvent_t pre_copy_start, pre_copy_stop;
    cudaEvent_t preprocess_start, preprocess_stop;
    cudaEvent_t mid_copy_start, mid_copy_stop;
    cudaEvent_t inference_start, inference_stop;
    cudaEvent_t post_copy_start, post_copy_stop;

    EngineStream()
    {
        cudaEventCreate(&pre_copy_start);
        cudaEventCreate(&pre_copy_stop);
        cudaEventCreate(&preprocess_start);
        cudaEventCreate(&preprocess_stop);
        cudaEventCreate(&mid_copy_start);
        cudaEventCreate(&mid_copy_stop);
        cudaEventCreate(&inference_start);
        cudaEventCreate(&inference_stop);
        cudaEventCreate(&post_copy_start);
        cudaEventCreate(&post_copy_stop);
    }
};


/// @brief TensorRT Engine Manager
class Engine
{
public:
    /// @brief Constructor creates GPU Buffers
    ///
    /// @param engine_path Path to TensorRT Engine
    /// @param classes_file File Where Classes Name is added
    Engine(std::string engine_path, std::string classes_file, int high_priority_clients, int streams = 1);
    ~Engine();
    /*
    / Run Inference Only Assuming Data is allocated on the GPU
    \param postProcess - Do Post Processing
    * \return
    * int
    * 0 Success
    * 1 Failed
    */
    int runInferenceOnly(std::vector<void *> buffers, int stream_id, bool postProcess);
    /// Preprocess Image on the GPU
    int preprocess_image_GPU(cv::cuda::GpuMat image, std::vector<void *> buffers, int stream_id);
    
    /// @brief Get Stream Specific Buffers
    /// @param stream_id 
    /// @return 
    std::vector<void *> get_stream_buffers(int stream_id);
    
    /// @brief Get Input and Output buffers size
    /// @return 
    int get_buffers_size();

    /// @brief Map Connection to Stream Id
    /// @return 
    int get_stream_id();
    
    /// @brief Release Stream after usage
    /// @param stream_id 
    void release_stream(int stream_id);
    /*
    Allocated Memory Buffers based on Input/OutPut Size
    */
    std::vector<void *> allocate_buffers();
    std::vector<nvinfer1::Dims> input_dims;  // we expect only one input
    std::vector<nvinfer1::Dims> output_dims; // and one output
    std::vector<EngineStream> engine_streams;
    size_t total_output_size  = 0;
    int high_priority_clients;
private:
    ///
    /// Load and warmup the network for inference
    /// @return
    /// int
    /// 0 Success
    /// 1 Failed
    ///
    int loadNetwork();

    /// @brief Post Process Outputs and print Top 5 classes
    /// @param buffers 
    /// @param stream_id 
    /// @param compute_confidence 
    void post_process_results_vision_classification(std::vector<void *> buffers, int stream_id, bool compute_confidence);
    
    /// @brief Post Process Detection Output
    /// @param buffers 
    /// @param stream_id 
    void post_process_results_vision_detection(std::vector<void *> buffers, int stream_id);

    /// @brief Load TRT binding info
    /// @return 
    int load_binding_info();

    /// @brief Get Input Image Size
    /// @return 
    cv::Size get_image_size();
    int total_streams;
    ModelType model_type;
    std::string classes_file, engine_path;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;

    Logger m_logger;

    std::mutex m;
    std::vector<bool> busy;
    std::condition_variable _cv;
};

/// @brief Get Total Size based on Dimensions
/// @param dims 
/// @return 
int32_t getSizeByDim(const nvinfer1::Dims &dims);