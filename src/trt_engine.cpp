/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Tensor RT Execution Engine
 *
 **/
#include "trt_engine.h"

extern int debug;

#define DEBUG_LOG \
    if (debug)    \
    printf

void Logger::log(Severity severity, const char *msg) noexcept
{
    // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
    // For the sake of this tutorial, will just log to the console.

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING)
    {
        std::cout << msg << std::endl;
    }
}

int32_t getSizeByDim(const nvinfer1::Dims &dims)
{
    int32_t size = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

int Engine::get_stream_id()
{
    while (true)
    {
        {
            // Make the lock in its own scope to release when exit
            std::lock_guard<std::mutex> lg{m};
            for (int i = 0; i < total_streams; i++)
            {
                if (!busy[i])
                {
                    busy[i] = true;
                    _cv.notify_one();
                    return i;
                }
            }
        }
        std::unique_lock<std::mutex> ul{m};
        _cv.wait(ul, []
                 { return true; });
    }
    return -1;
}


void Engine::release_stream(int stream_id)
{
    std::unique_lock<std::mutex> ul{m};
    busy[stream_id] = false;
    ul.unlock();
    _cv.notify_one();
}

Engine::Engine(std::string engine_path, std::string classes_file, int high_priority_clients, int streams)
{
    cudaSetDevice(0);
    this->classes_file = classes_file;
    this->engine_path = engine_path;
    this->total_streams = streams;
    this->high_priority_clients = high_priority_clients;
    if (engine_path.find("net") != std::string::npos)
    {
        this->model_type = ModelType::PyTorchVisionClassification;
    }
    else if (engine_path.find("yolov4") != std::string::npos)
    {
        this->model_type = ModelType::TensorFlowVisionDetection;
    }
    else if (engine_path.find("deeplab") != std::string::npos)
    {
        this->model_type = ModelType::PyTorchVisionSegmentation;
    }
    else
    {
        throw std::runtime_error("Unknown model type");
    }

    busy = std::vector<bool>(streams);

    for (int i = 0; i < streams; i++)
    {
        engine_streams.emplace_back(EngineStream());
        auto priority = 0;
        if (i < high_priority_clients)
        {
            std::cout << "Stream (" << i << ") has high priority"<< std::endl;
            priority = -5;
        }
        //        auto cudaRet = cudaStreamCreate(&engine_streams[i].m_cudaStream);
        auto cudaRet = cudaStreamCreateWithPriority(&engine_streams[i].m_cudaStream, cudaStreamNonBlocking, priority);

        // auto cudaRet = cudaStreamCreateWithFlags(&m_cudaStream, cudaStreamNonBlocking);
        if (cudaRet != 0)
        {
            throw std::runtime_error("Unable to create cuda stream");
        }
        engine_streams[i].cv_stream = cv::cuda::StreamAccessor::wrapStream(engine_streams[i].m_cudaStream);
    }

    if (this->loadNetwork())
    {
        throw std::runtime_error("Unable to Load Network");
    }
}

Engine::~Engine()
{
    std::cout << "Deconstructor Called \n";
    for (int i = 0; i < total_streams; i++)
    {
        if (engine_streams[i].m_cudaStream)
        {
            cudaStreamDestroy(engine_streams[i].m_cudaStream);
        }
        for (void *buf : engine_streams[i].stream_buffers)
            cudaFree(buf);
    }
}

int Engine::loadNetwork()
{
    // Step 1: Read the serialized model from disk
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    std::streamsize modelSize = file.tellg();
    file.seekg(0, std::ios::beg);

    void *modelMem = malloc(modelSize);
    if (!modelMem)
    {
        std::cerr << "\n[tensorrt-time] Failed to allocate memory to deserialize model!\n";
        exit(EXIT_FAILURE);
    }

    if (!file.read((char *)modelMem, modelSize))
    {
        throw std::runtime_error("Unable to read engine file");
    }

    // Step 2: Create RunTime
    std::unique_ptr<nvinfer1::IRuntime> runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(m_logger)};
    if (!runtime)
    {
        return 1;
    }

    // Step 3: Deserialize Engine
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>{runtime->deserializeCudaEngine(modelMem, modelSize)};

    // Step 4 Load Binding Infor
    if (load_binding_info())
        return 1;

    // Step 5: Prepare Needed Data for streams
    // Step 5.1: Create Context (Execution Manager)
    for (int stream_id = 0; stream_id < total_streams; stream_id++)
    {
        engine_streams[stream_id].m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());

        if (!engine_streams[stream_id].m_context)
        {
            return 1;
        }
        engine_streams[stream_id].m_context->setOptimizationProfileAsync(0, engine_streams[stream_id].m_cudaStream);
        // Step 5.2: Allocate Needed Buffers
        engine_streams[stream_id].stream_buffers = allocate_buffers();
        if (engine_streams[stream_id].stream_buffers.size() <= 0)
            return 1;

        // Step 5.3:Warming Up Model
        printf("Warming Model Stream %d \n", stream_id);
        for (int r = 0; r < 10; r++)
        {
            if (runInferenceOnly(engine_streams[stream_id].stream_buffers, stream_id, false))
                return 1;
        }
    }
    return 0;
}

int Engine::get_buffers_size()
{
    return input_dims.size() + output_dims.size();
}

int Engine::load_binding_info()
{
    for (int32_t i = 0; i < m_engine->getNbBindings(); ++i)
    {
        if (m_engine->bindingIsInput(i))
        {
            input_dims.emplace_back(m_engine->getBindingDimensions(i));
        }
        else
        {
            output_dims.emplace_back(m_engine->getBindingDimensions(i));
        }
    }

    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for network\n";
        return 1;
    }
    for (size_t i = 0; i < output_dims.size(); i++)
    {
        total_output_size += getSizeByDim(output_dims[i]);
    }
    return 0;
}

std::vector<void *> Engine::get_stream_buffers(int stream_id)
{
    return engine_streams[stream_id].stream_buffers;
}

std::vector<void *> Engine::allocate_buffers()
{
    auto buffers = std::vector<void *>(); // buffers for input and output data
    for (size_t i = 0; i < input_dims.size(); ++i)
    {
        auto binding_size = getSizeByDim(input_dims[i]) * sizeof(float);
        CUdeviceptr buf_A;
        CUresult cu_result;
        cu_result = cuMemAlloc(&buf_A, binding_size);
        if (cu_result != CUDA_SUCCESS)
        {
            fprintf(stderr, "cuMemAlloc error=%d\n", cu_result);
            return std::vector<void *>();
        }
        buffers.emplace_back((void *)buf_A);
    }
    for (size_t i = 0; i < output_dims.size(); ++i)
    {
        auto binding_size = getSizeByDim(output_dims[i]) * sizeof(float);
        CUdeviceptr buf_A;
        CUresult cu_result;
        cu_result = cuMemAlloc(&buf_A, binding_size);
        if (cu_result != CUDA_SUCCESS)
        {
            fprintf(stderr, "cuMemAlloc error=%d\n", cu_result);
            return std::vector<void *>();
        }
        buffers.emplace_back((void *)buf_A);
    }
    return buffers;
}

int Engine::runInferenceOnly(std::vector<void *> buffers, int stream_id, bool postProcess)
{
    int ret;
    if (debug)
    {
        assert(engine_streams[stream_id].stream_buffers.size() == buffers.size());
    }
    cudaEventRecord(engine_streams[stream_id].inference_start, engine_streams[stream_id].m_cudaStream);

    ret = engine_streams[stream_id].m_context->enqueueV2(buffers.data(), engine_streams[stream_id].m_cudaStream, nullptr);
    if (!ret)
    {
        return 1;
    }
    cudaEventRecord(engine_streams[stream_id].inference_stop, engine_streams[stream_id].m_cudaStream);

    ret = cudaStreamSynchronize(engine_streams[stream_id].m_cudaStream);
    if (ret)
    {
        return 1;
    }
    if (postProcess)
    {
        switch (this->model_type)
        {
        case ModelType::PyTorchVisionClassification:
        case ModelType::TensorFlowVisionClassification:
            post_process_results_vision_classification(buffers, stream_id, postProcess);
            break;
        case ModelType::PyTorchVisionSegmentation:
        case ModelType::TensorFlowVisionDetection:
        case ModelType::PyTorchVisionDetection:
            post_process_results_vision_detection(buffers, stream_id);
            break;
        default:
            fprintf(stderr, "Invalid Case");
            return 1;
        }
    }
    return 0;
}

// get classes names
std::vector<std::string> getClassNames(const std::string &imagenet_classes)
{
    std::ifstream classes_file(imagenet_classes);
    std::vector<std::string> classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with classes names.\n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}

void Engine::post_process_results_vision_classification(std::vector<void *> buffers, int stream_id, bool compute_confidence)
{
    // get class names
    assert(buffers.size() == 2);
    assert(output_dims.size() == 1);

    auto classes = getClassNames(classes_file);
    double sum;

    // copy results from GPU to CPU
    std::vector<float> cpu_output(getSizeByDim(output_dims[0]));
    cudaMemcpyAsync(cpu_output.data(), buffers[1], cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost, engine_streams[stream_id].m_cudaStream);
    cudaStreamSynchronize(engine_streams[stream_id].m_cudaStream);

    if (compute_confidence)
    {
        // calculate softmax
        std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val)
                       { return std::exp(val); });
        sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);
    }
    // find top classes predicted by the model
    std::vector<int> indices(getSizeByDim(output_dims[0]));
    std::iota(indices.begin(), indices.end(), 0); // generate sequence 0, 1, 2, 3, ..., 999
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2)
              { return cpu_output[i1] > cpu_output[i2]; });
    // print Top 3 results
    for (int i = 0; i < 3; i++)
    {
        DEBUG_LOG("class: %s", classes[indices[i]].c_str());
        if (compute_confidence)
            DEBUG_LOG(" | confidence: %f ", 100 * cpu_output[indices[i]] / sum);
        DEBUG_LOG("| index: %d \n", indices[i]);
    }
}

void Engine::post_process_results_vision_detection(std::vector<void *> buffers, int stream_id)
{
    std::vector<std::vector<float>> output(output_dims.size());
    for (size_t i = 0; i < output_dims.size(); i++)
    {
        output[i] = std::vector<float>(getSizeByDim(output_dims[i]));
        cudaMemcpy(output[i].data(), buffers[i + 1], getSizeByDim(output_dims[i]) * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << output[i][0] << std::endl;
    }
}

cv::Size Engine::get_image_size()
{
    switch (this->model_type)
    {
    case ModelType::PyTorchVisionClassification:
    case ModelType::PyTorchVisionDetection:
    case ModelType::PyTorchVisionSegmentation:
        return cv::Size(input_dims[0].d[2], input_dims[0].d[3]);
    case ModelType::TensorFlowVisionClassification:
    case ModelType::TensorFlowVisionDetection:
        return cv::Size(input_dims[0].d[1], input_dims[0].d[2]);
    default:
        return cv::Size(0, 0);
    }
}

int Engine::preprocess_image_GPU(cv::cuda::GpuMat image, std::vector<void *> buffers, int stream_id)
{
    cudaEventRecord(engine_streams[stream_id].preprocess_start, engine_streams[stream_id].m_cudaStream);

    cv::Size image_size = get_image_size();

    auto channels = 3;
    // Step 1: Resize Image
    cv::cuda::resize(image, image, image_size, 0, 0, cv::INTER_NEAREST, engine_streams[stream_id].cv_stream);

    // Step 2: Convert to RGB, OPENCV Default is BGR
    cv::cuda::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2RGB, 0, engine_streams[stream_id].cv_stream);

    // Step 3: Normalize
    image.convertTo(image, CV_32FC3, 1.f / 255.f, engine_streams[stream_id].cv_stream);
    cv::cuda::subtract(image, cv::Scalar(0.485f, 0.456f, 0.406f), image, cv::noArray(), -1, engine_streams[stream_id].cv_stream);
    cv::cuda::divide(image, cv::Scalar(0.229f, 0.224f, 0.225f), image, 1, -1, engine_streams[stream_id].cv_stream);

    // Step 4: from HWC to CHW
    std::vector<cv::cuda::GpuMat> chw;
    switch (this->model_type)
    {
    case ModelType::PyTorchVisionClassification:
    case ModelType::PyTorchVisionDetection:
    case ModelType::PyTorchVisionSegmentation:
        for (int32_t i = 0; i < channels; ++i)
        {
            chw.emplace_back(cv::cuda::GpuMat(image_size, CV_32FC1, (float *)buffers[0] + i * image_size.width * image_size.height));
        }
        cv::cuda::split(image, chw, engine_streams[stream_id].cv_stream);
        break;
    case ModelType::TensorFlowVisionClassification:
    case ModelType::TensorFlowVisionDetection:
        cudaMemcpyAsync(buffers[0], image.data, getSizeByDim(input_dims[0]) * image.elemSize(), cudaMemcpyDeviceToDevice, engine_streams[stream_id].m_cudaStream);
        // buffers[0] = image.data;
        break;
    default:
        return 1;
    }

    cudaEventRecord(engine_streams[stream_id].preprocess_stop, engine_streams[stream_id].m_cudaStream);
    return 0;
}