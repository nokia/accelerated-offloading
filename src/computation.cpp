/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Computation:
 * Implements processing workflows for different scenario.
 *
 **/

#include "computation.h"

extern int debug;

#define DEBUG_LOG \
    if (debug)    \
    printf

Engine *engine = NULL;

/// @brief Copy Data From the GPU to the Reply Buffer
/// @param reply_buf
/// @param inf_buffers
void fill_reply_buffer(void *reply_buf, std::vector<void *> inf_buffers)
{
    size_t start = 0;
    for (size_t i = 0; i < engine->output_dims.size(); i++)
    {
        auto output_size = getSizeByDim(engine->output_dims[i]) * sizeof(float);
        // 0 is input
        cudaMemcpy(reply_buf + start, inf_buffers[i + 1], output_size, cudaMemcpyDeviceToHost);
        start += output_size;
    }
}

/// @brief Map Memory address of the input and output to Inference buffers sent to the inference engine
/// @param input_buf
/// @param reply_buf
/// @return
std::vector<void *> map_reply_buffer(void *input_buf, void *reply_buf)
{
    std::vector<void *> inf_buffers(engine->get_buffers_size());
    size_t start = 0;
    inf_buffers[0] = input_buf;
    for (size_t i = 0; i < engine->output_dims.size(); i++)
    {
        inf_buffers[i + 1] = reply_buf + start;
        start += getSizeByDim(engine->output_dims[i]) * sizeof(float);
    }
    return inf_buffers;
}

/// @brief Process Raw Images
/// @param stream_id
/// @param addressing
/// @param request_buf
/// @param reply_buf
/// @param t_details
/// @return
int process_raw_data(int stream_id, MemoryAddressing addressing, void *request_buf, void *reply_buf, time_details *t_details)
{
    int result;
    float milliseconds = 0;
    std::chrono::_V2::system_clock::time_point post_copy_start, post_copy_stop, cpu_preprocess_start, cpu_preprocess_stop;

    std::vector<void *> inf_buffers;
    struct raw_request *c_request = (raw_request *)request_buf;
    auto start = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat frame;
    if (addressing == MemoryAddressing::CUDA_ADDRESSING)
    {
        size_t size[4];
        cudaMemcpy((void *)size, request_buf, sizeof(size_t) * 4, cudaMemcpyDeviceToHost);
        DEBUG_LOG("Classification Request (%lu) Size and image size (%lu,%lu) \n", size[1], size[2], size[3]);
        frame = cv::cuda::GpuMat(cv::Size(size[2], size[3]), CV_8UC3, &c_request->image_data);
        inf_buffers = map_reply_buffer(&c_request->image_data, reply_buf);
    }
    else
    {
        cudaEventRecord(engine->engine_streams[stream_id].pre_copy_start, engine->engine_streams[stream_id].m_cudaStream);
        cv::Mat temp(cv::Size(c_request->width, c_request->height), CV_8UC3, &c_request->image_data);
        frame.upload(temp, engine->engine_streams[stream_id].cv_stream);
        inf_buffers = engine->get_stream_buffers(stream_id);
        cudaEventRecord(engine->engine_streams[stream_id].pre_copy_start, engine->engine_streams[stream_id].m_cudaStream);
    }
    result = engine->preprocess_image_GPU(frame, inf_buffers, stream_id);
    if (result)
    {
        std::cout << "PreProcesing Failed \n";
        return result;
    }
    if (debug)
    {
        cv::Mat out;
        frame.download(out);
        imwrite("Case2.jpg", out);
    }
    result = engine->runInferenceOnly(inf_buffers, stream_id, debug);
    if (result)
    {
        std::cout << "Inference Failed \n";
        return result;
    }
    post_copy_start = std::chrono::high_resolution_clock::now();
    if (addressing == MemoryAddressing::HOST_ADDRESSING)
    {
        fill_reply_buffer(reply_buf, inf_buffers);
    }
    post_copy_stop = std::chrono::high_resolution_clock::now();

    auto end = std::chrono::high_resolution_clock::now();
    t_details->total_compute = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;

    if (addressing == MemoryAddressing::HOST_ADDRESSING)
    {
        cudaEventElapsedTime(&milliseconds, engine->engine_streams[stream_id].pre_copy_start, engine->engine_streams[stream_id].pre_copy_stop);
        t_details->pre_copy_time = milliseconds;
        t_details->post_copy_time = std::chrono::duration_cast<std::chrono::nanoseconds>(post_copy_stop - post_copy_start).count() * 1e-6;
    }
    cudaEventElapsedTime(&milliseconds, engine->engine_streams[stream_id].preprocess_start, engine->engine_streams[stream_id].preprocess_stop);
    t_details->preprocessing = milliseconds;

    cudaEventElapsedTime(&milliseconds, engine->engine_streams[stream_id].inference_start, engine->engine_streams[stream_id].inference_stop);
    t_details->inference_time = milliseconds;
    return 0;
}

/// @brief Process
/// @param stream_id
/// @param addressing
/// @param request_buf
/// @param reply_buf
/// @param t_details
/// @return
int process_processed_data(int stream_id, MemoryAddressing addressing, void *request_buf, void *reply_buf, time_details *t_details)
{
    int result;
    float milliseconds = 0;
    std::chrono::_V2::system_clock::time_point post_copy_start, post_copy_stop;

    struct processed_request *c_request = (processed_request *)request_buf;
    std::vector<void *> inf_buffers;

    auto start = std::chrono::high_resolution_clock::now();

    if (addressing == MemoryAddressing::HOST_ADDRESSING)
    {
        inf_buffers = engine->get_stream_buffers(stream_id);

        DEBUG_LOG("Image total size %lu \n", getSizeByDim(engine->input_dims[0]) * sizeof(float));

        cudaEventRecord(engine->engine_streams[stream_id].pre_copy_start, engine->engine_streams[stream_id].m_cudaStream);
        cudaMemcpyAsync(inf_buffers[0], &c_request->image_data, getSizeByDim(engine->input_dims[0]) * sizeof(float), cudaMemcpyHostToDevice, engine->engine_streams[stream_id].m_cudaStream);
        cudaEventRecord(engine->engine_streams[stream_id].pre_copy_stop, engine->engine_streams[stream_id].m_cudaStream);
    }
    else
    {
        inf_buffers = map_reply_buffer(&c_request->image_data, reply_buf);
    }

    result = engine->runInferenceOnly(inf_buffers, stream_id, debug);
    if (result)
    {
        std::cout << "Inference Failed \n";
        return result;
    }
    post_copy_start = std::chrono::high_resolution_clock::now();

    if (addressing == MemoryAddressing::HOST_ADDRESSING)
    {
        fill_reply_buffer(reply_buf, inf_buffers);
    }
    post_copy_stop = std::chrono::high_resolution_clock::now();

    auto end = std::chrono::high_resolution_clock::now();
    t_details->total_compute = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;
    if (addressing == MemoryAddressing::HOST_ADDRESSING)
    {
        cudaEventElapsedTime(&milliseconds, engine->engine_streams[stream_id].pre_copy_start, engine->engine_streams[stream_id].pre_copy_stop);
        t_details->pre_copy_time = milliseconds;
    }
    cudaEventElapsedTime(&milliseconds, engine->engine_streams[stream_id].inference_start, engine->engine_streams[stream_id].inference_stop);
    t_details->inference_time = milliseconds;
    return 0;
}

/// @brief Process Request based on Request Type
/// @param client_id
/// @param task
/// @param addressing
/// @param request_buf
/// @param reply_buf
/// @param t_details
/// @return
int process_request(int client_id, RemoteTask task, MemoryAddressing addressing, void *request_buf, void *reply_buf, time_details *t_details)
{
    int result;
    int stream_id;
    if (engine->high_priority_clients == 0)
    {
        DEBUG_LOG("No High Priority \n");
        stream_id = engine->get_stream_id();
    }
    else
    {
        stream_id = client_id;
    }

    DEBUG_LOG("Incoming Request from client #%d served from stream #%d\n", client_id, stream_id);

    switch (task)
    {
    case RemoteTask::PING:
        result = 0;
        break;
    case RemoteTask::CLASSIFICATION_RAW:
        result = process_raw_data(stream_id, addressing, request_buf, reply_buf, t_details);
        break;
    case RemoteTask::CLASSIFICATION_PROCESSED:
        result = process_processed_data(stream_id, addressing, request_buf, reply_buf, t_details);
        break;
    default:
        fprintf(stderr, "process_request failed not supported operation %d\n", static_cast<int>(task));
        result = 1;
    }
    engine->release_stream(stream_id);
    return result;
}

/// @brief Save Server side logs
/// @param execution_time
/// @param results_folder
/// @param file_name
void save_details(std::vector<time_details> execution_time, std::string results_folder, std::string file_name)
{
    results_folder = results_folder + "/";
    mkdir(results_folder.c_str(), 0700);
    std::ofstream myFile(results_folder + file_name + ".csv");
    // Send the column name to the stream
    myFile << "server_time"
           << ","
           << "decode_time"
           << ","
           << "pre_copy_time"
           << ","
           << "preprocessing"
           << ","
           << "mid_copy_time"
           << ","
           << "inference_time"
           << ","
           << "post_copy_time"
           << ","
           << "total_compute"
           << ","
           << "send_time"
           << "\n";

    // Send data to the stream
    for (size_t i = 0; i < execution_time.size(); ++i)
    {
        myFile << execution_time[i].server_time << ",";
        myFile << execution_time[i].decode_time << ",";
        myFile << execution_time[i].pre_copy_time << ",";
        myFile << execution_time[i].preprocessing << ",";
        myFile << execution_time[i].mid_copy_time << ",";
        myFile << execution_time[i].inference_time << ",";
        myFile << execution_time[i].post_copy_time << ",";
        myFile << execution_time[i].total_compute << ",";
        myFile << execution_time[i].send_time << "\n";
    }

    // Close the file
    myFile.close();
}