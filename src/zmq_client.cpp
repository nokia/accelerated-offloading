/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * ZMQ Client:
 * ZMQ Client Implementation
 **/

#include <zmq.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include "CLI11.hpp"
#include <chrono>
#include <memory>
#include <iostream>
#include "utils.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <client_helpers.hpp>

int debug;

#define DEBUG_LOG \
    if (debug)    \
    printf

class InferenceClient
{
private:
    FILE *file = NULL;
    void *image_data = NULL;
    size_t imgSize;
    cv::Mat frame;
    void *context = NULL;
    void *requester = NULL;
    std::string classes_file;
    client_args c_args;
    void *request_buffer = NULL;
    void *reply_buffer = NULL;
    std::vector<std::vector<int>> output_shape;
    size_t total_output;

public:
    InferenceClient(std::string server_address, client_args c_args, std::string classes_file, std::vector<std::vector<int>> _output_shape)
    {
        DEBUG_LOG("Connecting to ZMQ server…\n");
        context = zmq_ctx_new();
        requester = zmq_socket(context, ZMQ_REQ);
        zmq_connect(requester, server_address.c_str());
        this->classes_file = classes_file;
        this->c_args = c_args;
        request_buffer = malloc(MAX_MESSAGE);
        reply_buffer = malloc(MAX_MESSAGE * sizeof(float));
        output_shape = _output_shape;
        total_output = 1;
        for (size_t i = 0; i < output_shape.size(); i++)
        {
            for (size_t j = 0; j < output_shape[i].size(); j++)
            {
                total_output *= output_shape[i][j];
            }
        }
    }
    ~InferenceClient()
    {
        if (requester)
            zmq_close(requester);
        if (context)
            zmq_ctx_destroy(context);
        if (request_buffer)
            free(request_buffer);
        if (reply_buffer)
            free(reply_buffer);
        if (image_data)
            free(image_data);
    }
    /// @brief Send Ping Request
    /// @return 
    int Ping()
    {
        int result;
        ping_request request;
        request.task = RemoteTask::PING;

        DEBUG_LOG("Sending ping.\n");
        zmq_send(requester, &request, sizeof(ping_request), 0);
        ping_reply reply;

        result = zmq_recv(requester, &reply, sizeof(ping_reply), 0);
        if (result == -1)
        {
            return 1;
        }
        DEBUG_LOG("Ping worked !\n");
        DEBUG_LOG("Message sizes are %d and %d. \n", sizeof(ping_request), sizeof(ping_reply));
        return 0;
    }

    /// @brief Send Inference Raw Request
    /// @param image 
    /// @return 
    int InferenceRaw(std::string image)
    {
        if (frame.empty())
        {
            frame = cv::imread(image.c_str());
            if (frame.empty())
            {
                std::cerr << "Input image load failed\n";
                return 1;
            }
            imgSize = frame.total() * frame.elemSize();
            std::cout << imgSize << "," << frame.elemSize() << "," << frame.size().width << "," << frame.size().height << std::endl;
        }

        int result;
        auto zmq_request = (raw_request *)request_buffer;
        zmq_request->task = RemoteTask::CLASSIFICATION_RAW;
        zmq_request->request_size = 0;
        zmq_request->width = frame.size().width;
        zmq_request->height = frame.size().height;

        memcpy(&zmq_request->image_data, frame.data, imgSize);

        size_t total_size = imgSize + 2 * sizeof(size_t) + sizeof(remote_request);
        DEBUG_LOG("Total Size = %lu \n", total_size);
        zmq_send(requester, zmq_request, total_size, 0);

        result = zmq_recv(requester, reply_buffer, total_output * sizeof(float), 0);
        if (result == -1)
        {
            return 1;
        }
        if (debug)
            print_results(c_args, reply_buffer, classes_file, output_shape);

        return 0;
    }

    /// @brief Send Inference with preprocessed request
    /// @param image 
    /// @param input_shape 
    /// @return 
    int InferenceProcessed(std::string image, std::vector<int> input_shape)
    {
        if (frame.empty())
        {
            frame = preprocess_image(image, input_shape);
            imgSize = frame.total() * frame.elemSize();
        }

        int result;
        auto zmq_request = (processed_request *)request_buffer;
        zmq_request->task = RemoteTask::CLASSIFICATION_PROCESSED;
        zmq_request->request_size = 0;

        memcpy(&zmq_request->image_data, frame.data, imgSize);

        size_t total_size = imgSize + sizeof(remote_request);
        DEBUG_LOG("Total Size = %lu \n", total_size);
        zmq_send(requester, zmq_request, total_size, 0);

        result = zmq_recv(requester, reply_buffer, total_output * sizeof(float), 0);
        if (result == -1)
        {
            return 1;
        }
        if (debug)
            print_results(c_args, reply_buffer, classes_file, output_shape);
        return 0;
    }
};

int main(int argc, char **argv)
{

    std::string server_address;
    std::string image, results_folder, classes_file, model_name;
    int result;
    struct client_args c_args
    {
    };

    std::vector<int> input_shape;
    std::vector<std::vector<int>> output_shape;

    CLI::App app{"ZMQ Client"};

    app.add_option("--host", server_address, "RDMA Host")->default_val("7.7.7.2");
    app.add_option("--port", c_args.port, "ZMQ Server port")->default_val(5555);

    app.add_option("-i,--image", image, "Image")->default_val("dog.jpg");

    app.add_flag("-d,--debug", debug, "USE DEBUG")->default_val(0);
    app.add_option("-r,--requests", c_args.requests, "Total Requests")->default_val(1);

    app.add_option("--client-id", c_args.client_id, "Client ID")->default_val(0);

    app.add_option("-t,--task", c_args.task, "Task")->default_val(RemoteTask::PING);
    app.add_flag("-s,--save-logs", c_args.save_logs, "Save Logs")->default_val(false);
    app.add_option("-f,--folder", results_folder, "Logs Folder")->default_val("results");
    app.add_option("--model-name", model_name, "Model Name")->default_val("resnet50");

    CLI11_PARSE(app, argc, argv);

    if (read_model_configurations(&c_args, model_name, classes_file, input_shape, output_shape))
        return 1;

    server_address = "tcp://" + server_address + ":" + std::to_string(static_cast<int>(c_args.port));

    InferenceClient client(server_address, c_args, classes_file, output_shape);

    std::vector<double> execution_time(c_args.requests);
    for (int i = 0; i < c_args.requests; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        switch (c_args.task)
        {
        case RemoteTask::PING:
            result = client.Ping();
            break;
        case RemoteTask::CLASSIFICATION_RAW:
            result = client.InferenceRaw(image);
            break;
        case RemoteTask::CLASSIFICATION_PROCESSED:
            result = client.InferenceProcessed(image, input_shape);
            break;
        default:
            std::cerr << "Wrong Task \n";
            return 1;
        }
        if (result)
        {
            std::cerr << "Request " << i << " Failed \n";
            return result;
        }
        auto end = std::chrono::high_resolution_clock::now();
        execution_time[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;
    }

    double sum = 0;
    for (int i = 0; i < c_args.requests; i++)
        sum += execution_time[i];

    printf("Request took %f ms per request \n", sum / c_args.requests);
    if (c_args.save_logs)
    {
        std::string experiment_name = "zclient_";
        experiment_name = experiment_name + std::to_string(static_cast<int>(c_args.client_id)) + "_";
        experiment_name = experiment_name + std::to_string(static_cast<int>(c_args.task)) + "_";
        experiment_name = experiment_name + std::to_string(c_args.requests);
        save_details(execution_time, results_folder, experiment_name);
    }

    return 0;
}