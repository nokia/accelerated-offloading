/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * ZMQ Server:
 * ZMQ Server Implementation
 **/

#include <zmq.h>
#include <stdio.h>
#include "utils.h"
#include "CLI11.hpp"
#include <string.h>
#include <iostream>
#include "computation.h"
#include <mutex>
#include <condition_variable>
#include "monitor.hpp"

int debug;

#define DEBUG_LOG \
    if (debug)    \
    printf

extern Engine *engine;
std::vector<void *> request_buffer;
std::vector<void *> reply_buffer;
int streams, workers;
std::string results_folder;
int monitor_interval;
bool save_logs;
int total_requests, total_threads = 0;
bool high_priority_clients;

/// @brief ZMQ worker
/// @param context 
/// @return 
void *zmq_worker(void *context)
{
    int worker_id = total_threads++;
    void *worker_socket = zmq_socket(context, ZMQ_REP);
    int ret = zmq_connect(worker_socket, "inproc://workers");
    if (ret)
    {
        fprintf(stderr, "ERRORRR");
        return NULL;
    }
    size_t reply_size;
    std::vector<time_details> execution_time;
    Monitor m(monitor_interval);
    int requests = 0;
    int task;
    while (true)
    {
        //auto stream_id = get_stream_id();
        int size = zmq_recv(worker_socket, request_buffer[worker_id], MAX_MESSAGE, 0);
        if (size == -1)
        {
            std::cerr << "zmq_recv failed " << std::endl;
            break;
        }
        struct time_details t_details;
        auto start = std::chrono::high_resolution_clock::now();
        DEBUG_LOG("Received message of size %d \n", size);
        RemoteTask current_task = ((remote_request *)request_buffer[worker_id])->task;
        task = (int)current_task;
        if (current_task == RemoteTask::PING)
        {
            reply_size = sizeof(ping_reply);
        }
        else
        {
            reply_size = engine->total_output_size * sizeof(float);
        }
        int ret = process_request(worker_id, current_task, MemoryAddressing::HOST_ADDRESSING, request_buffer[worker_id], reply_buffer[worker_id], &t_details);
        if (ret)
        {
            std::cerr << "process_request failed " << std::endl;
            break;
        }
        auto send_start = std::chrono::high_resolution_clock::now();
        size = zmq_send(worker_socket, reply_buffer[worker_id], reply_size, 0);
        auto end = std::chrono::high_resolution_clock::now();
        t_details.send_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - send_start).count() * 1e-6;
        t_details.server_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;
        execution_time.push_back(t_details);
        if (size == -1)
        {
            std::cerr << "zmq_send failed " << std::endl;
            break;
        }
        //release_stream(stream_id);
        requests++;
        if (requests == total_requests && save_logs)
        {
            requests = 0;
            std::string experiment_name = "zserver_" + std::to_string(worker_id) + "_";
            experiment_name = experiment_name + std::to_string(task) + "_";
            experiment_name = experiment_name + std::to_string(total_requests);
            save_details(execution_time, results_folder, experiment_name);
            execution_time.clear();
            m.save_monitoring_state(results_folder, experiment_name);
            Monitor m(monitor_interval);
        }
    }
    return NULL;
}

/// @brief RUN ZMQ Server
/// @param server_address 
/// @param engine_file 
/// @param classes_file 
void RunServer(std::string server_address, std::string engine_file, std::string classes_file)
{

    engine = new Engine(engine_file, classes_file, high_priority_clients, streams);
    printf("Model Ready \n");
    //busy = std::vector<bool>(streams);
    //  Socket to talk to clients
    void *context = zmq_ctx_new();
    void *router = zmq_socket(context, ZMQ_ROUTER);
    int ret = zmq_bind(router, server_address.c_str());
    if (ret)
    {
        std::cerr << "Listening failed " << std::endl;
        return;
    }
    std::cout << "Server listening on " << server_address << "\n";
    request_buffer = std::vector<void *>(workers);
    reply_buffer = std::vector<void *>(workers);

    for (int i = 0; i < workers; i++)
    {
        request_buffer[i] = malloc(MAX_MESSAGE);
        reply_buffer[i] = malloc(MAX_MESSAGE);
    }
    void *dealer = zmq_socket(context, ZMQ_DEALER);

    ret = zmq_bind(dealer, "inproc://workers");
    if (ret)
    {
        std::cerr << "Dealer Failed " << std::endl;
        return;
    }

    for (int thread_nbr = 0; thread_nbr < workers; thread_nbr++)
    {
        pthread_t worker;
        pthread_create(&worker, NULL, zmq_worker, context);
    }
    //  Connect work threads to client threads via a queue
    ret = zmq_proxy(router, dealer, NULL);
    if (ret)
    {
        std::cerr << "ZMQ Device Failed" << std::endl;
        return;
    }
}
int main(int argc, char **argv)
{

    std::string engine_file, classes_file;
    int zmq_port;

    CLI::App app{"ZMQ Server"};

    app.add_option("--port", zmq_port, "ZMQ Listen port")->default_val(5555);
    app.add_flag("-d,--debug", debug, "USE DEBUG")->default_val(0);

    app.add_option("--classes-file", classes_file, "Classes File")->default_val("models/imagenet_classes.txt");
    app.add_option("--engine-file", engine_file, "Tensor RT Engine File")->default_val("models/resnet50.trt");
    app.add_option("--streams", streams, "Number of Cuda Streams and TRTContexts")->default_val(0);
    app.add_option("--workers", workers, "Number of ZMQ Workers")->default_val(1);
    app.add_option("-f,--folder", results_folder, "Logs Folder")->default_val("results");
    app.add_option("-i,--monitor-interval", monitor_interval, "Monitor Interval")->default_val(20);
    app.add_flag("-s,--save-logs", save_logs, "Save Logs")->default_val(false);
    app.add_option("--priority-clients", high_priority_clients, "High Priority Client")->default_val(0);
    app.add_option("--total-requests", total_requests, "Total requests")->default_val(1000);

    CLI11_PARSE(app, argc, argv);
    if (workers < 1)
    {
        return 1;
    }
    if (streams == 0)
    {
        streams = workers;
    }

    std::string server_address = "tcp://*:" + std::to_string(static_cast<int>(zmq_port));
    RunServer(server_address, engine_file, classes_file);

    return 0;
}