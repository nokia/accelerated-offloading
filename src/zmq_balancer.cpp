/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * ZMQ Balancer:
 * Load Balancer (ZMQ client to ZMQ server)
 **/

#include <zmq.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include "utils.h"
#include "CLI11.hpp"
#include <chrono>
#include <memory>
#include <iostream>
#include <proxy_helpers.hpp>

int debug;

#define DEBUG_LOG \
    if (debug)    \
    printf

std::string server_address;
int server_port, connections, total_threads;
std::vector<std::vector<int>> output_shape;

/// @brief Inference Proxy Client
class InferenceProxyClient
{
private:
    void *context = NULL;
    void *requester = NULL;
    std::vector<std::vector<int>> output_shape;
    size_t total_output;
    void *request_buffer = NULL;
    void *reply_buffer = NULL;
    int client_id;


public:
    InferenceProxyClient(std::string server_address, std::vector<std::vector<int>> _output_shape)
    {
        client_id = total_threads++;
        DEBUG_LOG("Connecting %d to ZMQ server…\n", client_id);
        context = zmq_ctx_new();
        requester = zmq_socket(context, ZMQ_REQ);
        zmq_connect(requester, server_address.c_str());
        request_buffer = malloc(MAX_MESSAGE);
        reply_buffer = malloc(MAX_MESSAGE);
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

    ~InferenceProxyClient()
    {
        if (requester)
            zmq_close(requester);
        if (context)
            zmq_ctx_destroy(context);
        if (request_buffer)
            free(request_buffer);
        if (reply_buffer)
            free(reply_buffer);
    }

    /// @brief Forward Requests
    /// @param responder 
    /// @return 
    int Forward(void* responder)
    {
        int size = zmq_recv(responder, request_buffer, MAX_MESSAGE, 0);
        if (size == -1)
        {
            std::cerr << "zmq_recv failed " << std::endl;
            return 1;
        }
        DEBUG_LOG("Received message of size %d to client %d.\n", size, client_id);
        int ret = zmq_send(requester, request_buffer, size, 0);
        if (ret ==-1)
        {
            std::cerr << "zmq_send failed " << std::endl;
            return 1;
        }

        int reply_size = zmq_recv(requester, reply_buffer, MAX_MESSAGE, 0);
        if (size == -1)
        {
            std::cerr << "zmq_recv failed " << std::endl;
            return 1;
        }

        size = zmq_send(responder, reply_buffer, reply_size, 0);
        if (size == -1)
        {
            std::cerr << "zmq_send failed " << std::endl;
            return 1;
        }
        return 0;
    }
};


/// @brief ZMQ Worker 
/// @param context 
/// @return 
void *zmq_worker(void *context)
{
    void *worker_socket = zmq_socket(context, ZMQ_REP);
    int ret = zmq_connect(worker_socket, "inproc://workers");
    if (ret)
    {
        fprintf(stderr, "ERRORRR");
        return NULL;
    }
    InferenceProxyClient client(server_address, output_shape);
    while (1)
    {
        int ret = client.Forward(worker_socket);
        if(ret){
            printf("??????????");
            break;
        }
    }
    return NULL;
}

/// @brief Run ZMQ Server
/// @param local_address 
void RunServer(std::string local_address)
{
    void *context = zmq_ctx_new();
    void *router = zmq_socket(context, ZMQ_ROUTER);
    int ret = zmq_bind(router, local_address.c_str());
    if (ret)
    {
        std::cerr << "Listening failed " << std::endl;
        return;
    }
    std::cout << "Server listening on " << local_address << "\n";
    void *dealer = zmq_socket(context, ZMQ_DEALER);

    ret = zmq_bind(dealer, "inproc://workers");
    if (ret)
    {
        std::cerr << "Dealer Failed " << std::endl;
        return;
    }

    for (int thread_nbr = 0; thread_nbr < connections; thread_nbr++)
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
    std::string local_address;
    std::string model_name;
    int local_port;

    CLI::App app{"ZMQ Plain Proxy"};

    app.add_option("--server-address", server_address, "Target Server Address")->default_val("7.7.7.2");
    app.add_option("--server-port",server_port, " Target Server port")->default_val(5555);
    app.add_option("--local-port", local_port, "ZMQ Listen port")->default_val(5555);
    app.add_flag("-d,--debug", debug, "USE DEBUG")->default_val(0);
    app.add_option("-c,--connections", connections, "Number of Cuda Streams and TRTContexts")->default_val(1);
    app.add_option("--model-name", model_name, "Model Name")->default_val("resnet50");

    CLI11_PARSE(app, argc, argv);
    if (read_outputshape_configurations(model_name, output_shape))
        return 1;
    
    server_address = "tcp://" + server_address + ":" + std::to_string(static_cast<int>(server_port));
    local_address = "tcp://*:" + std::to_string(static_cast<int>(local_port));

    RunServer(local_address);
    return 0;
}