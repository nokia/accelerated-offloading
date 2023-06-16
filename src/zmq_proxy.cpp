/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * ZMQ Proxy:
 * Protocol Translation (ZMQ client to RDMA/GDR server)
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
#include "rdma_common.h"
#include <proxy_helpers.hpp>

int debug;

#define DEBUG_LOG \
    if (debug)    \
    printf

size_t total_output;
int total_threads = 0;
struct client_args c_args
{
};
std::vector<rdma_client *> client_list;


/// @brief Forward TCP requests via RDMA
/// @param client 
/// @param c_args 
/// @param request_size 
/// @return 
int ForwardRDMARequests(rdma_client *client, struct client_args c_args, size_t request_size)
{
    struct ibv_wc wc; // Work Completion
    struct ibv_sge sge;
    struct ibv_send_wr send_wr = {};
    struct ibv_send_wr *bad_send_wr;
    struct ibv_recv_wr recv_wr = {};
    struct ibv_recv_wr *bad_recv_wr;
    int ret;

    /* Prepost receive */
    // Prepare scatter gather pointers
    sge.addr = (uintptr_t)client->reply_buf;
    sge.length = sizeof(client->reply_size);
    sge.lkey = client->reply_mr->lkey;
    // Prepare receive work requests
    recv_wr.wr_id = 0;
    recv_wr.sg_list = &sge;
    recv_wr.num_sge = 1;

    if (ibv_post_recv(client->cm_id->qp, &recv_wr, &bad_recv_wr))
        return 1;

    sge.addr = (uintptr_t)client->request_buf;
    sge.lkey = client->request_mr->lkey;
    sge.length = request_size;
    send_wr.wr_id = 1;
    send_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    send_wr.wr.rdma.rkey = ntohl(client->pdata.buf_rkey);
    send_wr.wr.rdma.remote_addr = be64toh(client->pdata.buf_va);
    // This flag creates a CQ Event
    send_wr.send_flags = IBV_SEND_SIGNALED;
    send_wr.sg_list = &sge;
    send_wr.num_sge = 1;

    if (ibv_post_send(client->cm_id->qp, &send_wr, &bad_send_wr))
    {
        DEBUG_LOG("ibv_post_send() failed\n");
        return 1;
    }
    // Wait for send Completion
    ret = rdma_process_work_completion(client);
    if (ret)
    {
        DEBUG_LOG("rdma_process_work_completion() failed\n");
        return 1;
    }
    /* Wait for receive completion (You can asser the wc.wr_id)*/
    ret = rdma_process_work_completion(client, 2, &wc);
    if (ret)
    {
        DEBUG_LOG("rdma_process_work_completion() failed\n");
        return 1;
    }

    return 0;
}

/// @brief ZMQ Handler
/// @param context 
/// @return 
void *zmq_worker(void *context)
{
    int client_id = total_threads++;
    DEBUG_LOG("Connecting %d to RDMA server…\n", client_id);

    void *worker_socket = zmq_socket(context, ZMQ_REP);
    int ret = zmq_connect(worker_socket, "inproc://workers");
    if (ret)
    {
        fprintf(stderr, "ERRORRR");
        return NULL;
    }
    auto client = client_list[client_id];

    size_t reply_size;
    while (1)
    {
        int size = zmq_recv(worker_socket, client->request_buf, MAX_MESSAGE, 0);
        if (size == -1)
        {
            std::cerr << "zmq_recv failed " << std::endl;
            break;
        }
        DEBUG_LOG("Received message of size %d in client %d \n", size, client_id);
        RemoteTask current_task = ((remote_request *)client->request_buf)->task;
        if (current_task == RemoteTask::PING)
        {
            reply_size = sizeof(ping_reply);
        }
        else
        {
            reply_size = total_output * sizeof(float);
        }
        ret = ForwardRDMARequests(client, c_args, size);
        if (ret)
        {
            std::cerr << "Forwarding request failed " << std::endl;
            break;
        }

        size = zmq_send(worker_socket, client->reply_buf, reply_size, 0);
        if (size == -1)
        {
            std::cerr << "zmq_send failed " << std::endl;
            break;
        }
    }
    return NULL;
}

/// @brief Run ZMQ Server
/// @param local_address 
/// @param clients 
void RunServer(std::string local_address, int clients)
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

    for (int thread_nbr = 0; thread_nbr < clients; thread_nbr++)
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

    std::string server_address, local_address;
    std::string results_folder, model_name;


    int zmq_port, ret, final_result, clients;
    std::vector<std::vector<int>> output_shape;

    CLI::App app{"ZMQ Plain Proxy"};

    app.add_option("--server-address", server_address, "RDMA Server Address")->default_val("7.7.7.2");

    app.add_option("--rdma-port", c_args.port, "RDMA Server port")->default_val(20079);
    app.add_option("--zmq-port", zmq_port, "ZMQ Listen port")->default_val(5555);

    app.add_flag("-d,--debug", debug, "USE DEBUG")->default_val(0);
    app.add_option("-r,--requests", c_args.requests, "Total Requests")->default_val(1000000);
    app.add_option("-t,--task", c_args.task, "Task")->default_val(RemoteTask::MIXTURE);
    app.add_flag("-s,--save-logs", c_args.save_logs, "Save Logs")->default_val(0);
    app.add_option("-f,--folder", results_folder, "Logs Folder")->default_val("results");

    app.add_option("-c,--clients", clients, "RDMA Clients")->default_val(1);
    app.add_option("--model-name", model_name, "Model Name")->default_val("resnet50");

    CLI11_PARSE(app, argc, argv);
    if (read_outputshape_configurations(model_name, output_shape))
        return 1;

    if (clients < 1)
    {
        return 1;
    }
    for (int i = 0; i < clients; i++)
    {
        c_args.client_id = i;
        auto client = rdma_init_client(server_address, std::to_string(static_cast<int>(c_args.port)));
        if (!client)
        {
            DEBUG_LOG("rdma_init_client() %d failed \n", i);
            final_result = 1;
            goto proxy_closure;
        }
        ret = prepare_rdma_client(c_args, client);
        if (ret)
        {
            DEBUG_LOG("rdma_init_client() %d failed \n", i);
            final_result = 1;
            goto proxy_closure;
        }
        printf("RDMA Client # %d connected \n", i);
        client_list.emplace_back(client);
    }
    local_address = "tcp://*:" + std::to_string(static_cast<int>(zmq_port));

    total_output = 1;
    for (size_t i = 0; i < output_shape.size(); i++)
    {
        for (size_t j = 0; j < output_shape[i].size(); j++)
        {
            total_output *= output_shape[i][j];
        }
    }

    RunServer(local_address, clients);

proxy_closure:
    for (int i = 0; i < clients; i++)
    {
        ret = rdma_free_client(client_list[i]);
        if (ret)
        {
            fprintf(stderr, "failure in free client %d, error %d\n", i, ret);
            return 1;
        }
    }
    return final_result;
}