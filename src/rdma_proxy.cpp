/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * RDMA Proxy:
 * Implements RDMA Proxy that maps RDMA Connections between client and server. 
 * The current version maps to one server but it can be a load balancer
 *
 **/

#include "rdma_common.h"
#include "CLI11.hpp"
#include <thread>
#include <proxy_helpers.hpp>
#include <mutex>
#include <condition_variable>

int debug;

#define DEBUG_LOG \
    if (debug)    \
    printf

std::mutex m;
std::vector<bool> busy;
std::condition_variable _cv;

/// @brief Map Client to Server Connection
/// @param connections 
/// @return 
int get_connection_id(int connections)
{
    while (true)
    {
        {
            // Make the lock in its own scope to release when exit
            std::lock_guard<std::mutex> lg{m};
            for (int i = 0; i < connections; i++)
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

/// @brief Release Server Connection
/// @param connection_id 
void release_client_proxy(int connection_id)
{
    std::unique_lock<std::mutex> ul{m};
    busy[connection_id] = false;
    ul.unlock();
    _cv.notify_one();
}

/// @brief Handle Client Conenction and allocate needed resources.
/// @param s_args 
/// @param client_session 
/// @param proxy_client 
/// @return 
int handle_client_connection(server_args s_args, struct rdma_client *client_session, rdma_client *proxy_client)
{
    struct ibv_qp_init_attr qp_attr = {};
    struct privatedata rep_pdata;
    struct rdma_conn_param conn_param = {};
    int err;

    struct ibv_sge sge; // Scatter Gather Items
    struct ibv_recv_wr recv_wr = {};
    struct ibv_recv_wr *bad_recv_wr;

    /* Create verbs objects now that we know which device to use */
    // 1. Create Protection Domain
    client_session->pd = ibv_alloc_pd(client_session->cm_id->verbs);
    if (!client_session->pd)
    {
        DEBUG_LOG("ibv_alloc_pd() failure \n");
        return 1;
    }
    // 2. Create Completion Channel
    client_session->comp_chan = ibv_create_comp_channel(client_session->cm_id->verbs);
    if (!client_session->comp_chan)
    {
        DEBUG_LOG("ibv_create_comp_channel() failure \n");
        return 1;
    }
    // 3. Create Completetion Queue This queue is with completetion channel to listen to events.
    client_session->cq = ibv_create_cq(client_session->cm_id->verbs, RDMA_CQ_DEPTH, NULL, client_session->comp_chan, 0);
    if (!client_session->cq)
    {
        DEBUG_LOG("ibv_create_cq() failure \n");
        return 1;
    }
    // 4. Ask the cq to listen to notifications (this insures that next event is monitored)
    // This is for the recieve
    if (ibv_req_notify_cq(client_session->cq, 0))
    {
        DEBUG_LOG("ibv_req_notify_cq() failure \n");
        return 1;
    }
    // 5. Allocate Memory Buffers
    // 5.1 Fix Buffers Sizes for all operations
    client_session->request_size = proxy_client->request_size;
    client_session->reply_size = proxy_client->reply_size;

    // 6 No need to allocate buffers, use the same as proxy
    client_session->request_buf = proxy_client->request_buf;
    client_session->reply_buf = proxy_client->reply_buf;
    // 6.1 Register Request Buffer
    client_session->request_mr = ibv_reg_mr(client_session->pd, client_session->request_buf, client_session->request_size,
                                            (
                                                IBV_ACCESS_LOCAL_WRITE |
                                                IBV_ACCESS_REMOTE_READ |
                                                IBV_ACCESS_REMOTE_WRITE));
    if (!client_session->request_mr)
    {
        DEBUG_LOG("request buffer ibv_reg_mr() failure \n");
        return 1;
    }
    // 6.1 Register Reply Buffer
    client_session->reply_mr = ibv_reg_mr(client_session->pd, client_session->reply_buf, client_session->reply_size, IBV_ACCESS_LOCAL_WRITE);
    if (!client_session->reply_mr)
    {
        DEBUG_LOG("reply buffer ibv_reg_mr() failure \n");
        return 1;
    }

    // 7. Set QP Attributes
    qp_attr.cap.max_send_wr = MAX_WR;
    qp_attr.cap.max_send_sge = MAX_SGE;
    qp_attr.cap.max_recv_wr = MAX_WR;
    qp_attr.cap.max_recv_sge = MAX_SGE;
    qp_attr.send_cq = client_session->cq;
    qp_attr.recv_cq = client_session->cq;
    qp_attr.qp_type = IBV_QPT_RC;
    // qp_attr.sq_sig_all = 1;

    // 8. Create QP
    err = rdma_create_qp(client_session->cm_id, client_session->pd, &qp_attr);
    if (err)
        return err;

    // 9. Post receive before accepting connection because we are using send
    sge.addr = (uintptr_t)client_session->request_buf; // + sizeof(uint32_t);
    sge.length = client_session->request_size;
    sge.lkey = client_session->request_mr->lkey;

    recv_wr.sg_list = &sge;
    recv_wr.num_sge = 1;

    // 10. Post Receive (because we are using send)
    if (ibv_post_recv(client_session->cm_id->qp, &recv_wr, &bad_recv_wr))
        return 1;

    // 11. Allocate Buffer and key
    rep_pdata.buf_va = be64toh((uintptr_t)client_session->request_buf);
    rep_pdata.buf_rkey = htonl(client_session->request_mr->rkey);
    rep_pdata.addressing = proxy_client->pdata.addressing;

    // 12. Accept Connection and wait for ack
    conn_param.responder_resources = 1;
    conn_param.private_data = &rep_pdata;
    conn_param.private_data_len = sizeof rep_pdata;
    conn_param.retry_count = 7;
    /* Accept connection */

    err = rdma_accept(client_session->cm_id, &conn_param);
    if (err)
    {
        DEBUG_LOG("rdma_accept() failure\n");
        return 1;
    }

    // err = rdma_block_for_cm_event(client_session->cm_channel, RDMA_CM_EVENT_ESTABLISHED, RDMA_CM_EVENT_REJECTED);
    return err;
}

/// @brief Forward Requests to the Server
/// @param proxy 
/// @param proxy_args 
/// @param request_size 
/// @return 
int ForwardRDMARequests(rdma_client *proxy, struct client_args proxy_args, size_t request_size)
{
    struct ibv_wc wc; // Work Completion
    struct ibv_sge sge;
    struct ibv_send_wr send_wr = {};
    struct ibv_send_wr *bad_send_wr;
    struct ibv_recv_wr recv_wr = {};
    struct ibv_recv_wr *bad_recv_wr;
    int ret;

    /* PrePost receive */
    // Prepare scatter gather pointers
    sge.addr = (uintptr_t)proxy->reply_buf;
    sge.length = sizeof(proxy->reply_size);
    sge.lkey = proxy->reply_mr->lkey;
    // Prepare receive work requests
    recv_wr.wr_id = 0;
    recv_wr.sg_list = &sge;
    recv_wr.num_sge = 1;

    if (ibv_post_recv(proxy->cm_id->qp, &recv_wr, &bad_recv_wr))
        return 1;

    sge.addr = (uintptr_t)proxy->request_buf;
    sge.lkey = proxy->request_mr->lkey;
    sge.length = request_size;
    send_wr.wr_id = 1;
    send_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    send_wr.wr.rdma.rkey = ntohl(proxy->pdata.buf_rkey);
    send_wr.wr.rdma.remote_addr = be64toh(proxy->pdata.buf_va);
    // This flag creates a CQ Event
    send_wr.send_flags = IBV_SEND_SIGNALED;
    send_wr.sg_list = &sge;
    send_wr.num_sge = 1;
    auto start = std::chrono::high_resolution_clock::now();
    if (ibv_post_send(proxy->cm_id->qp, &send_wr, &bad_send_wr))
    {
        DEBUG_LOG("ibv_post_send() failed\n");
        return 1;
    }
    // Wait for send Completion
    ret = rdma_process_work_completion(proxy);
    if (ret)
    {
        DEBUG_LOG("rdma_process_work_completion() failed\n");
        return 1;
    }
    auto send_done = std::chrono::high_resolution_clock::now();

    /* Wait for receive completion (You can asser the wc.wr_id)*/
    ret = rdma_process_work_completion(proxy, 2, &wc);
    if (ret)
    {
        DEBUG_LOG("rdma_process_work_completion() failed\n");
        return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto send_time = std::chrono::duration_cast<std::chrono::nanoseconds>(send_done - start).count() * 1e-6;
    auto wait_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - send_done).count() * 1e-6;

    DEBUG_LOG("Sending to Target took %f ms, and waiting took %f ms\n", send_time, wait_time);
    return 0;
}

/// @brief Handle Clients requests
/// @param s_args 
/// @param proxy_args 
/// @param client_session 
/// @param proxy_client 
/// @param total_requests 
/// @param total_output 
/// @return 
int handle_requests(struct server_args s_args, client_args proxy_args, struct rdma_client *client_session, rdma_client *proxy_client, int total_requests, size_t total_output)
{
    int ret, reply_size;
    struct ibv_sge sge; // Scatter Gather Items
    struct ibv_send_wr send_wr = {};
    struct ibv_send_wr *bad_send_wr;
    struct ibv_recv_wr recv_wr = {};
    struct ibv_recv_wr *bad_recv_wr;
    for (int i = 0; i < total_requests; i++)
    {
        /* Wait for receive completion */
        ret = rdma_process_work_completion(client_session);
        if (ret)
        {
            DEBUG_LOG("rdma_process_work_completion() failed \n");
            return ret;
        }
        remote_request *request = (remote_request *)client_session->request_buf;
        switch (request->task)
        {
        case RemoteTask::PING:
            reply_size = sizeof(ping_reply);
            break;

        case RemoteTask::CLASSIFICATION_RAW:
        case RemoteTask::CLASSIFICATION_PROCESSED:
            reply_size = total_output * sizeof(float);
            break;
        default:
            reply_size = 0; // total_output * sizeof(float);
            printf("Error, option is wrong \n");
            return 1;
        }
        DEBUG_LOG("Received message of size %d \n", request->request_size);

        /* Forward Request */
        ForwardRDMARequests(proxy_client, proxy_args, request->request_size);
        // Before reply (Post recv request)
        if (i < total_requests - 1)
        {
            sge.addr = (uintptr_t)client_session->request_buf; // + sizeof(uint32_t);
            sge.length = client_session->request_size;
            sge.lkey = client_session->request_mr->lkey;

            recv_wr.sg_list = &sge;
            recv_wr.num_sge = 1;

            // 10. Post Receive (because we are using send)
            if (ibv_post_recv(client_session->cm_id->qp, &recv_wr, &bad_recv_wr))
                return 1;
        }

        sge.addr = (uintptr_t)client_session->reply_buf;
        sge.length = reply_size;
        sge.lkey = client_session->reply_mr->lkey;
        send_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        send_wr.wr.rdma.rkey = ntohl(client_session->pdata.buf_rkey);
        send_wr.wr.rdma.remote_addr = be64toh(client_session->pdata.buf_va);
        // This flag creates a CQ Event
        send_wr.send_flags = IBV_SEND_SIGNALED;
        send_wr.sg_list = &sge;
        send_wr.num_sge = 1;
        if (ibv_post_send(client_session->cm_id->qp, &send_wr, &bad_send_wr))
        {
            DEBUG_LOG("ibv_post_send() failure");
            return 1;
        }

        /* Wait for receive completion */
        ret = rdma_process_work_completion(client_session, 2);
        if (ret)
        {
            DEBUG_LOG("rdma_process_work_completion() failed \n");
            return ret;
        }
    }
    return 0;
}


/// @brief dispatch client connection and handler
/// @param s_args 
/// @param proxy_args 
/// @param client_session 
/// @param proxy_client 
/// @param total_output 
void dispatch_client(server_args s_args, client_args proxy_args, rdma_client *client_session, rdma_client *proxy_client, size_t total_output)
{
    int err;
    err = handle_client_connection(s_args, client_session, proxy_client);
    if (err)
    {
        goto client_closure;
    }
    DEBUG_LOG("Client ID %d (%s) Session is Ready \n", client_session->pdata.client_id, get_ip_address(client_session->cm_id));
    err = handle_requests(s_args, proxy_args, client_session, proxy_client, client_session->pdata.requests, total_output);
    if (err)
    {
        DEBUG_LOG("Failed to Handle Requests \n");
    }
client_closure:
    DEBUG_LOG("Client %d Done!, releasing proxy %d.\n", client_session->pdata.client_id, proxy_args.client_id);
    release_client_proxy(proxy_args.client_id);

    if (err)
    {
        fprintf(stderr, "Failed to Clean CLient Session \n");
    }
}

/// @brief Create A list of Connections to the Server
/// @param proxy_client_list 
/// @param connections 
/// @param server_address 
/// @param proxy_args 
/// @return 
int create_connections_pool(std::vector<rdma_client *> *proxy_client_list, int connections, std::string server_address, client_args proxy_args)
{
    int ret;
    for (int i = 0; i < connections; i++)
    {
        proxy_args.client_id = i;
        auto proxy_client = rdma_init_client(server_address, std::to_string(static_cast<int>(proxy_args.port)));
        if (!proxy_client)
        {
            DEBUG_LOG("rdma_init_client() %d failed \n", i);
            return 1;
        }
        ret = prepare_rdma_client(proxy_args, proxy_client);
        if (ret)
        {
            DEBUG_LOG("rdma_init_client() %d failed \n", i);
            return 1;
        }
        printf("RDMA Proxy Client # %d connected \n", i);
        proxy_client_list->emplace_back(proxy_client);
    }
    return 0;
}

int main(int argc, char *argv[])
{
    struct server_args s_args
    {
    };
    struct client_args proxy_args
    {
    };

    int ret, final_result, connections, connection_id;
    struct rdma_server *server;
    struct rdma_client *client_session;
    std::vector<std::thread> client_threads;

    std::vector<rdma_client *> proxy_client_list;
    std::string server_address, model_name;
    std::vector<std::vector<int>> output_shape;

    CLI::App app{"RDMA Proxy"};
    app.add_option("--server-address", server_address, "RDMA Server Address")->default_val("7.7.7.2");
    app.add_option("--target-port", proxy_args.port, "RDMA Target port")->default_val(20079);
    app.add_option("-r,--requests", proxy_args.requests, "Total Requests")->default_val(1000000);
    app.add_option("-t,--task", proxy_args.task, "Task")->default_val(RemoteTask::MIXTURE);
    app.add_flag("-s,--save-logs", proxy_args.save_logs, "Save Logs")->default_val(0);

    app.add_option("-p,--port", s_args.port, "RDMA Listen port")->default_val(20079);
    app.add_flag("-d,--debug", debug, "USE DEBUG")->default_val(0);
    app.add_option("-k,--keep-alive", s_args.keep_alive, "Keep Alive")->default_val(true);

    app.add_option("-c,--connections", connections, "RDMA Connections to Backend")->default_val(1);
    app.add_option("--model-name", model_name, "Model Name")->default_val("resnet50");

    CLI11_PARSE(app, argc, argv);
    
    if (read_outputshape_configurations(model_name, output_shape))
        return 1;

    size_t total_output = 1;
    for (size_t i = 0; i < output_shape.size(); i++)
    {
        for (size_t j = 0; j < output_shape[i].size(); j++)
        {
            total_output *= output_shape[i][j];
        }
    }
    busy = std::vector<bool>(connections);

    // Create connections pool
    ret = create_connections_pool(&proxy_client_list, connections, server_address, proxy_args);
    if (ret)
    {
        goto server_closure;
    }

    server = rdma_init_server(s_args.port);
    if (!server)
    {
        DEBUG_LOG("Cannot Init RDMA Device \n");
        goto server_closure;
    }
    else
    {
        DEBUG_LOG("RDMA device is ready \n");
    }
wait_client:
    printf("Wait for client \n");

    client_session = (rdma_client *)calloc(1, sizeof *client_session);
    client_session->cm_channel = server->cm_channel;

    ret = rdma_server_wait_for_client(server, client_session);
    if (ret)
    {
        final_result = 1;
        goto server_closure;
    }
    connection_id = get_connection_id(connections);
    proxy_args.client_id = connection_id;
    client_threads.emplace_back(std::thread(dispatch_client, s_args, proxy_args, client_session, proxy_client_list[connection_id], total_output));

    if (s_args.keep_alive)
    {
        DEBUG_LOG("Jump Up\n");

        goto wait_client;
    }

    for (std::thread &th : client_threads)
    {
        // If thread Object is Joinable then Join that thread.
        if (th.joinable())
            th.join();
    }

server_closure:
    // Close proxy
    for (int i = 0; i < connections; i++)
    {
        printf("%d \n", i);
        ret = rdma_free_client(proxy_client_list[i]);
        if (ret)
        {
            fprintf(stderr, "failure in free client %d, error %d\n", i, ret);
            return 1;
        }
    }
    ret = rdma_free_server(server);
    if (ret)
    {
        fprintf(stderr, "failure in free server, error %d\n", ret);
        return 1;
    }
    return final_result;
}