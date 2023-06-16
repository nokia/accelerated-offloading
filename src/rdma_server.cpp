/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * RDMA Server:
 * RDMA Server Executor
 *
 **/

#include "rdma_common.h"
#include "CLI11.hpp"
#include "computation.h"
#include <thread>
#include "rdma_server_helpers.hpp"
#include "monitor.hpp"
int debug;

#define DEBUG_LOG \
    if (debug)    \
    printf

extern Engine *engine;

/// @brief Handle Client Conenction and allocate needed resources.
/// @param s_args 
/// @param client_session 
/// @return 
int handle_client_connection(server_args s_args, struct rdma_client *client_session)
{
    struct ibv_qp_init_attr qp_attr = {};
    struct privatedata rep_pdata;
    struct rdma_conn_param conn_param = {};
    int err;

    struct ibv_sge sge; // Scatter Gather Items
    struct ibv_recv_wr recv_wr = {};
    struct ibv_recv_wr *bad_recv_wr;

    /* Create verbs objects now that we know which device to use */
    // 1. Create Proctection Domain
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
    // 3. Create Completetion Queue This queue is with completeion channel to listen to events.
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
    client_session->request_size = MAX_MESSAGE;
    client_session->reply_size = MAX_MESSAGE;

    err = allocate_client_connection_memory(s_args.addressing, client_session->request_buf, client_session->request_size, client_session->reply_buf, client_session->reply_size);
    if (err)
    {
        DEBUG_LOG("Failed to allocate memory \n");
        return err;
    }
    // 6. Register Memory region
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

    // 9. Post receive before accepting connection becuase we are using send
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
    rep_pdata.addressing = s_args.addressing;

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

/// @brief Handle Clients Requests. Number of Requests is negotiated before this
/// @param s_args 
/// @param client_session 
/// @param total_requests 
/// @param results_folder 
/// @return 
int handle_requests(struct server_args s_args, struct rdma_client *client_session, int total_requests, std::string results_folder)
{
    int ret;
    struct ibv_sge sge; // Scatter Gather Items
    struct ibv_send_wr send_wr = {};
    struct ibv_send_wr *bad_send_wr;
    struct ibv_recv_wr recv_wr = {};
    struct ibv_recv_wr *bad_recv_wr;
    RemoteTask current_task;
    std::vector<time_details> execution_time;
    Monitor m(s_args.monitor_interval);
    for (int i = 0; i < total_requests; i++)
    {
        struct time_details t_details;
        /* Wait for receive completion */
        ret = rdma_process_work_completion(client_session);
        if (ret)
        {
            DEBUG_LOG("rdma_process_work_completion() failed \n");
            return ret;
        }
        auto start = std::chrono::high_resolution_clock::now();
        if (client_session->pdata.task == RemoteTask::MIXTURE)
        {
            // This have a small overhead as data is copied from GPU.
            if (s_args.addressing == MemoryAddressing::CUDA_ADDRESSING)
            {
                cudaMemcpy(&current_task, client_session->request_buf, sizeof(remote_request), cudaMemcpyDeviceToHost);
            }
            else
            {
                memcpy(&current_task, client_session->request_buf, sizeof(remote_request));
            }
        }
        else
        {
            current_task = client_session->pdata.task;
        }
        /* Process Request based on type */
        ret = process_request(client_session->pdata.client_id, current_task, s_args.addressing, client_session->request_buf, client_session->reply_buf, &t_details);
        if (ret)
        {
            DEBUG_LOG("Compute Result Failed %d \n", ret);
            return ret;
        }
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
        switch (current_task)
        {
        case RemoteTask::PING:
            sge.length = sizeof(ping_reply);
            break;
        default:

            sge.length = engine->total_output_size * sizeof(float);
            break;
        }
        sge.lkey = client_session->reply_mr->lkey;
        send_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        send_wr.wr.rdma.rkey = ntohl(client_session->pdata.buf_rkey);
        send_wr.wr.rdma.remote_addr = be64toh(client_session->pdata.buf_va);
        // This flag creates a CQ Event
        send_wr.send_flags = IBV_SEND_SIGNALED;
        send_wr.sg_list = &sge;
        send_wr.num_sge = 1;
        auto send_start = std::chrono::high_resolution_clock::now();
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

        auto end = std::chrono::high_resolution_clock::now();
        t_details.send_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - send_start).count() * 1e-6;
        t_details.server_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;
        execution_time.push_back(t_details);
    }
    if (client_session->pdata.save_logs)
    {
        std::string experiment_name = "rserver_" + std::to_string(static_cast<int>(client_session->pdata.client_id)) + "_";
        experiment_name = experiment_name + std::to_string(static_cast<int>(s_args.addressing)) + "_";
        experiment_name = experiment_name + std::to_string(static_cast<int>(client_session->pdata.task)) + "_";
        experiment_name = experiment_name + std::to_string(client_session->pdata.requests);
        save_details(execution_time, results_folder+"_"+std::to_string(client_session->pdata.total_clients), experiment_name);
        m.save_monitoring_state(results_folder+"_"+std::to_string(client_session->pdata.total_clients), experiment_name);
    }
    return 0;
}


/// @brief dispatch client connection and handler
/// @param s_args 
/// @param client_session 
/// @param results_folder 
void dispatch_client(server_args s_args, rdma_client *client_session, std::string results_folder)
{
    int err;
    err = handle_client_connection(s_args, client_session);
    if (err)
    {
        goto client_closure;
    }
    DEBUG_LOG("Client ID %d (%s) Session is Ready \n", client_session->pdata.client_id, get_ip_address(client_session->cm_id));
    err = handle_requests(s_args, client_session, client_session->pdata.requests, results_folder);
    if (err)
    {
        DEBUG_LOG("Failed to Handle Requests \n");
    }
client_closure:
    DEBUG_LOG("Client %d Done!\n", client_session->pdata.client_id);
    err = rdma_free_client_connection(s_args, client_session);
    if (err)
    {
        fprintf(stderr, "Failed to Clean CLient Session \n");
    }
}

int main(int argc, char *argv[])
{
    struct server_args s_args
    {
    };

    int ret, final_result, streams;
    struct rdma_server *server;
    struct rdma_client *client_session;
    std::string results_folder, engine_file, classes_file;
    std::vector<std::thread> client_threads;
    int high_priority_clients;
    CLI::App app{"RDMA Server"};

    app.add_option("-p,--port", s_args.port, "RDMA Listen port")->default_val(20079);
    app.add_option("-a,--addressing", s_args.addressing, "Memory Addressing")->default_val(MemoryAddressing::HOST_ADDRESSING);
    app.add_flag("-d,--debug", debug, "USE DEBUG")->default_val(0);
    app.add_option("-k,--keep-alive", s_args.keep_alive, "Keep Alive")->default_val(true);
    app.add_option("-i,--monitor-interval", s_args.monitor_interval, "Monitor Interval")->default_val(20);
    app.add_option("-f,--folder", results_folder, "Logs Folder")->default_val("results");

    app.add_option("--classes-file", classes_file, "Classes File")->default_val("models/imagenet_classes.txt");
    app.add_option("--engine-file", engine_file, "Tensor RT Engine File")->default_val("models/resnet50.trt");

    app.add_option("--streams", streams, "Tensor RT Streams")->default_val(10);
    app.add_option("--priority-clients", high_priority_clients, "High Priority Client")->default_val(0);

    CLI11_PARSE(app, argc, argv);
    // Initialize Server Listener
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
    
    // Initialize GPU and DNN model
    if (connect_gpu())
    {
        goto server_closure;
    }
    engine = new Engine(engine_file, classes_file, high_priority_clients, streams);
    printf("Model Ready \n");
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
    client_threads.emplace_back(std::thread(dispatch_client, s_args, client_session, results_folder));

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
    ret = rdma_free_server(server);
    if (ret)
    {
        fprintf(stderr, "failure in free server, error %d\n", ret);
        return 1;
    }
    return final_result;
}
