/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * RDMA Common:
 * Implements RDMA Common Functionalities for Client and Server
 *
 **/

#include "rdma_common.h"

extern int debug;

#define DEBUG_LOG \
    if (debug)    \
    printf

struct rdma_server *rdma_init_server(int port)
{
    struct rdma_server *server;
    int ret_val;
    struct sockaddr_in sin;

    server = (rdma_server *)calloc(1, sizeof *server);

    if (!server)
    {
        DEBUG_LOG("rdma_server memory allocation failed\n");
        goto clean;
    }

    server->cm_channel = rdma_create_event_channel();
    if (!server->cm_channel)
    {
        DEBUG_LOG("rdma_create_event_channel failed\n");
        goto clean;
    }
    ret_val = rdma_create_id(server->cm_channel, &server->listen_id, NULL, RDMA_PS_TCP);
    if (ret_val)
    {
        DEBUG_LOG("rdma_server memory allocation failed\n");
        goto clean;
    }

    sin.sin_family = AF_INET;
    sin.sin_port = htons(port); // Fixed Port
    sin.sin_addr.s_addr = INADDR_ANY;

    /* Bind to local port and listen for connection request */
    ret_val = rdma_bind_addr(server->listen_id, (struct sockaddr *)&sin);
    if (ret_val)
    {
        DEBUG_LOG("rdma_bind_addr failed\n");
        goto clean;
    }

    // Wait for connected request
    ret_val = rdma_listen(server->listen_id, 8);
    if (ret_val)
    {
        DEBUG_LOG("rdma_listen failed\n");
        goto clean;
    }

    return server;

clean:
    ret_val = rdma_free_server(server);
    if (ret_val)
    {
        fprintf(stderr, "failure in free server, error %d\n", ret_val);
    }
    return NULL;
}

struct rdma_client *rdma_init_client(std::string address, std::string port)
{
    int err, n;
    struct rdma_client *client;
    struct addrinfo *res, *t;
    struct addrinfo hints = {
        .ai_family = AF_INET,
        .ai_socktype = SOCK_STREAM};

    client = (rdma_client *)calloc(1, sizeof *client);

    /* Create CM needed structure */
    client->cm_channel = rdma_create_event_channel();
    if (!client->cm_channel)
        goto clean;

    err = rdma_create_id(client->cm_channel, &client->cm_id, NULL, RDMA_PS_TCP);
    if (err)
    {
        goto clean;
    }

    n = getaddrinfo(address.c_str(), port.c_str(), &hints, &res);
    if (n < 0)
        goto clean;

    /*
     * Resolve server address and route.
     * This is done in a loop in the case of mulitple addresses.
     */
    // step 1: do action
    for (t = res; t; t = t->ai_next)
    {
        err = rdma_resolve_addr(client->cm_id, NULL, t->ai_addr, RESOLVE_TIMEOUT_MS);
        if (!err)
            break;
    }
    if (err)
        goto clean;

    err = rdma_block_for_cm_event(client->cm_channel, RDMA_CM_EVENT_ADDR_RESOLVED, RDMA_CM_EVENT_ADDR_ERROR);
    if (err)
        goto clean;

    err = rdma_resolve_route(client->cm_id, RESOLVE_TIMEOUT_MS);
    if (err)
        goto clean;

    err = rdma_block_for_cm_event(client->cm_channel, RDMA_CM_EVENT_ROUTE_RESOLVED, RDMA_CM_EVENT_ROUTE_ERROR);
    if (err)
    {
        goto clean;
    }
    return client;
clean:
    err = rdma_free_client(client);
    return NULL;
}

int rdma_block_for_cm_event(struct rdma_event_channel *cm_channel, enum rdma_cm_event_type evt_type, enum rdma_cm_event_type alt_evt)
{
    struct rdma_cm_event *event;
    int ret;
    do
    {
        ret = rdma_get_cm_event(cm_channel, &event);
        if (event->event == evt_type)
        {
            ret = rdma_ack_cm_event(event);
            return ret;
        }
        else if (event->event == alt_evt)
        {
            DEBUG_LOG("Wrong Event %d \n", alt_evt);
            ret = rdma_ack_cm_event(event);
            return 1;
        }
        else
        {
            std::cerr << "CM Event Error" << event->event << std::endl;
        }

    } while (!ret);
    return 1;
}

int rdma_server_wait_for_client(struct rdma_server *server, struct rdma_client *client_connection)
{
    int ret_val;
    struct rdma_cm_event *event;

    do
    {
        ret_val = rdma_get_cm_event(server->cm_channel, &event);
        if (event->event == RDMA_CM_EVENT_CONNECT_REQUEST)
        {
            client_connection->cm_id = event->id;
            memcpy(&client_connection->pdata, event->param.conn.private_data, sizeof(privatedata));
            ret_val = rdma_ack_cm_event(event);
            return ret_val;
        }
        else if (event->event == RDMA_CM_EVENT_ESTABLISHED || event->event == RDMA_CM_EVENT_DISCONNECTED)
        {
            ret_val = rdma_ack_cm_event(event);
            if (ret_val)
                return ret_val;
        }
        else
        {
            std::cerr << "CM Event error, recived " << event->event << " Instead of " << RDMA_CM_EVENT_CONNECT_REQUEST << std::endl;
        }
    } while (!ret_val);
    return 1;
}

int prepare_rdma_client(struct client_args c_args, struct rdma_client *client)
{
    int err;
    struct privatedata rep_pdata;
    struct ibv_qp_init_attr qp_attr = {};
    struct rdma_cm_event *event;
    struct rdma_conn_param conn_param = {};

    /* Create verbs objects now that we know which device to use
     * This is part of the iniitation.
     */
    // 1. Create protection Domains
    client->pd = ibv_alloc_pd(client->cm_id->verbs);
    if (!client->pd)
    {
        DEBUG_LOG("ibv_alloc_pd() failure");
        goto prep_err;
    }
    // 2. create completetion channel,
    // This channel is used for automatic notification instead of polling.
    client->comp_chan = ibv_create_comp_channel(client->cm_id->verbs);
    if (!client->comp_chan)
    {
        DEBUG_LOG("ibv_create_comp_channel() failure");
        goto prep_err;
    }
    // 3. Create thc completetion queue
    client->cq = ibv_create_cq(client->cm_id->verbs, RDMA_CQ_DEPTH, NULL, client->comp_chan, 0);
    if (!client->cq)
    {
        DEBUG_LOG("ibv_create_cq() failure");
        goto prep_err;
    }

    // 4. Set to 0 to allow for all events
    // Notify CQ is needed per request
    if (ibv_req_notify_cq(client->cq, 0))
    {
        DEBUG_LOG("ibv_req_notify_cq() failure");
        goto prep_err;
    }

    // 5. Allocate buffer memory
    // 5.1 Fix Buffers Sizes for all operations
    client->request_size = MAX_MESSAGE;
    client->reply_size = MAX_MESSAGE;

    // 5.2 Allocate Request Buffer
    client->request_buf = calloc(1, client->request_size);
    if (!client->request_buf)
    {
        DEBUG_LOG("request buffer allocation failure");
        return 1;
    }

    // 5.3 Allocate Reply Buffer
    client->reply_buf = calloc(1, client->reply_size);
    if (!client->reply_buf)
    {
        DEBUG_LOG("reply buffer allocation failure");
        return 1;
    }

    // 6. Register memory region
    // 6.1 Register Request Buffer
    client->request_mr = ibv_reg_mr(client->pd, client->request_buf, client->request_size, IBV_ACCESS_LOCAL_WRITE);
    if (!client->request_mr)
    {
        DEBUG_LOG("request buffer ibv_reg_mr() failure");
        goto prep_err;
    }
    // 6.1 Register Reply Buffer
    client->reply_mr = ibv_reg_mr(client->pd, client->reply_buf, client->reply_size,
                                  (
                                      IBV_ACCESS_LOCAL_WRITE |
                                      IBV_ACCESS_REMOTE_READ |
                                      IBV_ACCESS_REMOTE_WRITE));
    if (!client->reply_mr)
    {
        DEBUG_LOG("reply buffer ibv_reg_mr() failure");
        goto prep_err;
    }

    /*
     * Configure and Create the queue pair
     */
    qp_attr.cap.max_send_wr = MAX_WR;
    qp_attr.cap.max_send_sge = MAX_SGE;
    qp_attr.cap.max_recv_wr = MAX_WR;
    qp_attr.cap.max_recv_sge = MAX_SGE;

    qp_attr.send_cq = client->cq;
    qp_attr.recv_cq = client->cq;
    qp_attr.qp_type = IBV_QPT_RC; // Reliable Communication
    // qp_attr.sq_sig_all = 1; // Default is zero to make all signaling on demand.

    err = rdma_create_qp(client->cm_id, client->pd, &qp_attr);
    if (err)
    {
        DEBUG_LOG("rdma_create_qp() failure");
        goto prep_err;
    }

    // Configure memory data for remote write
    // Start Connection
    // 11. Allocate Buffer and key
    rep_pdata.buf_va = be64toh((uintptr_t)client->reply_buf);
    rep_pdata.buf_rkey = htonl(client->reply_mr->rkey);
    rep_pdata.task = c_args.task;
    rep_pdata.requests = c_args.requests;
    rep_pdata.save_logs = c_args.save_logs;
    rep_pdata.client_id = c_args.client_id;
    rep_pdata.total_clients = c_args.total_clients;

    conn_param.initiator_depth = 1;
    conn_param.private_data = &rep_pdata;
    conn_param.private_data_len = sizeof rep_pdata;

    // conn_param.retry_count = 7;

    /* Connect to server */

    err = rdma_connect(client->cm_id, &conn_param);
    if (err)
    {
        DEBUG_LOG("rdma_connect() failure");
        goto prep_err;
    }

    do
    {
        err = rdma_get_cm_event(client->cm_channel, &event);
        if (event->event == RDMA_CM_EVENT_ESTABLISHED)
        {
            memcpy(&client->pdata, event->param.conn.private_data, sizeof(privatedata));
            err = rdma_ack_cm_event(event);
            break;
        }
    } while (!err);
    if (!err)
    {
        return 0;
    }

prep_err:
    return 1;
}

int rdma_free_server(struct rdma_server *server)
{
    DEBUG_LOG("Closing RDMA Device\n");
    int ret;
    if (server)
    {
        if (server->listen_id)
        {
            DEBUG_LOG("rdma_destroy_id(%p)\n", server->listen_id);
            ret = rdma_destroy_id(server->listen_id);
            if (ret)
            {
                fprintf(stderr, "failure in rdma_destroy_id(), error %d\n", ret);
                return ret;
            }
        }
        if (server->cm_channel)
        {
            DEBUG_LOG("rdma_destroy_event_channel(%p)\n", server->cm_channel);
            rdma_destroy_event_channel(server->cm_channel);
        }
        free(server);
    }
    return 0;
}

int rdma_free_client(struct rdma_client *client)
{

    DEBUG_LOG("Closing RDMA Client \n");
    int ret;
    if (client)
    {
        ret = rdma_disconnect(client->cm_id);
        if (ret)
        {
            fprintf(stderr, "failure in ibv_destroy_qp(), error %d\n", ret);
        }
        if (client->cm_id->qp)
        {
            ret = ibv_destroy_qp(client->cm_id->qp);
            if (ret)
            {
                fprintf(stderr, "failure in ibv_destroy_qp(), error %d\n", ret);
            }
        }
        // Deregister and free Memory Region

        if (client->request_mr)
        {
            ret = ibv_dereg_mr(client->request_mr);
            if (ret)
            {
                fprintf(stderr, "failure in ibv_dereg_mr(), error %d\n", ret);
            }
            if (client->request_buf)
            {
                free(client->request_buf);
            }
        }

        if (client->reply_mr)
        {
            ret = ibv_dereg_mr(client->reply_mr);
            if (ret)
            {
                fprintf(stderr, "failure in ibv_dereg_mr(), error %d\n", ret);
            }
            if (client->reply_buf)
            {
                free(client->reply_buf);
            }
        }
        // Destroy CQ
        if (client->cq)
        {
            ret = ibv_destroy_cq(client->cq);
            if (ret)
            {
                fprintf(stderr, "failure in ibv_destroy_cq(), error %d\n", ret);
            }
        }
        // Destroy comp_chan
        if (client->comp_chan)
        {
            ret = ibv_destroy_comp_channel(client->comp_chan);
            if (ret)
            {
                fprintf(stderr, "failure in ibv_destroy_comp_channel(), error %d\n", ret);
            }
        }

        // Destroy PD
        if (client->pd)
        {
            ret = ibv_dealloc_pd(client->pd);
            if (ret)
            {
                fprintf(stderr, "failure in ibv_dealloc_pd(), error %d\n", ret);
            }
        }

        if (client->cm_id)
        {
            DEBUG_LOG("rdma_destroy_id(%p)\n", client->cm_id);
            ret = rdma_destroy_id(client->cm_id);
            if (ret)
            {
                fprintf(stderr, "failure in rdma_destroy_id(), error %d\n", ret);
                return ret;
            }
        }
        if (client->cm_channel)
        {
            DEBUG_LOG("rdma_destroy_event_channel(%p)\n", client->cm_channel);
            rdma_destroy_event_channel(client->cm_channel);
        }
        free(client);
    }
    return 0;
}

int rdma_process_work_completion(rdma_client *client_connection, int ack_count, ibv_wc *rwc)
{
    void *cq_context;
    struct ibv_cq *evt_cq;
    struct ibv_wc wc; // Work Completion

    int ret;
    ret = ibv_get_cq_event(client_connection->comp_chan, &evt_cq, &cq_context);
    if (ret)
    {
        DEBUG_LOG("ibv_get_cq_event() failure \n");
        return 1;
    }
    if (ibv_req_notify_cq(client_connection->cq, 0))
    {
        DEBUG_LOG("ibv_req_notify_cq() Failed \n");
        return 1;
    }
    ret = ibv_poll_cq(client_connection->cq, 1, &wc);
    if (ret < 1)
    {
        DEBUG_LOG("ibv_poll_cq() failure \n");
        return 1;
    }
    if (wc.status != IBV_WC_SUCCESS)
    {
        DEBUG_LOG("Wrong WC STATUS %d \n", wc.status);
        return 1;
    }
    if (ack_count)
    {
        ibv_ack_cq_events(client_connection->cq, ack_count);
    }
    if (rwc)
    {
        memccpy(rwc, &wc, 1, sizeof wc);
    }
    return 0;
}

char *get_ip_address(rdma_cm_id *cm_id)
{
    struct sockaddr *client_add;
    struct sockaddr_in *addr_in;
    client_add = rdma_get_peer_addr(cm_id);
    addr_in = (struct sockaddr_in *)client_add;
    // printf("IP address: %s\n", );
    return inet_ntoa(addr_in->sin_addr);
}