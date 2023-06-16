/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * RDMA Common:
 * Implements RDMA Common Functionalities for Client and Server
 *
 **/

#pragma once

#include <stdio.h>
#include <cstdio>
#include <malloc.h>
#include <rdma/rdma_cma.h>
#include <string.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include "utils.h"
#include <iostream>

#define RESOLVE_TIMEOUT_MS 5000
#define RDMA_CQ_DEPTH 512
#define MAX_WR 8
#define MAX_SGE 2

/// @brief Private Data Copied Between RDMA client and Server
struct privatedata
{
    uint64_t buf_va;
    uint32_t buf_rkey;
    RemoteTask task; // Needed for non-mixed experiments and logging reasons.
    int requests;
    int client_id;
    int total_clients;
    bool save_logs;
    MemoryAddressing addressing;
};

/// @brief RDMA Server Device.
/// This structure for holding shared rdma configurations between clients.
struct rdma_server
{
    // shared event channel between all clients
    struct rdma_event_channel *cm_channel;
    struct rdma_cm_id *listen_id;
};

/// @brief This structure is used for holding client information.
struct rdma_client
{
    struct rdma_event_channel *cm_channel;
    struct rdma_cm_id *cm_id;
    struct ibv_pd *pd;                  // Protection Domain
    struct ibv_comp_channel *comp_chan; // Completetion Channel
    struct ibv_cq *cq;                  // Completetion Queue

    void *request_buf;
    size_t request_size;
    struct ibv_mr *request_mr; // Memory Region
    void *reply_buf;
    size_t reply_size;
    struct ibv_mr *reply_mr; // Memory Region

    struct privatedata pdata;
};

/// @brief Initialize RDMA Server Listeners
struct rdma_server *rdma_init_server(int port);

/// @brief Initialize Client Connection
/// @param address
/// @param port
struct rdma_client *rdma_init_client(std::string address, std::string port);

/// @brief Free Sever Resources
/// @param server
/// @return
int rdma_free_server(struct rdma_server *server);

/// @brief Free Client Resources
/// @param client
/// @return
int rdma_free_client(struct rdma_client *client);

/// @brief Wait (Block) for Connection Manager Event
/// @param cm_channel
/// @param  evt_type
/// @param  alt_evt
/// @return
int rdma_block_for_cm_event(struct rdma_event_channel *cm_channel, enum rdma_cm_event_type evt_type, enum rdma_cm_event_type alt_evt);

/// @brief Wait for Client Connection
/// @param server
/// @param client_connection
/// @return
int rdma_server_wait_for_client(struct rdma_server *server, struct rdma_client *client_session);

/// @brief Process Work Completion Event
/// @param client_connection
/// @param ack_count
/// @param rwc
/// @return
int rdma_process_work_completion(rdma_client *client_session, int ack_count = 0, ibv_wc *wc = NULL);

/// @brief Initialize RDMA Client Connection
/// @param c_args
/// @param client
/// @return
int prepare_rdma_client(struct client_args c_args, struct rdma_client *client);

/// @brief Load Internal IP address
/// @param cm_id 
/// @return 
char *get_ip_address(rdma_cm_id *cm_id);

int connect_gpu();