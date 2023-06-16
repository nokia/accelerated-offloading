/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Utils:
 * Contains base classes for the whole project
 **/

#pragma once

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#define MAX_MESSAGE 1024 * 1024 * 100

enum class MemoryAddressing
{
    HOST_ADDRESSING = 0,
    CUDA_ADDRESSING = 1
};


enum class RemoteTask
{
    PING = 0,
    CLASSIFICATION_RAW = 1,
    CLASSIFICATION_PROCESSED = 2,
    MIXTURE = 3,
};

enum class ModelType
{
    PyTorchVisionClassification = 000,
    PyTorchVisionDetection = 001,
    PyTorchVisionSegmentation = 002,
    TensorFlowVisionClassification = 100,
    TensorFlowVisionDetection = 101,
};

struct client_args
{
    int requests;
    RemoteTask task;
    ModelType model_type;
    bool save_logs;
    int client_id;
    int port;
    int total_clients;
};

struct server_args
{
    MemoryAddressing addressing;
    bool keep_alive;
    int port;
    int monitor_interval;
};

/*
This structure is used for holding request information.
*/
struct remote_request
{
    RemoteTask task;
    size_t request_size;
};

struct ping_request : remote_request
{
    bool data;
};

struct raw_request : remote_request
{
    size_t width;
    size_t height;
    uintptr_t image_data;
};

struct processed_request : remote_request
{
    uintptr_t image_data;
};

/*
This structure is used for holding reply information.
*/
struct ping_reply
{
    bool result;
};