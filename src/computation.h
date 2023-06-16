/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Computation.h:
 * Headers for computation functions
 *
 **/

#pragma once

#include "utils.h"
#include <iostream>
#include <fstream>
#include "cuda.h"
#include "trt_engine.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <vector>
#include <chrono>
#include <sys/stat.h>

struct time_details
{
    double server_time = 0;
    double decode_time = 0;
    double pre_copy_time = 0;
    double preprocessing = 0;
    double mid_copy_time = 0;
    double inference_time = 0;
    double post_copy_time = 0;
    double total_compute = 0;
    double send_time = 0;
};

/// @brief Process Request based on Request Type
/// @param client_id 
/// @param task 
/// @param addressing 
/// @param request_buf 
/// @param reply_buf 
/// @param t_details 
/// @return 
int process_request(int client_id, RemoteTask task, MemoryAddressing addressing, void *request_buf, void *reply_buf, time_details *t_details);

/// @brief Save Server side logs
/// @param execution_time 
/// @param results_folder 
/// @param file_name 
void save_details(std::vector<time_details> execution_time, std::string results_folder, std::string file_name);