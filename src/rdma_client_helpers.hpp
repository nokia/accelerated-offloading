/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * RDMA Client Helpers:
 * Implements RDMA Client Common Functionalities
 *
 **/

#include "utils.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
extern int debug;

#define DEBUG_LOG \
    if (debug)    \
    printf

struct client_time_details
{
    size_t time_stamp;
    double client_time;
    double send_time;
    double wait_time;
};

/// @brief Compute Image Dimensions
/// @param model_type 
/// @param input_shape 
/// @return 
cv::Size get_image_size(ModelType model_type, std::vector<int> input_shape)
{
    switch (model_type)
    {
    case ModelType::PyTorchVisionClassification:
    case ModelType::PyTorchVisionDetection:
    case ModelType::PyTorchVisionSegmentation:
        return cv::Size(input_shape[1], input_shape[2]);

    case ModelType::TensorFlowVisionClassification:
    case ModelType::TensorFlowVisionDetection:
        return cv::Size(input_shape[0], input_shape[1]);
    default:
        return cv::Size(0, 0);
    }
}

/// @brief PreProcess Image
/// @param model_type 
/// @param image 
/// @param input_shape 
/// @return 
cv::Mat preprocess_image(ModelType model_type, std::string image, std::vector<int> input_shape)
{
    cv::Mat frame;
    frame = cv::imread(image.c_str());
    if (frame.empty())
    {
        std::cerr << "Input image load failed\n";
        return frame;
    }
    cv::Size image_size = get_image_size(model_type, input_shape);
    cv::resize(frame, frame, image_size, cv::InterpolationFlags::INTER_NEAREST);
    cv::cvtColor(frame, frame, cv::ColorConversionCodes::COLOR_BGR2RGB);
    frame.convertTo(frame, CV_32F, 1.0 / 255);
    cv::subtract(frame, cv::Scalar(0.485f, 0.456f, 0.406f), frame, cv::noArray(), -1);
    cv::divide(frame, cv::Scalar(0.229f, 0.224f, 0.225f), frame, 1, -1);

    switch (model_type)
    {
    case ModelType::PyTorchVisionClassification:
    case ModelType::PyTorchVisionDetection:
    case ModelType::PyTorchVisionSegmentation:
        frame = cv::dnn::blobFromImage(frame);
        break;
    case ModelType::TensorFlowVisionClassification:
    case ModelType::TensorFlowVisionDetection:
        break;
    default:
        cv::Mat r;
        return r;
    }
    return frame;
}

FILE *file = NULL;
size_t imgSize;
cv::Mat frame;
cv::Mat processed_frame;

/// @brief Prepare RDMA Request 
/// @param c_args 
/// @param client 
/// @param sge 
/// @param image 
/// @param input_shape 
/// @return 
int prepare_request(struct client_args c_args, struct rdma_client *client, struct ibv_sge *sge, std::string image, std::vector<int> input_shape)
{
    switch (c_args.task)
    {
    case RemoteTask::PING:
        /* Fill Request Bufer */
        DEBUG_LOG("Sending ping. \n");
        sge->length = sizeof(ping_request);
        break;
    case RemoteTask::CLASSIFICATION_RAW:
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
            auto c_request = (raw_request *)client->request_buf;
            c_request->task = RemoteTask::CLASSIFICATION_RAW;
            c_request->width = frame.size().width;
            c_request->height = frame.size().height;
            c_request->request_size = imgSize + 2 * sizeof(size_t) + sizeof(remote_request);
            memcpy(&c_request->image_data, frame.data, imgSize);
            std::cout <<"total request size: " << c_request->request_size <<"\n";
        }
        sge->length = imgSize + 2 * sizeof(size_t) + sizeof(remote_request);
        break;
    case RemoteTask::CLASSIFICATION_PROCESSED:
        if (processed_frame.empty())
        {
            processed_frame = preprocess_image(c_args.model_type, image, input_shape);
            imgSize = processed_frame.total() * processed_frame.elemSize();
            std::cout << imgSize << "," << processed_frame.elemSize() << "," << processed_frame.total() << "," << frame.elemSize() << std::endl;
            std::cout << processed_frame.size().width << "," << processed_frame.size().height << std::endl;
            auto c_request = (processed_request *)client->request_buf;
            c_request->task = RemoteTask::CLASSIFICATION_PROCESSED;
            c_request->request_size = imgSize + 2 * sizeof(size_t) + sizeof(remote_request);
            memcpy(&c_request->image_data, processed_frame.data, imgSize);
        }
        sge->length = imgSize + sizeof(remote_request);
        break;
    default:
        fprintf(stderr, "Invalid Task %d\n", static_cast<int>(c_args.task));
        return 1;
    }
    return 0;
}

/// @brief Save Client Logs
/// @param execution_time 
/// @param results_folder 
/// @param file_name 
void save_details(std::vector<client_time_details> execution_time, std::string results_folder, std::string file_name)
{
    results_folder = results_folder + "/";
    mkdir(results_folder.c_str(), 0700);
    std::ofstream myFile(results_folder + file_name + ".csv");

    // Send the column name to the stream
    myFile << "time_stamp"
           << ","
           << "client_time"
           << ","
           << "send_time"
           << ","
           << "wait_time"
           << "\n";

    // Send data to the stream
    for (size_t i = 0; i < execution_time.size(); ++i)
    {
        myFile << execution_time[i].time_stamp << ",";
        myFile << execution_time[i].client_time << ",";
        myFile << execution_time[i].send_time << ",";
        myFile << execution_time[i].wait_time << "\n";
    }

    // Close the file
    myFile.close();
}