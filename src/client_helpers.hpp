
/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Client Helpers:
 * Implements common client functionalities such as logging and preprocessing
 *
 **/

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>
#include <string.h>

extern int debug;

#define DEBUG_LOG \
    if (debug)    \
    printf

/// @brief saves client total latency results
/// @param total_latency
/// @param results_folder
/// @param file_name
void save_details(std::vector<double> total_latency, std::string results_folder, std::string file_name)
{
    results_folder = results_folder + "/";
    mkdir(results_folder.c_str(), 0700);
    std::ofstream myFile(results_folder + file_name + ".csv");

    // Send the column name to the stream
    myFile << "client_time"
           << "\n";

    // Send data to the stream
    for (size_t i = 0; i < total_latency.size(); ++i)
    {
        myFile << total_latency[i] << "\n";
    }

    // Close the file
    myFile.close();
}

cv::Mat preprocess_image(std::string image, std::vector<int> input_shape)
{
    cv::Mat frame;
    frame = cv::imread(image.c_str());
    if (frame.empty())
    {
        std::cerr << "Input image load failed\n";
        return frame;
    }
    cv::resize(frame, frame, cv::Size(input_shape[1], input_shape[2]), cv::InterpolationFlags::INTER_NEAREST);
    cv::cvtColor(frame, frame, cv::ColorConversionCodes::COLOR_BGR2RGB);
    frame.convertTo(frame, CV_32F, 1.0 / 255);
    cv::subtract(frame, cv::Scalar(0.485f, 0.456f, 0.406f), frame, cv::noArray(), -1);
    cv::divide(frame, cv::Scalar(0.229f, 0.224f, 0.225f), frame, 1, -1);
    frame = cv::dnn::blobFromImage(frame);

    return frame;
}

std::vector<std::string> classes;


/// @brief prints results of classification models
/// @param reply_buffer 
/// @param output_shape 
void print_results_vision_classification(void *reply_buffer, std::vector<std::vector<int>> output_shape)
{
    assert(output_shape.size() == 1);
    assert(output_shape[0].size() == 1);
    std::vector<float> inf_output(classes.size());
    memcpy(inf_output.data(), reply_buffer, inf_output.size() * sizeof(float));
    std::cout << "Read " << inf_output.size() << " class" << std::endl;
    std::transform(inf_output.begin(), inf_output.end(), inf_output.begin(), [](float val)
                   { return std::exp(val); });
    double sum = std::accumulate(inf_output.begin(), inf_output.end(), 0.0);
    std::vector<int> indices(classes.size());
    std::iota(indices.begin(), indices.end(), 0); // generate sequence 0, 1, 2, 3, ..., 999
    std::sort(indices.begin(), indices.end(), [&inf_output](int i1, int i2)
              { return inf_output[i1] > inf_output[i2]; });
    // print Top 3 results
    for (int i = 0; i < 3; i++)
    {
        DEBUG_LOG("class: %s", classes[indices[i]].c_str());
        DEBUG_LOG(" | confidence: %f ", 100 * inf_output[indices[i]] / sum);
        DEBUG_LOG("| index: %d \n", indices[i]);
    }
}

/// @brief prints results of detection models
/// @param reply_buffer 
/// @param output_shape 
void print_results_vision_detection(void *reply_buffer, std::vector<std::vector<int>> output_shape)
{
    DEBUG_LOG("Vision Detection is called \n");
    std::vector<std::vector<float>> output(output_shape.size());
    size_t start = 0;
    for (size_t i = 0; i < output_shape.size(); i++)
    {
        size_t s = 1;
        for (size_t x = 0; x < output_shape[i].size(); x++)
            s *= output_shape[i][x];

        output[i] = std::vector<float>(s);
        memcpy(output[i].data(), reply_buffer + start, s * sizeof(float));
        start += (s * sizeof(float));
        std::cout << output[i][0] << std::endl;
    }
}

/// @brief print request results
/// @param c_args 
/// @param reply_buffer 
/// @param classes_file 
/// @param output_shape 
/// @return 
int print_results(struct client_args c_args, void *reply_buffer, std::string classes_file, std::vector<std::vector<int>> output_shape)
{

    std::string class_name;
    switch (c_args.task)
    {
    case RemoteTask::PING:
        DEBUG_LOG("Ping. \n");
        break;
    case RemoteTask::CLASSIFICATION_RAW:
    case RemoteTask::CLASSIFICATION_PROCESSED:
        if (classes.size() == 0)
        {
            std::ifstream classes_file_stream(classes_file);
            if (!classes_file_stream.good())
            {
                std::cerr << "ERROR: can't read file with classes names.\n";
                return 1;
            }
            while (std::getline(classes_file_stream, class_name))
            {
                classes.push_back(class_name);
            }
        }
        switch (c_args.model_type)
        {
        case ModelType::PyTorchVisionClassification:
        case ModelType::TensorFlowVisionClassification:
            print_results_vision_classification(reply_buffer, output_shape);
            break;
        case ModelType::PyTorchVisionSegmentation:
        case ModelType::TensorFlowVisionDetection:
        case ModelType::PyTorchVisionDetection:
            print_results_vision_detection(reply_buffer, output_shape);
            break;
        default:
            return 1;
        }
        break;
    default:
        fprintf(stderr, "Invalid Task %d\n", static_cast<int>(c_args.task));
        return 1;
    }
    return 0;
}

/// @brief Reads DNN model configurations
/// @param c_args 
/// @param model_name 
/// @param classes_file 
/// @param input_shape 
/// @param output_shape 
/// @return 
int read_model_configurations(client_args *c_args, std::string model_name, std::string &classes_file, std::vector<int> &input_shape, std::vector<std::vector<int>> &output_shape)
{

    if (model_name.find("net") != std::string::npos)
    {
        input_shape.emplace_back(3);
        input_shape.emplace_back(224);
        input_shape.emplace_back(224);
        output_shape.emplace_back(std::vector<int>{1000});
        c_args->model_type = ModelType::PyTorchVisionClassification;
        classes_file = "models/imagenet_classes.txt";
    }
    else if (model_name.find("yolov4") != std::string::npos)
    {
        input_shape.emplace_back(416);
        input_shape.emplace_back(416);
        input_shape.emplace_back(3);
        output_shape.emplace_back(std::vector<int>{13, 13, 3, 85});
        output_shape.emplace_back(std::vector<int>{26, 26, 3, 85});
        output_shape.emplace_back(std::vector<int>{52, 52, 3, 85});
        c_args->model_type = ModelType::TensorFlowVisionDetection;
        classes_file = "models/imagenet_classes.txt";
    }
    else if (model_name.find("deeplabv3") != std::string::npos)
    {
        input_shape.emplace_back(520);
        input_shape.emplace_back(520);
        input_shape.emplace_back(3);
        output_shape.emplace_back(std::vector<int>{1, 21, 520, 520});
        output_shape.emplace_back(std::vector<int>{1, 21, 520, 520});
        c_args->model_type = ModelType::PyTorchVisionSegmentation;
        classes_file = "models/imagenet_classes.txt";
    }
    else
    {
        fprintf(stderr, "unknown model");
        return 1;
    }
    return 0;
}