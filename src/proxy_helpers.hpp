/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Proxy Helpers:
 * Implements Proxy Common Functionalities
 *
 **/

#include <stdio.h>
#include <vector>
#include <string.h>
#include "utils.h"

/// @brief Read Model OutputShape
/// @param model_name 
/// @param output_shape 
/// @return 
int read_outputshape_configurations(std::string model_name, std::vector<std::vector<int>>& output_shape){

	if (model_name.find("net") != std::string::npos)
	{
		output_shape.emplace_back(std::vector<int>{1000});
	}
	else if (model_name.find("yolov4") != std::string::npos)
	{
		output_shape.emplace_back(std::vector<int>{13, 13, 3, 85});
		output_shape.emplace_back(std::vector<int>{26, 26, 3, 85});
		output_shape.emplace_back(std::vector<int>{52, 52, 3, 85});
	}
    else if (model_name.find("deeplabv3") != std::string::npos)
	{
		output_shape.emplace_back(std::vector<int>{1, 21, 520, 520});
		output_shape.emplace_back(std::vector<int>{1, 21, 520, 520});
	}
	else
	{
		fprintf(stderr, "unknown model");
		return 1;
	}
	return 0;
}