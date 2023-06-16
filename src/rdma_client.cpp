/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * RDMA Client:
 * RDMA Client Implementation
 *
 **/

#include "rdma_common.h"
#include "CLI11.hpp"
#include "rdma_client_helpers.hpp"
#include "client_helpers.hpp"

int debug;

#define DEBUG_LOG \
	if (debug)    \
	printf

std::string image, results_folder;

/// @brief Send Requests to servers. Total request is negotiated before this step
/// @param c_args 
/// @param client 
/// @param total_requests 
/// @param classes_file 
/// @param input_shape 
/// @param output_shape 
/// @return 
int send_requests(struct client_args c_args, struct rdma_client *client, int total_requests, std::string classes_file, std::vector<int> input_shape, std::vector<std::vector<int>> output_shape)
{
	struct ibv_wc wc; // Work Completion
	struct ibv_sge sge;
	struct ibv_send_wr send_wr = {};
	struct ibv_send_wr *bad_send_wr;
	struct ibv_recv_wr recv_wr = {};
	struct ibv_recv_wr *bad_recv_wr;
	int ret;

	std::vector<client_time_details> execution_time(total_requests);
	for (int i = 0; i < total_requests; i++)
	{
		struct client_time_details t_details;
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

		if (prepare_request(c_args, client, &sge, image, input_shape))
			return 1;

		sge.addr = (uintptr_t)client->request_buf;
		sge.lkey = client->request_mr->lkey;

		send_wr.wr_id = 1;
		send_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
		send_wr.wr.rdma.rkey = ntohl(client->pdata.buf_rkey);
		send_wr.wr.rdma.remote_addr = be64toh(client->pdata.buf_va);
		// This flag creates a CQ Event
		send_wr.send_flags = IBV_SEND_SIGNALED;
		send_wr.sg_list = &sge;
		send_wr.num_sge = 1;

		auto start = std::chrono::high_resolution_clock::now();
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
			return ret;
		}
		auto send_done = std::chrono::high_resolution_clock::now();
		/* Wait for receive completion (You can asser the wc.wr_id)*/
		ret = rdma_process_work_completion(client, 2, &wc);
		if (ret)
		{
			DEBUG_LOG("rdma_process_work_completion() failed\n");
			return ret;
		}
		if (debug)
		{
			if (print_results(c_args, client->reply_buf, classes_file, output_shape))
				return 1;
		}
		auto end = std::chrono::high_resolution_clock::now();
		t_details.client_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;
		t_details.send_time = std::chrono::duration_cast<std::chrono::nanoseconds>(send_done - start).count() * 1e-6;
		t_details.wait_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - send_done).count() * 1e-6;
		t_details.time_stamp = std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch()).count();
		execution_time[i] = t_details;
	}
	double sum = 0;
	for (int i = 0; i < c_args.requests; i++)
		sum += execution_time[i].client_time;
	printf("Request took %f ms per request \n", sum / c_args.requests);
	if (c_args.save_logs)
	{
		std::string experiment_name = "rclient_" + std::to_string(static_cast<int>(c_args.client_id)) + "_";
		experiment_name = experiment_name + std::to_string(static_cast<int>(client->pdata.addressing)) + "_";
		experiment_name = experiment_name + std::to_string(static_cast<int>(c_args.task)) + "_";
		experiment_name = experiment_name + std::to_string(c_args.requests);
		save_details(execution_time, results_folder, experiment_name);
	}
	return 0;
}

int main(int argc, char *argv[])
{
	struct client_args c_args
	{
	};

	struct rdma_client *client;
	int ret, final_result;
	std::string host, classes_file, model_name;
	std::vector<int> input_shape;
	std::vector<std::vector<int>> output_shape;
	CLI::App app{"RDMA Client"};

	app.add_option("--host", host, "RDMA Host")->default_val("7.7.7.2");
	app.add_option("-p,--port", c_args.port, "RDMA Server port")->default_val(20079);
	app.add_flag("-d,--debug", debug, "USE DEBUG")->default_val(0);
	app.add_option("-r,--requests", c_args.requests, "Total Requests")->default_val(1);
	app.add_option("-t,--task", c_args.task, "Task")->default_val(RemoteTask::PING);
	app.add_option("-i,--image", image, "Image")->default_val("dog.jpg");
	app.add_flag("-s,--save-logs", c_args.save_logs, "Save Logs")->default_val(0);
	app.add_option("-f,--folder", results_folder, "Logs Folder")->default_val("results");
	app.add_option("--client-id", c_args.client_id, "Client ID")->default_val(0);
	app.add_option("--model-name", model_name, "Model Name")->default_val("resnet50");
	app.add_option("-c,--total-clients", c_args.total_clients, "Total Clients")->default_val(1);

	CLI11_PARSE(app, argc, argv);

	if (read_model_configurations(&c_args, model_name, classes_file, input_shape, output_shape))
		return 1;

	if (debug && c_args.task != RemoteTask::PING)
	{
		std::cout << static_cast<int>(c_args.model_type) << std ::endl;
		for (size_t i = 0; i < input_shape.size(); i++)
		{
			std::cout << input_shape[i] << "\t";
		}
		std::cout << std::endl;
		for (size_t l = 0; l < output_shape.size(); l++)
		{
			for (size_t i = 0; i < output_shape[l].size(); i++)
			{
				std::cout << output_shape[l][i] << "\t";
			}
			std::cout << std::endl;
		}
	}
	client = rdma_init_client(host, std::to_string(static_cast<int>(c_args.port)));
	if (!client)
	{
		DEBUG_LOG("rdma_init_client() failed \n");
		final_result = 1;
		goto client_closure;
	}
	DEBUG_LOG("RDMA client is ready \n");

	ret = prepare_rdma_client(c_args, client);
	if (ret)
	{
		DEBUG_LOG("rdma_init_client() failed \n");
		final_result = 1;
		goto client_closure;
	}
	DEBUG_LOG("RDMA client %d is connected ... Sending Requests \n", c_args.client_id);
	ret = send_requests(c_args, client, c_args.requests, classes_file, input_shape, output_shape);
	if (ret)
	{
		final_result = 1;
	}
	final_result = 0;
	goto client_closure;

client_closure:
	ret = rdma_free_client(client);
	if (ret)
	{
		fprintf(stderr, "failure in free client, error %d\n", ret);
		final_result = 1;
	}
	DEBUG_LOG("RDMA client %d is existing \n", c_args.client_id);
	return final_result;
}
