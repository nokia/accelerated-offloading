# © 2023 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause
#
# MultiClient:
# Starts Multiple Clients in parallel.


#!/usr/bin/env python3
import argparse
import asyncio
import math


async def run_client(application_name: str, host_address: str, host_port: int, clients: int, client_id: str, debug_mode: bool, requests: int, task: int, model_name: str, save_logs: bool, results_folder: str):
    """Start Client

    Args:
        application_name (str): Client Application Name
        host_address (str): Host Address 
        host_port (int): Host Port
        clients (int): Total Clients
        client_id (str): Client ID
        debug_mode (bool): Run with Debug
        requests (int): Total Requests
        task (int): Task Code
        model_name (str): Model Name
        save_logs (bool): Save Client Logs
        results_folder (str): Logs Folder
    """
    print("Running ", f"./{application_name}", "--host", host_address, "--port", str(host_port),
          "--client-id", str(client_id), "-r", str(requests), "-t", str(task), debug_mode, "-c", str(clients), "--model-name", model_name, save_logs, "-f", results_folder)

    cmd = [f"./{application_name}", "--host", host_address, "--port", str(host_port), "--client-id", str(client_id), "-r", str(requests), "-t", str(
        task), "--model-name", model_name]

    if debug_mode:
        cmd.append("-d")
    if save_logs:
        cmd.append("-s")
        cmd.extend(["-f", results_folder])
    if application_name == "rdma_client":
        cmd.extend(["-c", str(clients)])

    proc = await asyncio.create_subprocess_exec(
        *cmd, shell=False,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    stdout, stderr = await proc.communicate()

    print(f'[Client {client_id} exited with {proc.returncode}]')

    if stdout:
        print(f'[stdout]\n{stdout.decode()}')
    if stderr:
        print(f'[stderr]\n{stderr.decode()}')


async def run_all(clients: int, application_name: str, host_address: str, host_port: int, debug_mode: bool, requests: int, task: int,
                  model_name: str, save_logs: bool, results_folder: str):
    """Start All Clients

    Args:
        clients (int): _description_
        application_name (str): _description_
        host_address (str): _description_
        host_port (int): _description_
        debug_mode (bool): _description_
        requests (int): _description_
        task (int): _description_
        model_name (str): _description_
        save_logs (bool): _description_
        results_folder (str): _description_
    """
    running_clients = [run_client(
        application_name, host_address, host_port, clients, cid, debug_mode, requests, task, model_name, save_logs, results_folder) for cid in range(clients)]
    await asyncio.gather(*running_clients)


def main():
    parser = argparse.ArgumentParser(description="Running Multiple Clients")
    parser.add_argument("--host", default="7.7.7.1", type=str, dest="host_address",
                        help="Host Address")
    parser.add_argument("-p", "--port", default=20079, type=int, dest="host_port",
                        help="Host port")
    parser.add_argument("-c", "--clients", type=int, dest="clients",
                        default="1", help="Number of clients")
    parser.add_argument("-a", "--application-name", type=str, dest="application_name",
                        default="rdma_client", help="Client Application")
    parser.add_argument("-t", "--task", type=int, dest="task", required=True,
                        help="Task")
    parser.add_argument("-r", "--requests", type=int, dest="requests",
                        default="1", help="Requests")
    parser.add_argument("-s", "--save-logs", default="",
                        action="store_const", const="-s", dest="save_logs", help="Save Logs")
    parser.add_argument("-f", "--folder", type=str, dest="results_folder",
                        default="results", help="Results Folder")
    parser.add_argument("-d", "--debug-mode", default="",
                        action="store_const", const="-d", dest="debug_mode", help="Debug Mode")
    parser.add_argument("-m", default="single", dest="mode",
                        help="execution mode", choices=["single", "range"])
    parser.add_argument("--model-name", default="resnet50", type=str,
                        dest="model_name", help="Model Name")

    args = parser.parse_args()
    results_folder = args.results_folder + "_" + str(args.clients)
    asyncio.run(run_all(args.clients, args.application_name, args.host_address, args.host_port, args.debug_mode, args.requests,
                        args.task, args.model_name, args.save_logs, results_folder))


if __name__ == '__main__':
    main()
