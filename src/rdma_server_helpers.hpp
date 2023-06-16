/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * RDMA Server Helper:
 * Implements RDMA Server Helpers
 *
 **/
#include "cuda.h"
extern int debug;

#define DEBUG_LOG \
    if (debug)    \
    printf

#define ASSERT(x)                                                                          \
    do                                                                                     \
    {                                                                                      \
        if (!(x))                                                                          \
        {                                                                                  \
            fprintf(stdout, "Assertion \"%s\" failed at %s:%d\n", #x, __FILE__, __LINE__); \
        }                                                                                  \
    } while (0)

#define CUCHECK(stmt)                   \
    do                                  \
    {                                   \
        CUresult result = (stmt);       \
        ASSERT(CUDA_SUCCESS == result); \
    } while (0)

/// @brief Debug print information about all available CUDA devices
void print_gpu_devices_info()
{
    int device_count = 0;
    int i;

    CUCHECK(cuDeviceGetCount(&device_count));
    DEBUG_LOG("The number of supporting CUDA devices is %d.\n", device_count);

    for (i = 0; i < device_count; i++)
    {
        CUdevice cu_dev;
        char name[128];
        int pci_bus_id = 0;
        int pci_device_id = 0;
        int pci_func = 0; /*always 0 for CUDA device*/

        CUCHECK(cuDeviceGet(&cu_dev, i));
        CUCHECK(cuDeviceGetName(name, sizeof(name), cu_dev));
        CUCHECK(cuDeviceGetAttribute(&pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, cu_dev));       /*PCI bus identifier of the device*/
        CUCHECK(cuDeviceGetAttribute(&pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, cu_dev)); /*PCI device (also known as slot) identifier of the device*/

        DEBUG_LOG("device %d, handle %d, name \"%s\", BDF %02x:%02x.%d\n",
                  i, cu_dev, name, pci_bus_id, pci_device_id, pci_func);
    }
}

static CUcontext cuContext;

/// @brief Connect to Local GPU
/// @return
int connect_gpu()
{
    CUresult cu_result;
    cu_result = cuInit(0);
    if (cu_result != CUDA_SUCCESS)
    {
        fprintf(stderr, "cuInit(0) returned %d\n", cu_result);
        return 1;
    }
    if (debug)
    {
        print_gpu_devices_info();
    }
    /* Pick up device by given dev_id  Always use ID 0*/
    CUdevice cu_dev;
    CUCHECK(cuDeviceGet(&cu_dev, 0));
    DEBUG_LOG("creating CUDA Contnext\n");
    /* Create context */
    cu_result = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, cu_dev);
    if (cu_result != CUDA_SUCCESS)
    {
        fprintf(stderr, "cuCtxCreate() error=%d\n", cu_result);
        return 1;
    }

    DEBUG_LOG("making it the current CUDA Context\n");
    cu_result = cuCtxSetCurrent(cuContext);
    if (cu_result != CUDA_SUCCESS)
    {
        fprintf(stderr, "cuCtxSetCurrent() error=%d\n", cu_result);
        return 1;
    }
    int leastPriority, greatestPriority;
    auto error = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    if (error != cudaSuccess)
    {
        return 1;
    }
    DEBUG_LOG("Priority Range from %d to %d \n", leastPriority, greatestPriority);
    return 0;
}

/// @brief Allocate Client Needed Memory
/// @param addressing 
/// @param request_buf 
/// @param request_size 
/// @param reply_buf 
/// @param reply_size 
/// @return 
int allocate_client_connection_memory(MemoryAddressing addressing, void *&request_buf, size_t request_size, void *&reply_buf, size_t reply_size)
{
    cudaSetDevice(0);
    if (addressing != MemoryAddressing::HOST_ADDRESSING)
    {
        CUresult cu_result;
        const size_t gpu_page_size = 64 * 1024;
        size_t request_aligned_size;
        size_t reply_aligned_size;
        DEBUG_LOG("Allocating GPU Memory \n");

        request_aligned_size = (request_size + gpu_page_size - 1) & ~(gpu_page_size - 1);
        reply_aligned_size = (reply_size + gpu_page_size - 1) & ~(gpu_page_size - 1);

        DEBUG_LOG("cuMemAlloc() of a %zd bytes Request GPU buffer\n", request_aligned_size);
        CUdeviceptr request_A;
        cu_result = cuMemAlloc(&request_A, request_aligned_size);
        if (cu_result != CUDA_SUCCESS)
        {
            fprintf(stderr, "cuMemAlloc error=%d\n", cu_result);
            return 1;
        }
        DEBUG_LOG("allocated GPU Request buffer address at %016llx pointer=%p\n", request_A, (void *)request_A);
        request_buf = ((void *)request_A);

        DEBUG_LOG("cuMemAlloc() of a %zd bytes Reply GPU buffer\n", reply_aligned_size);
        CUdeviceptr reply_A;
        cu_result = cuMemAlloc(&reply_A, reply_aligned_size);
        if (cu_result != CUDA_SUCCESS)
        {
            fprintf(stderr, "cuMemAlloc error=%d\n", cu_result);
            return 1;
        }
        DEBUG_LOG("allocated GPU Reply buffer address at %016llx pointer=%p\n", reply_A, (void *)reply_A);
        reply_buf = (void *)reply_A;
        return 0;
    }
    // 5.1 Allocate Request Buffer
    request_buf = calloc(1, request_size);
    if (!request_buf)
    {
        DEBUG_LOG("request buffer allocation failure");
        return 1;
    }

    // 5.2 Allocate Reply Buffer
    reply_buf = calloc(1, reply_size);
    if (!reply_buf)
    {
        DEBUG_LOG("reply buffer allocation failure");
        return 1;
    }

    return 0;
}

/// @brief Release Client Resources
/// @param s_args 
/// @param client_session 
/// @return 
int rdma_free_client_connection(struct server_args s_args, struct rdma_client *client_session)
{
    DEBUG_LOG("Closing RDMA Client %d Session\n", client_session->pdata.client_id);
    int ret;
    if (client_session)
    {
        // Deregister and free Memory Region

        if (client_session->request_mr)
        {
            ret = ibv_dereg_mr(client_session->request_mr);
            if (ret)
            {
                fprintf(stderr, "failure in ibv_dereg_mr(), error %d\n", ret);
            }
            if (client_session->request_buf)
            {
                if (s_args.addressing != MemoryAddressing::HOST_ADDRESSING)
                {
                    DEBUG_LOG("Freeing request_buf \n");
                    cuMemFree((CUdeviceptr)client_session->request_buf);
                }
                else
                {
                    free(client_session->request_buf);
                }
            }
        }

        if (client_session->reply_mr)
        {
            ret = ibv_dereg_mr(client_session->reply_mr);
            if (ret)
            {
                fprintf(stderr, "failure in ibv_dereg_mr(), error %d\n", ret);
            }
            if (client_session->reply_buf)
            {
                if (s_args.addressing != MemoryAddressing::HOST_ADDRESSING)
                {
                    DEBUG_LOG("Freeing reply_buf \n");
                    cuMemFree((CUdeviceptr)client_session->reply_buf);
                }
                else
                {
                    free(client_session->reply_buf);
                }
            }
        }
        if (client_session->cm_id)
        {
            if (client_session->cm_id->qp)
            {
                ret = ibv_destroy_qp(client_session->cm_id->qp);
                if (ret)
                {
                    fprintf(stderr, "failure in ibv_destroy_qp(), error %d\n", ret);
                }
            }
        }
        // Destroy CQ
        if (client_session->cq)
        {
            ret = ibv_destroy_cq(client_session->cq);
            if (ret)
            {
                fprintf(stderr, "failure in ibv_destroy_cq(), error %d\n", ret);
            }
        }
        // Destroy comp_chan
        if (client_session->comp_chan)
        {
            ret = ibv_destroy_comp_channel(client_session->comp_chan);
            if (ret)
            {
                fprintf(stderr, "failure in ibv_destroy_comp_channel(), error %d\n", ret);
            }
        }

        // Destroy PD
        if (client_session->pd)
        {
            ret = ibv_dealloc_pd(client_session->pd);
            if (ret)
            {
                fprintf(stderr, "failure in ibv_dealloc_pd(), error %d\n", ret);
            }
        }

        if (client_session->cm_id)
        {
            DEBUG_LOG("rdma_destroy_id(%p)\n", client_session->cm_id);
            ret = rdma_destroy_id(client_session->cm_id);
            if (ret)
            {
                fprintf(stderr, "failure in rdma_destroy_id(), error %d\n", ret);
                return ret;
            }
        }
        DEBUG_LOG("free client_session resources \n");

        free(client_session);
    }
    return 0;
}