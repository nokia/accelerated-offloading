/**
 * © 2023 Nokia
 * Licensed under the BSD 3-Clause License
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Monitor:
 * Implements resource monitoring features.
 *
 **/

#include "utils.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "sys/times.h"
#include "sys/vtimes.h"
#include <thread>
#include <vector>
#include <chrono>

extern int debug;

struct monitor_state
{
    double cpu_time;
    double host_memory;
    double gpu_memory;
};

/// @brief Monitoring Class
class Monitor
{
private:
    int interval;
    bool running = false;
    std::thread monitor_thread;
    std::vector<monitor_state> monitor_states;
    clock_t lastCPU, lastSysCPU, lastUserCPU;
    unsigned long long lastTotalUser, lastTotalUserLow, lastTotalSys, lastTotalIdle;

    /// @brief Read Value of memory
    /// @param line 
    /// @return 
    int parse_line(char *line)
    {
        // This assumes that a digit will be found and the line ends in " Kb".
        int i = strlen(line);
        const char *p = line;
        while (*p < '0' || *p > '9')
            p++;
        line[i - 3] = '\0';
        i = atoi(p);
        return i;
    }

    /// @brief Get Current Memory Consumption
    /// @return 
    int get_current_memory_consumption()
    { // Note: this value is in KB!
        FILE *file = fopen("/proc/self/status", "r");
        int result = -1;
        char line[128];

        while (fgets(line, 128, file) != NULL)
        {
            if (strncmp(line, "VmSize:", 7) == 0)
            {
                result = parse_line(line);
                break;
            }
        }
        fclose(file);
        return result / 1024;
    }

    /// @brief Initialize
    void init()
    {
        FILE *file = fopen("/proc/stat", "r");
        fscanf(file, "cpu %llu %llu %llu %llu", &lastTotalUser, &lastTotalUserLow,
               &lastTotalSys, &lastTotalIdle);
        fclose(file);
    }

    /// @brief Get CPU Time for the whole Machine
    /// @return 
    double get_cpu_time_total()
    {
        FILE *file;
        unsigned long long totalUser, totalUserLow, totalSys, totalIdle, total;

        file = fopen("/proc/stat", "r");
        fscanf(file, "cpu %llu %llu %llu %llu", &totalUser, &totalUserLow,
               &totalSys, &totalIdle);
        fclose(file);

        if (totalUser < lastTotalUser || totalUserLow < lastTotalUserLow ||
            totalSys < lastTotalSys || totalIdle < lastTotalIdle)
        {
            // Overflow detection. Just skip this value.
            return 0;
        }
        else
        {
            total = (totalUser - lastTotalUser) + (totalUserLow - lastTotalUserLow) +
                    (totalSys - lastTotalSys);
        }

        lastTotalUser = totalUser;
        lastTotalUserLow = totalUserLow;
        lastTotalSys = totalSys;
        lastTotalIdle = totalIdle;

        return total;
    }

    /// @brief Get GPU Memory
    /// @return 
    double get_gpu_memory()
    {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        auto used = total - free;
        return used / (1024.0 * 1024.0);
    }

    /// @brief Start Monitor
    void launch_monitor()
    {
        while (running)
        {
            monitor_state state = {};
            state.cpu_time = get_cpu_time_total();
            state.host_memory = get_current_memory_consumption();
            state.gpu_memory = get_gpu_memory();
            monitor_states.emplace_back(state);
            std::this_thread::sleep_for(std::chrono::milliseconds(interval));
        }
    }

public:
    Monitor(int _interval) // interval in ms
    {
        interval = _interval;
        init();
        running = true;
        monitor_thread = std::thread(&Monitor::launch_monitor, this);
    }

    /// @brief Save Monitoring Logs to disk
    /// @param results_folder 
    /// @param file_name 
    void save_monitoring_state(std::string results_folder, std::string file_name)
    {
        running = false;
        results_folder = results_folder + "/";

        mkdir(results_folder.c_str(), 0700);
        std::ofstream myFile(results_folder + file_name + "_resources.csv");
        // Send the column name to the stream
        myFile << "cpu_time"
               << ","
               << "host_memory"
               << ","
               << "gpu_memory"
               << "\n";

        // Send data to the stream
        for (size_t i = 0; i < monitor_states.size(); ++i)
        {
            myFile << monitor_states[i].cpu_time << ",";
            myFile << monitor_states[i].host_memory << ",";
            myFile << monitor_states[i].gpu_memory << "\n";
        }

        // Close the file
        myFile.close();
    }

    ~Monitor()
    {
        running = false;
        if (monitor_thread.joinable())
        {
            monitor_thread.join();
        }
    }
};
