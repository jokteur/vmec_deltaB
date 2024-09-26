#pragma once
#include <string>
#include <mpi.h>
#include <fstream>
#include "types.h"

// Logger singleton class
class Logger {
private:
    int rank;
    std::string prefix;
    Logger(const std::string& filename_prefix) : prefix(filename_prefix) {
#ifdef NEW_VENUS_MPI_LOGGING
#endif
    }
    ~Logger() {
#ifdef NEW_VENUS_MPI_LOGGING
#endif
    }
public:
    static Logger& instance(const std::string& filename_prefix = "output") {
        static Logger instance(filename_prefix);
        return instance;
    }

    void flush() {
#ifdef NEW_VENUS_MPI_LOGGING
        std::ofstream log_file;
        log_file.open(prefix + "_" + std::to_string(rank) + ".log");
        log_file.close();
#endif
    }

    void log(const std::string& message) {
#ifdef NEW_VENUS_MPI_LOGGING
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::ofstream log_file;
        log_file.open(prefix + "_" + std::to_string(rank) + ".log", std::ios_base::app);
        log_file << message;
        log_file.close();
#endif
    }
};