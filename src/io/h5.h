#pragma once

#include <mpi.h>
#include <highfive/highfive.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>

#include "io/output/print.h"

namespace H5 {
    inline void check_collective_io(const HighFive::DataTransferProps& xfer_props) {
        auto mnccp = HighFive::MpioNoCollectiveCause(xfer_props);
        if (mnccp.getLocalCause() || mnccp.getGlobalCause()) {
            println("The operation was successful, but couldn't use collective MPI-IO. local cause: {} global cause: {}",
                mnccp.getLocalCause(), mnccp.getGlobalCause());
        }
    }
}