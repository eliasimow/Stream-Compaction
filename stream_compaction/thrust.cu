#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            thrust::host_vector<int> iHost(idata, idata + n);
            thrust::host_vector<int> oHost(odata, odata + n);

            thrust::device_vector<int> iDevice = iHost;
            thrust::device_vector<int> oDevice = oHost;
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
             thrust::exclusive_scan(iDevice.begin(), iDevice.end(), oDevice.begin());

             thrust::copy(oDevice.begin(), oDevice.end(), oHost.begin());

             // Copy from host_vector back into your raw output pointer
             std::copy(oHost.begin(), oHost.end(), odata);
            timer().endGpuTimer();
        }
    }
}
