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
            thrust::device_vector<int> iDevice(idata, idata + n);
            thrust::device_vector<int> oDevice(n);
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            cudaDeviceSynchronize();


            timer().startGpuTimer();

            thrust::exclusive_scan(iDevice.begin(), iDevice.end(), oDevice.begin());
            cudaDeviceSynchronize();

            timer().endGpuTimer();
            
            thrust::copy_n(oDevice.begin(), n, odata);
        }
    }
}
