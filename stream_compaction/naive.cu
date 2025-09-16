#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;

        int blockSize = 128;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void scan(int n, int* odata, const int* idata, int levelPowered) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            int offset = 1 << (levelPowered - 1);
            if (index >= offset) {
                odata[index] = idata[index] + idata[index - offset];
            }
            else {
                odata[index] = idata[index];
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            int requiredLevels = ilog2ceil(n);

            int* pong1;
            int* pong2;

            cudaMalloc((void**)&pong1, (1 << requiredLevels) * sizeof(int));
            cudaMalloc((void**)&pong2, (1 << requiredLevels) * sizeof(int));
            cudaMemcpy(pong1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            int levelPowered = 1;

            timer().startGpuTimer();

            while (levelPowered <= requiredLevels) {
                scan << < fullBlocksPerGrid, blockSize >> > ((1 << requiredLevels), pong2, pong1, levelPowered);
                levelPowered++;
                std::swap(pong1, pong2);
            }

            timer().endGpuTimer();

            // TODO
            cudaMemcpy(odata, pong1, n * sizeof(int), cudaMemcpyDeviceToHost);

            for (int i = n-1; i >= 0; --i) {
                odata[i] = i == 0 ? 0 : odata[i - 1];
            }

            cudaFree(pong1);
            cudaFree(pong2);
        }
    }
}
