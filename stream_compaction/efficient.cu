#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        //int blockSize = 256;

        __global__ void upScan(int n, int* data, int levelPowered) {
            if ((blockIdx.x * blockDim.x) + threadIdx.x > n) {
                return;
            }
            int stride = (1 << levelPowered);
            int groups = n / (1 << levelPowered);
            int index = ((blockIdx.x * blockDim.x) + threadIdx.x) * (1 << levelPowered) -1;
            if (index >  n || index < 0) {
                return;
            }
            data[index] = data[index] + data[index - (1 << (levelPowered-1))];
        }

        __global__ void downSweep(int n, int* data, int levelPowered) {
            if ((blockIdx.x * blockDim.x) + threadIdx.x > n) {
                return;
            }
            int index = ((blockIdx.x * blockDim.x) + threadIdx.x) * (1 << levelPowered) - 1;
            if (index > n || index < 0) {
                return;
            }
            int store = data[index];
            data[index] = data[index - (1 << (levelPowered - 1))] + data[index];
            data[index - (1 << (levelPowered - 1))] = store;          
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool time, int blockSize) {
            // TODO
            int requiredLevels = ilog2ceil(n);
            int fullSize = (1 << requiredLevels);
            int dynamicBlockSize = 2;

            int* data;
            cudaMalloc((void**)&data, fullSize * sizeof(int));
            cudaMemcpy(data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            int levelPowered = 1;
            dim3 fullBlocksPerGrid(fullSize / (1 << levelPowered) / blockSize);

            if (time)
                timer().startGpuTimer();

            int runcount = 0;
            while (levelPowered <= requiredLevels) {
                fullBlocksPerGrid = dim3((fullSize / (1 << levelPowered) + blockSize) / blockSize);
                upScan << < fullBlocksPerGrid, blockSize >> > (fullSize-1, data, levelPowered);
                runcount++;
                levelPowered++;
            }

            if (time) {
                timer().endGpuTimer();
            }

            
            levelPowered--;
            fullBlocksPerGrid = dim3(1, 1, 1);
            cudaMemset(data + fullSize - 1, 0, sizeof(int));



            while (levelPowered > 0) {
                fullBlocksPerGrid = dim3((fullSize / (1 << levelPowered) + blockSize) / blockSize);
                downSweep << < fullBlocksPerGrid, blockSize >> > (fullSize-1, data, levelPowered);
                levelPowered--;
            }
         
            cudaMemcpy(odata, data, n * sizeof(int), cudaMemcpyDeviceToHost);     

            cudaFree(data);
        }



        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata, int blockSize) {
            // TODO           
            int* boolsGpu;
            int* odataGpu;
            int* scanGpu;
            int* dataGpu;

            int requiredLevels = ilog2ceil(n);
            int fullSize = (1 << requiredLevels);
            cudaMalloc((void**)&boolsGpu, sizeof(int) * fullSize);
            cudaMalloc((void**)&scanGpu, sizeof(int) * fullSize);
            cudaMalloc((void**)&odataGpu, sizeof(int) * n);
            cudaMalloc((void**)&dataGpu, sizeof(int) * n);

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            Common::kernResetIntBuffer << < fullBlocksPerGrid, blockSize >> > (n, dataGpu, 0);
            Common::kernResetIntBuffer << < fullBlocksPerGrid, blockSize >> > (n, odataGpu, 0);

            cudaMemcpy(dataGpu, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            Common::kernMapToBoolean << < fullBlocksPerGrid, blockSize >> > (fullSize, boolsGpu, dataGpu);

            ///SCAN:

                int levelPowered = 1;
                cudaMemcpy(scanGpu, boolsGpu, sizeof(int) * fullSize, cudaMemcpyDeviceToDevice);

                while (levelPowered <= requiredLevels) {
                    fullBlocksPerGrid = dim3((fullSize / (1 << levelPowered) + blockSize) / blockSize);
                    upScan << < fullBlocksPerGrid, blockSize >> > (fullSize - 1, scanGpu, levelPowered);
                    levelPowered++;
                }

                levelPowered--;
                fullBlocksPerGrid = dim3(1, 1, 1);
                cudaMemset(scanGpu + fullSize - 1, 0, sizeof(int));

                while (levelPowered > 0) {
                    fullBlocksPerGrid = dim3((fullSize / (1 << levelPowered) + blockSize) / blockSize);
                    downSweep << < fullBlocksPerGrid, blockSize >> > (fullSize - 1, scanGpu, levelPowered);
                    levelPowered--;
                }

            cudaMemcpy(odata, scanGpu, n * sizeof(int), cudaMemcpyDeviceToHost);

            //0 check for last element: this causes headaches if you just return store!
            int store = idata[n - 1] == 0 ? odata[n - 1] : odata[n - 1] + 1;
            fullBlocksPerGrid = dim3((n + blockSize - 1) / blockSize);

            Common::kernScatter << < fullBlocksPerGrid, blockSize >> > (n, odataGpu, dataGpu, boolsGpu, scanGpu);
            timer().endGpuTimer();

            cudaMemcpy(odata, odataGpu, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(boolsGpu);
            cudaFree(scanGpu);
            cudaFree(odataGpu);
            cudaFree(dataGpu);

            return store;
        }
    }
}
