#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata, bool time) {
            if(time)
                timer().startCpuTimer();
            // TODO
            for (int i = 0; i < n; ++i) {
                odata[i] = i == 0 ? 0 : odata[i - 1] + idata[i - 1];
            }

            if(time)
                timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    count++;
                }
            }

            for (int i = count; i < n; ++i) {
                odata[i] = 0;
            }

            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* map = new int[n];
            int* indices = new int[n];

            for (int i = 0; i < n; ++i) {
                map[i] = idata[i] == 0 ? 0 : 1;
            }

            scan(n, indices, map, false);

            //scatter:
            for (int i = 0; i < n; ++i) {
                if (map[i]) odata[indices[i]] = idata[i];
            }
            //clear rest of array:
            for (int i = indices[n - 1]; i < n; ++i) {
                odata[i] = 0;
            }

            int storage = indices[n - 1];
            delete map;
            delete indices;

            timer().endCpuTimer();      
            return storage;
        }
    }
}
