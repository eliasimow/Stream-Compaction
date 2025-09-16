# CUDA Stream Compaction
*  Eli Asimow
* [LinkedIn](https://www.linkedin.com/in/eli-asimow/), [personal website](https://easimow.com)
* Tested on: Windows 11, AMD Ryzen 7 7435HS @ 2.08GHz 16GB, Nvidia GeForce RTX 4060 GPU

## **Overview**
This repo includes a collection of standard sorting and compaction algorithms, implemented in four different contexts. Those four are a CPU implementation, naive and efficient GPU implementations, and lastly a thrust library wrapper. These methods have identical results, but their means of calculation and time complexity vary a fair bit. 

**CPU Scan**

Algorithm Time Complexity: O(N), O(N) calculations

Adds sum collectively as the algorithm marches through the set.


**GPU Naive Scan**

Algorithm Time Complexity: O(log2(N)), O(NLog2(N)) calculations

Construct a binary tree of sums across log2n kernel calls. During each kernel call, all sum calculations are determined in parallel. 

**GPU Work Efficient Scan**

Time Complexity: O(log2​(N)), O(N) Calculations
A Down Sweep / Up Sweep enhancement of naive scan. 
Observe that this is the best of both worlds! This approach has the efficiency of CPU Scan, and the parallel time complexity of the naive GPU implementation. We can further optimize this implementation by considering the stride of our indexing. For each kernel run, we only care about operating on indices that are a multiple of that power of 2. It’s this consideration that led to my work efficient scan outperforming its peers. 

**CPU Compact Without Scan**

Time Complexity: O(N), O(N) Calculations

Tracks filtered count while iterating through the set.


**CPU Compact with Scan**

Time Complexity: O(N), O(N) Calculations + one additional scan pass

While CPU scan has a greater memory overhead than our other CPU option, it’s still equally efficient in terms of speed.


**GPU Work Efficient Compact**

Time Complexity: O(log2​(N)), O(N) calculations

By using the GPU scan kernel already implemented above, we can reduce the time complexity of this filter past linear, making this our speediest option for filtering.


## **Performance**

<img width="1006" height="622" alt="Scan Methods Computation Time for Input Array Length" src="https://github.com/user-attachments/assets/dae3081e-3cce-42b7-bae9-6124e7a63e71" />

Although at lower array lengths the memory overhead of the GPU implementations inhibits their computation time, by the measured length 4,194,304 they were superior to their CPU counterpart. That’s excluding the naive implementation, which featured staggeringly slow times compared to the other GPU implementations, clocking in at ten times slower in the final test size of 2^29 elements.


An interesting point to observe here is the superiority of the thrust method. It keeps a consistently faster performance time than my efficient scan implementation across all the higher input array tests.
	To dive into this a bit deeper and determine what the bottleneck was for my efficient implementation, I booted up NSIGHT Compute. There, I compared the bottlenecks for my work-efficient scan and the Cuda library thrust scan. Let’s look at them side by side.
	We can see a massive difference in memory usage. Work Efficient uses a staggeringly large amount of L2 Global memory, whereas thrust works with much more L1 shared memory. The most probable explanation for this is that thrust is tiling its operation, working through it in smaller chunks with memory adjacency. We should also consider the cost of the kernels invoked in work-efficient scan. Work-efficient runs Log2(n) kernels, whereas thrust only runs 3. When you contrast thrust’s 17,895,808 threads with work-efficients 134,217,856, the reason for the speed disparity seems fairly self-evident. 
	Lastly, I found some interesting results while studying block size. It seems that there’s some alternating effect on work-efficient scan as block size toggles between even and odd powers of two. This result was consistent across multiple reruns. My best estimate is that this is related to my Nvidia laptop’s 32 thread warp size!
