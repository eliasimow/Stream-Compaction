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

The additional scan pass makes this a strictly worse performing implementation than its CPU sibling. 

**GPU Work Efficient Compact**

Time Complexity: O(log2​(N)), O(N) calculations

By using the GPU scan kernel already implemented above, we can reduce the time complexity of this filter past linear, making this our speediest option for filtering.


**Example Output**

```****************
** SCAN TESTS **
****************
    [   7   8  19  34  32  18  37  47  46  35  48  21  13 ...  49   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 3.3725ms    (std::chrono Measured)
    [   0   7  15  34  68 100 118 155 202 248 283 331 352 ... 102745754 102745803 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 3.5867ms    (std::chrono Measured)
    [   0   7  15  34  68 100 118 155 202 248 283 331 352 ... 102745698 102745733 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 2.83651ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 1.7663ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.798592ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.658656ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.43155ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.25734ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   2   3   2   2   2   1   3   2   1   0   3   3 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 9.763ms    (std::chrono Measured)
    [   3   2   3   2   2   2   1   3   2   1   3   3   1 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 8.778ms    (std::chrono Measured)
    [   3   2   3   2   2   2   1   3   2   1   3   3   1 ...   3   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 26.286ms    (std::chrono Measured)
    [   3   2   3   2   2   2   1   3   2   1   3   3   1 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 5.69862ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 6.09875ms    (CUDA Measured)
    passed
```



## **Performance**

<img width="1006" height="622" alt="Scan Methods Computation Time for Input Array Length" src="https://github.com/user-attachments/assets/dae3081e-3cce-42b7-bae9-6124e7a63e71" />

For data collecting purposes, I averaged three results and used 128 block size for all cases other than naive, in which I used 32. I documented results up to 2^28; 2^30 is where computation breaks down, at least on my laptop. The results are largely unsurprising. Although at lower array lengths the memory overhead of the GPU implementations inhibits their computation time, by the measured length 4,194,304 they were superior to their CPU counterpart. That’s excluding the naive implementation, which featured staggeringly slow times compared to the other GPU implementations, clocking in at ten times slower in the final test size of 2^28 elements.

<img width="1418" height="151" alt="ThrustOverview" src="https://github.com/user-attachments/assets/d07550e5-ec49-42b1-8b6f-662e2e8395d2" />

An interesting point to observe here is the superiority of the thrust method. It keeps a consistently faster performance time than my efficient scan implementation across all the higher input array tests.
To dive into this a bit deeper and determine what the bottleneck was for my efficient implementation, I booted up NSIGHT Compute. There, I compared the bottlenecks for my work-efficient scan and the Cuda library thrust scan. Let’s look at them side by side.

<img width="1762" height="703" alt="ScanOverview" src="https://github.com/user-attachments/assets/357fca4b-5d41-4018-b510-9b5b51e61e37" />


The first most obvious takeaway is in the count of kernels invoked. Work-efficient runs Log2(n) kernels, whereas thrust only runs 3. When you contrast thrust’s ~152,253,000 threads with work-efficients ~536,878,000, the reason for the speed disparity seems fairly self-evident. Thrust manages to do the same amount of work more efficiently than in my implementation. I don't think the issue is in memory access, as in our most read/write intensive kernels, e.g. the first upsweep and final downsweep, we see fairly high memory thresholds of 87.89% and 89.91&. Compare with thrust's 95% throughput, I'm unconvinced that this is the difference maker. Instead, I'd suggest the issue is compute throughput, where my implementation never acheives thrust's efficiency of 35%. In summary, Thrust makes better usage of both its I/O and compute capabilities in its condensed 3 kernels, and that allows it to acheive its faster time. 

<img width="500" height="371" alt="Work Efficient Scan Time vs  Block Size With 2^29 elements" src="https://github.com/user-attachments/assets/251701df-4030-4a85-8cc9-a1f5cd870a98" />
<img width="500" height="371" alt="GPU Compact vs  Block Size With 2^29 elements" src="https://github.com/user-attachments/assets/53af342e-db41-48f8-b3f5-fb831841ea86" />

Lastly, I found some interesting results while studying block size. It seems that there’s some alternating effect on work-efficient scan as block size toggles between even and odd powers of two. This result was consistent across multiple reruns. My best estimate is that this is related to my Nvidia laptop’s 32 thread warp size. It seems isolated to my work efficient scan implementation, and did not emerge in my compact studies. Having seen the data for these, and the unaffected naive implementation, I decided a steady block size of 128 would be sufficient for testing. 
