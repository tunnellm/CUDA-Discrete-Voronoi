# CUDA-Discrete-Voronoi
 CUDA Accelerated Discrete Voronoi Diagram

We created four parallel implementations, as well as a sequential version. The sequential version utilizes inline functions, as well as \#pragma directives to direct the compiler to unroll the loop and vectorize relevant portions of the code. The actual implementation is rather simple, using a C++ style iterator to loop through a vector initialized to the index values. We use modular arithmetic on the value of the index and calculate the Euclidean distance as expected.

The device memory implementation mallocs two arrays equal to the size of the Voronoi diagram in the GPU memory, and then initializes one array to a large default value for the distance comparison. The actual algorithm is incredibly similar to the sequential version, with the only major difference being the removal of the C++ style iterator in favor of CUDA-style coalesced memory accesses. The shared memory version is similar to the device memory version, with the only difference being the removal of one malloced array and the implementation of two shared memory segments equal to the block size.


Finally, we have two Thrust implementations with minor differences between each. These implementations rely on a function object (functor). The functor used is essentially another implementation of the exact algorithm used in the aforementioned versions. The main difference between the Thrust implementation and the others are the use of fancy iterators.