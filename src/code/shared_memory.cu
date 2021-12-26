#include "../constants.hpp"
#include "../color.hpp"

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <random>


// Can't do std climits 
#define LIMIT 1000000

__constant__ int particleLocations[NUM_PARTICLES][2];

__global__ void cu_varonoi (int *array_d) {
	
	int element = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int id = threadIdx.x;
	
	__shared__ int lattice[BLOCK_SIZE];
	__shared__ float distance[BLOCK_SIZE];
	
	if (element < (ARRAY_X * ARRAY_Y)) {
		
		/** Set the distance to the max prior to loop. */
		distance[id] = LIMIT;
		__syncthreads();

		for (int i = 0; i < NUM_PARTICLES; ++i) {
			
			/** 
			*	Calculate the euclidean distance. We avoid using powf
			*	here by explicitly multiplying. CUDA best practices
			*	states that powf is an expensive function due to the
			*	necessary checks during computation.
			*	*/
			
			float temp = sqrtf((float) 
				((float) (particleLocations[i][0] - (element % ARRAY_X)) * 
					(particleLocations[i][0] - (element % ARRAY_X))) +
				((float) (particleLocations[i][1] - (element / ARRAY_X)) * 
					(particleLocations[i][1] - (element / ARRAY_X))));
			__syncthreads();
			
			if (temp < distance[id]) {
				distance[id] = temp;
				lattice[id] = i;
			}
			__syncthreads();
		
		}
		
		// Copy to device memory.
		array_d[element] = lattice[id];
	}
}

int main(int argc, char ** argv) {
		
	int * array = new int[ARRAY_X * ARRAY_Y];
	int *array_d;
	
	cudaError_t result;
	
	int hostParticles[NUM_PARTICLES][2] = { 
			// Removed for the sake of brevity
											};
	
	cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventRecord(start, 0);
	
	result = cudaMalloc ((void**) &array_d, sizeof(int) * (ARRAY_X * ARRAY_Y));
	result = cudaMemcpyToSymbol (particleLocations, hostParticles, sizeof(int) * 
	                            (2 * NUM_PARTICLES));
	
	if (result != cudaSuccess) {
		std::cerr << "cudaMalloc (thread) failed." << std::endl;
		delete array;
		exit(1);
	}
		
	dim3 dimblock (BLOCK_SIZE);
	dim3 dimgrid (ceil ((float) (ARRAY_X * ARRAY_Y)/BLOCK_SIZE));
	
	cu_varonoi <<<dimgrid, dimblock>>> (array_d);
	
	result = cudaMemcpy (array, array_d, sizeof(int) * (ARRAY_X * ARRAY_Y), 
	                                               cudaMemcpyDeviceToHost);
	float elapsedTime;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime (&elapsedTime, start, stop);
	if (result != cudaSuccess) {
		std::cerr << "cudaMemcpy host <- dev (thread) failed." << std::endl;
		delete array;
		exit(1);
	}
	
	
	result = cudaFree (array_d);
	
	
	// std::vector<int> out;
	
	// out.insert(out.begin(), std::begin(array), std::end(array));
	
	// printColor (out);
	delete array;
	std::cout << elapsedTime << "\t" << std::endl;

	
	
	return 0;
}