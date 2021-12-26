#include "../constants.hpp"
#include "../color.hpp"


#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>
#include <thrust/generate.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>

#include <thrust/copy.h>
#include <thrust/fill.h>

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <math.h>
#include <cstring>
#include <sstream> 
#include <string>

#define LIMIT 1000000

// We have to define a const int here for use in unroll.
const int num = NUM_PARTICLES;

__constant__ int particleLocations[NUM_PARTICLES][2];


/** 
* 	Struggling to declare this struct in another 
*	file or even below main in this one. CUDA does 
*	not like to play nice with other files it would
*	seem.
*/
struct smallest_distance {
	
	smallest_distance(){}//int& arrayX, int& arrayY, int& index) : _arrayX(arrayX), _arrayY(arrayY), _index(index) {}
	
	__device__
	void operator() (int& optimal, float& distance, const int& location, const int& _index) {
				
		/** 
		*	CUDA best practices guide states that for small problems, the use of
		*	simple multiplication is preferred to the use of powf due to extensive
		*	checks that are required. This ugly expansion is to save read/writes to
		*	memory unnecessarily. Compiler flags for sqrt estimation may be preferred
		*	as the exact distance is not important.
		*/
		
		float temp = sqrtf((float) 
						  ((float) (particleLocations[_index][0] - (location % ARRAY_X)) * 
								   (particleLocations[_index][0] - (location % ARRAY_X))) +
		 				  ((float) (particleLocations[_index][1] - (location / ARRAY_X)) * 
						           (particleLocations[_index][1] - (location / ARRAY_X))));
		__syncthreads();
		// float original = thrust::get<1>(x) * thrust::get<1>(x);
		if (temp < distance) {
			distance = temp;
			optimal = _index;
		}
		__syncthreads();
	}
};

int main(int argc, char ** argv) {
	cudaError_t result;
	int hostParticles[NUM_PARTICLES][2] = { {100, 100}, 
											{1180, 100}, 
											{640, 360}, 
											{100, 620}, 
											{1180, 620}
											};
	
	cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	
	cudaEventRecord(start, 0);
	
	
	
	
	// Pass the array to constant memory.
	result = cudaMemcpyToSymbol (particleLocations, hostParticles, sizeof(int) * (2 * NUM_PARTICLES));
	if (result != cudaSuccess) {
		std::cerr << "cudaMalloc (thread) failed." << std::endl;
		exit(1);
	}
	
	// Initialize a device vector
	thrust::device_vector<int> lattice(ARRAY_X * ARRAY_Y);
	thrust::device_vector<float> distance(ARRAY_X * ARRAY_Y);
	
	thrust::fill(distance.begin(), distance.end(), LIMIT);
	
	thrust::counting_iterator<int> elementStart(0);
	thrust::counting_iterator<int> elementEnd(lattice.size());
	
	for (int i = 0; i < NUM_PARTICLES; ++i)
		
		/** 
		*	Decoding this for_each loop:
		*	We create a tuple containing the two vectors we wish to edit,
		*	along with an index passed as a counting iterator.
		*	
		*	We then create a zip function on the functor we created so that
		*	we may pass to it all of these values. The values were previously
		*	passed in a different way, but I digress.
		*/
		
		thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(
										  lattice.begin(), 
										  distance.begin(), 
										  elementStart,
										  thrust::make_constant_iterator(i)
										  )), 
		thrust::make_zip_iterator(thrust::make_tuple(
										  lattice.end(), 
										  distance.end(), 
										  elementEnd,
										  thrust::make_constant_iterator(i)
										  )),
		thrust::make_zip_function(smallest_distance()));
	
	
	/** We transfer the items back to the host now that the for loop 
	*	is finished. We can take the time here.
	*/
	
	thrust::host_vector<int> outputLattice = lattice;
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	float elapsedTime;
	cudaEventElapsedTime (&elapsedTime, start, stop);
	
	std::cout << "\t" << elapsedTime << "\t" << std::endl;
	
	// Declare the size of vector or Thrust will cause segfault
	// std::vector<int> out (ARRAY_X * ARRAY_Y);
	// thrust::copy(outputLattice.begin(), outputLattice.end(), out.begin());
	// printColor (out);
	
	
	return 0;
}

