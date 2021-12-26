#include "../constants.hpp"
#include "../color.hpp"

#include <iostream>
#include <chrono>
#include <tuple>
#include <vector>
#include <climits>
#include <cmath>
#include <array>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <fstream>
#include <cstring>
#include <sstream> 
#include <string>

// Define a constant here for use in unroll.
const int num = NUM_PARTICLES;

inline std::tuple<int, int> coordinateExpansion (const int & coords);

int main (int argc, char ** argv) {
	
	std::array<std::array<int, 2>, 100> particleLocations = 
	{{
		// removed for brevity
	}};
	
	
	auto startTime = std::chrono::steady_clock::now();
	
	// Initialize a vector to the size of the x*y grid
	std::vector<int> lattice(ARRAY_Y * ARRAY_X);
		
	// Set the elements equal to their position
	std::iota(lattice.begin(), lattice.end(), 0);
	
	int minimum = -1;
	float distance = INT_MAX;
	float temp;
	
	#pragma GCC ivdep
	for (auto & it : lattice) {
		
		std::tuple<int, int> comparison = coordinateExpansion(it);
		
		#pragma GCC unroll num
		for (int i = 0; i < NUM_PARTICLES; ++i) {
			
			// Compute the euclidean distance for each item
			temp = std::sqrt(std::pow(particleLocations[i][0] - std::get<0>(comparison), 2) + 
				         std::pow(particleLocations[i][1] - std::get<1>(comparison), 2));
			if ( temp < distance) {
				// std::cout << it << " new min: " << i << std::endl;
				distance = temp;
				minimum = i;
			}
		}
		
		// Update element with the minimum after inner loop

		it = minimum;
		
		minimum = -1;
		distance = INT_MAX;
	}
	auto endTime = std::chrono::steady_clock::now();
	
	// std::chrono::duration<double> elapsed_seconds = endTime-startTime;
	
	// std::cout << elapsed_seconds.count() << std::endl;
	
	printColor(lattice);
	
	return 1;
}

inline std::tuple<int, int> coordinateExpansion (const int & coords) {
	return std::make_tuple(coords % ARRAY_X, coords / ARRAY_X);
}