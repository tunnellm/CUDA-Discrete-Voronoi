#include "constants.hpp"
#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <iostream>


void printColor (std::vector<int> lattice) {

	std::array<std::string, NUM_COLORS> colors = {
		
		"0,0,0,",
		"255,0,0,",
		"0,255,0,",
		"0,0,255,",
		"255,255,0,",
		"0,255,255,",
		"255,0,255,",
		"192,192,192,",
		"128,0,0,",
		"0,128,0,",
		"128,0,128,",
		"0,128,128,",
		"0,0,128,",
		"240,248,255,",
		"138,43,226,",
		"95,159,159,",
		"95,158,160,",
		"95,158,160,",
		"152,245,255,",
		"142,229,238,",
		"122,197,205,",
		"83,134,139,",
		"0,191,255,",
		"0,178,238,",
		"0,154,205,",
		"0,104,139,",
		"30,144,255,",
		"30,144,255,",
		"28,134,238,",
		"24,116,205,",
		"16,78,139,",
		"173,216,230,",
		"191,239,255,",
		"178,223,238,",
		"154,192,205,",
		"104,131,139,",
		"224,255,255,",
		"224,255,255,",
		"209,238,238,",
		"180,205,205,",
		"122,139,139,",
		"135,206,250,",
		"176,226,255,",
		"164,211,238,",
		"141,182,205,",
		"96,123,139,",
		"132,112,255,",
		"176,196,222,",
		"202,225,255,",
		"188,210,238,",
		"162,181,205,",
		"110,123,139,",
		"176,224,230,",
		"65,105,225,",
		"72,118,255,",
		"67,110,238,",
		"58,95,205,",
		"39,64,139,",
		"0,34,102,",
		"135,206,255,",
		"126,192,238,",
		"108,166,205,",
		"74,112,139,",
		"106,90,205,",
		"131,111,255,",
		"122,103,238,",
		"105,89,205,",
		"71,60,139,",
		"70,130,180,",
		"99,184,255,",
		"92,172,238,",
		"79,148,205,",
		"54,100,139,"
	};
	
	std::stringstream streamString;

	for (auto & it : lattice)
		streamString << colors.at(it % NUM_COLORS);
	
	std::string out = streamString.str();
	out.resize(out.size() - 1);
	
	std::cout << out << std::endl;
}