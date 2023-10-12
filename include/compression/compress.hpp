#pragma once

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "utils.hpp"
#include "macros.hpp"
#include "logger.hpp"

using namespace std;

class Compress{
public:
	string input_file_name, output_file_name;
	uint input_file_size_bytes=0, output_file_size_bytes=0;

	Compress(string ip_file, string op_file){
		input_file_name = ip_file;
		output_file_name = op_file;
	}

	void compressController();

	int storeSymbols(uint *num_of_symbols, struct Symbol *symbols);

	void boundaryPackageMerge(struct Symbol *symbols, uint symbol_count, struct Codetable *codetable);

};