#pragma once

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include "utils.hpp"
#include "macros.hpp"
#include "logger.hpp"
#include "decompressKernels.cuh"

using namespace std;

class Decompress{
public:
	string input_file_name, output_file_name;
	uint input_file_size_bytes=0;

	Decompress(string ip_file, string op_file){
		input_file_name = ip_file;
		output_file_name = op_file;
	}

	void decompressController();
};