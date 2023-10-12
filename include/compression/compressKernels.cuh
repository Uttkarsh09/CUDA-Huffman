#pragma once
#ifndef CMPRESS_KERNELS
#define COMPRESS_KERNELS

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define HIST_THREADS 192
#define WARP_SIZE 32
#define WARP_COUNT (HIST_THREADS / WARP_SIZE)
#define HIST_SIZE 256
#define S_HIST_SIZE (WARP_COUNT * HIST_SIZE)
#define HIST_BLOCK 240
#define MERGE_THREADBLOCK_SIZE 256

__global__ void cu_histgram(
	uint *d_partial_histograms,
	uint *d_data,
	uint data_count,
	uint byte_count
);

__global__ void mergeHistogram(
	uint *d_historgam,
	uint *d_partial_histograms,
	uint histogram_count
);

void generateCharHistogram(uint *d_Histogram, uint*d_Data, uint byteCount);

void encode(
	uint *output_file, 
	uint output_file_size,
	uint *d_input_file,
	uint input_file_size,
	uint gap_array_elements_num,
	struct Codetable* codetable
);

#endif