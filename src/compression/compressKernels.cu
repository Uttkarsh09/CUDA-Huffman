#include <iostream>
#include "compressKernels.cuh"
#include "macros.hpp"
#include "logger.hpp"

inline __device__ void addByte(uint *s_warp_hist, uint data){
	atomicAdd(s_warp_hist + data, 1);
}


inline __device__ void addWord(uint *s_warp_hist, uint data){
	addByte(s_warp_hist, (data >> 0) & 0xFFU);
	addByte(s_warp_hist, (data >> 8) & 0xFFU);
	addByte(s_warp_hist, (data >> 16) & 0xFFU);
	addByte(s_warp_hist, (data >> 24) & 0xFFU);
}


__global__ void cu_histgram( uint *d_partial_histograms, uint *d_data, uint data_count, uint byte_count ){
	__shared__ uint s_hist[S_HIST_SIZE];
	uint *s_warp_hist = s_hist + (threadIdx.x >> 5) * HIST_SIZE;
	uint warp_lane = threadIdx.x&31;

	for(uint i=warp_lane ; i<HIST_SIZE ; i+=WARP_SIZE){
		s_warp_hist[i] = 0;
	}

	__syncthreads();

	unsigned int pos=0;
	// ? Traversing the input file d_data contains input_file contents
	for( pos = (blockIdx.x*blockDim.x) + threadIdx.x; pos<data_count-1; pos += (blockDim.x*gridDim.x) ){
		unsigned int data = d_data[pos];
		addWord(s_warp_hist, data);
	}

	if(pos == data_count-1){
		unsigned int data = d_data[pos];
		// ?  masks the data variable to keep only the least significant 8 bits (1 byte) and sets all higher-order bits to zero
		switch(byte_count&3){
			case 1:
				addByte(s_warp_hist, (data >>  0) & 0xFFU );
				break;
			case 2:
				addByte(s_warp_hist, (data >>  0) & 0xFFU );
				addByte(s_warp_hist, (data >>  8) & 0xFFU );
				break;
			case 3:
				addByte(s_warp_hist, (data >>  0) & 0xFFU );
				addByte(s_warp_hist, (data >>  8) & 0xFFU );
				addByte(s_warp_hist, (data >> 16) & 0xFFU );
				break;
			default:
				addByte(s_warp_hist, (data >>  0) & 0xFFU );
				addByte(s_warp_hist, (data >>  8) & 0xFFU );
				addByte(s_warp_hist, (data >> 16) & 0xFFU );
				addByte(s_warp_hist, (data >> 24) & 0xFFU );
		}
	}

	__syncthreads();

	//
	for(unsigned int bin = threadIdx.x; bin<HIST_SIZE; bin += HIST_THREADS){
		unsigned int sum = 0;
		for(unsigned int i = 0; i < WARP_COUNT; i++){
			sum += s_hist[bin + i * HIST_SIZE];
		}
		d_partial_histograms[blockIdx.x * HIST_SIZE + bin] = sum;
	}
}


__global__ void mergeHistogram( uint *d_historgam, uint *d_partial_histograms,	uint histogram_count ){
	uint sum = 0;

	for(uint i=threadIdx.x ; i<histogram_count ; i+=MERGE_THREADBLOCK_SIZE){
		sum += d_partial_histograms[blockIdx.x + i * HIST_SIZE];
	}

	__shared__ uint data[MERGE_THREADBLOCK_SIZE];
	data[threadIdx.x] = sum;

	for(uint stride=MERGE_THREADBLOCK_SIZE/2 ; stride>0 ; stride>>=1){
		__syncthreads();
		if(threadIdx.x < stride){
			data[threadIdx.x] += data[threadIdx.x + stride];
		}
	}

	if(threadIdx.x == 0){
		d_historgam[blockIdx.x] = data[0];
	}
}


void generateCharHistogram(uint *d_histogram, uint *d_inputfile, uint byte_count){
	uint data_count = (byte_count + 3) / 4;
	uint *d_partial_histograms;
	Logger* l = Logger::GetInstance("./logs/logs.log");

	l->logIt(l->INFO, "Allocating memory for partial histograms sizeof(int)[%d]*HIST_BLOCK[%d] * HIST_SIZE[%d]", sizeof(int), HIST_BLOCK, HIST_SIZE);
	l->logIt(l->INFO, "Total Device Memory allocated - sizeof(int)*HIST_BLOCK * HIST_SIZE = %d", sizeof(int)*HIST_BLOCK * HIST_SIZE);
	cudaMalloc((void**)&d_partial_histograms, sizeof(int)*HIST_BLOCK * HIST_SIZE);
	CUERROR
	l->logIt(l->INFO, "DONE - Allocating memory");

	cu_histgram<<<HIST_BLOCK, HIST_THREADS>>> (d_partial_histograms, d_inputfile, data_count, byte_count);
	mergeHistogram<<<HIST_SIZE, MERGE_THREADBLOCK_SIZE>>>(d_histogram, d_partial_histograms, HIST_BLOCK);

	cudaFree(d_partial_histograms);
	return;
}


__global__ void cuEncoder(
	unsigned int *outputfile,
	unsigned int outputfilesize,
	unsigned int *inputfile,
	unsigned int inputfilesize,
	struct Codetable *codetable,
	volatile unsigned long long int *inclusive_sum,
	unsigned int *gap_array_bytes,
	unsigned int gap_array_elements_num,
	unsigned int *counter)
{

	unsigned int *threadInput;
	unsigned int threadInput_idx = 0;
	unsigned int block_idx = 0;
	unsigned int blockNum = inputfilesize / (THREAD_ELEMENT * THREAD_NUM) + (inputfilesize % (THREAD_ELEMENT * THREAD_NUM) != 0);
	__shared__ struct Codetable shared_codetable[MAX_CODE_NUM];
	__shared__ unsigned long long int shared_exclusive_sum;
	__shared__ unsigned int shared_block_idx;

	// ? All w threads in CUDA block i copies the codebook to compute h in the shared memory
	for (int i = threadIdx.x; i < MAX_CODE_NUM; i += blockDim.x)
	{
		shared_codetable[i] = codetable[i];
	}
	if (threadIdx.x == 0)
	{
		shared_block_idx = atomicAdd(counter, 1);
	}
	__syncthreads();
	block_idx = shared_block_idx;
	threadInput_idx = (block_idx * blockDim.x + threadIdx.x) * THREAD_ELEMENT;

	while (block_idx < blockNum)
	{
		unsigned int window = 0;
		unsigned int window_pos = 0;
		unsigned int output_pos = 0;
		unsigned int input_pos = 0;
		unsigned int output_code_bits = 0;
		unsigned int input = 0;
		threadInput = inputfile + (threadInput_idx / 4);

		// Get length per thread input
		// /------------------------------------------------------------/
		input = threadInput[0];
		while (input_pos < THREAD_ELEMENT && threadInput_idx + input_pos < inputfilesize)
		{
			const struct Codetable code = shared_codetable[GET_CHAR(input, input_pos & 3)];
			output_code_bits += code.length;
			input_pos++;
			if ((input_pos & 3) == 0)
				input = threadInput[input_pos / 4];
		}
		// /------------------------------------------------------------/

		// compute prefixsums
		// prefixsums in warps
		int warpLane = threadIdx.x & (WARP_SIZE - 1);
		int warpIdx = threadIdx.x / WARP_SIZE;
		unsigned int tmp_output_code_bits = output_code_bits;
		unsigned int tmp_value = 0;
		for (int delta = 1; delta < WARP_SIZE; delta <<= 1)
		{
			tmp_value = __shfl_up_sync(0xFFFFFFFF, tmp_output_code_bits, delta, WARP_SIZE);
			if (warpLane >= delta)
				tmp_output_code_bits += tmp_value;
		}

		// prefixsums in blocks
		__shared__ unsigned int shared_output_bits[THREAD_NUM / WARP_SIZE];
		if (warpLane == WARP_SIZE - 1)
		{
			shared_output_bits[warpIdx] = tmp_output_code_bits;
		}
		__syncthreads();

		// Threads in the first warps compute prefixsums
		if (threadIdx.x < TNUM_DIV_WSIZE)
		{
			tmp_value = shared_output_bits[threadIdx.x];
			const unsigned int shfl_mask = ~((~0) << TNUM_DIV_WSIZE);
			for (int delta = 1; delta < TNUM_DIV_WSIZE; delta <<= 1)
			{
				unsigned int tmp = __shfl_up_sync(shfl_mask, tmp_value, delta, TNUM_DIV_WSIZE);
				if (threadIdx.x >= delta)
					tmp_value += tmp;
			}
			shared_output_bits[threadIdx.x] = tmp_value;
		}
		__syncthreads();

		// The first block looks back 32 blocks simultaneously.
		if (warpIdx == 0)
		{
			int posIdx = block_idx - warpLane;
			unsigned long long int local_inclusive_sum = 0;
			unsigned long long int exclusive_sum = 0;
			if (warpLane == 0)
			{
				local_inclusive_sum = shared_output_bits[TNUM_DIV_WSIZE - 1];
				if (block_idx == 0)
				{
					inclusive_sum[block_idx] = local_inclusive_sum | FLAG_P;
					shared_exclusive_sum = 0;
				}
				else
				{
					inclusive_sum[block_idx] = local_inclusive_sum | FLAG_A;
				}
			}

			if (block_idx > 0)
			{
				while (1)
				{
					while (posIdx > 0 && (exclusive_sum == 0))
					{
						exclusive_sum = inclusive_sum[posIdx - 1];
					}
					unsigned long long int tmp_sum = 0;
					for (unsigned int delta = 1; delta < WARP_SIZE; delta <<= 1)
					{
						tmp_sum = __shfl_down_sync(0xFFFFFFFF, exclusive_sum, delta, WARP_SIZE);
						if (warpLane < (WARP_SIZE - delta) && ((exclusive_sum & FLAG_P) == 0))
							exclusive_sum += tmp_sum;
					}
					local_inclusive_sum += (exclusive_sum & (~FLAG_MASK));
					exclusive_sum = __shfl_sync(0xFFFFFFFF, exclusive_sum, 0);

					if (exclusive_sum & FLAG_P)
					{
						break;
					}

					posIdx -= WARP_SIZE;
					exclusive_sum = 0;
				}
				if (warpLane == 0)
				{
					inclusive_sum[block_idx] = ((local_inclusive_sum & (~FLAG_MASK)) | FLAG_P);
					shared_exclusive_sum = local_inclusive_sum - shared_output_bits[TNUM_DIV_WSIZE - 1];
				}
			}
		}
		__syncthreads();

		unsigned long long int exclusive_sum = 0;
		unsigned int tmp_count_plus = 0;
		if (warpLane == 0)
		{
			exclusive_sum = shared_exclusive_sum;
		}
		if (warpIdx > 0 && warpLane == 0)
		{
			tmp_count_plus = shared_output_bits[warpIdx - 1];
		}
		exclusive_sum = __shfl_sync(0xFFFFFFFF, exclusive_sum, 0);
		tmp_output_code_bits += __shfl_sync(0xFFFFFFFF, tmp_count_plus, 0);

		exclusive_sum = exclusive_sum + tmp_output_code_bits - output_code_bits;
		// /------------------------------------------------------------/
		// Output encoded data
		// /------------------------------------------------------------/
		unsigned int output_bits = (exclusive_sum & (SEGMENT_SIZE - 1));
		window_pos = (exclusive_sum & (MAX_BITS - 1));
		output_pos = (exclusive_sum / MAX_BITS);

		window = 0;
		input_pos = 0;
		int first_flag = 1;
		int last_out_flag = 0;
		input = threadInput[0];
		while (input_pos < THREAD_ELEMENT && threadInput_idx + input_pos < inputfilesize)
		{
			struct Codetable code = shared_codetable[GET_CHAR(input, input_pos & 3)];
			input_pos++;
			if ((input_pos & 3) == 0)
				input = threadInput[input_pos / 4];
			while (window_pos + code.length < MAX_BITS && threadInput_idx + input_pos < inputfilesize && input_pos < THREAD_ELEMENT)
			{
				window <<= code.length;
				window += code.code;
				window_pos += code.length;

				output_bits += code.length;
				if (threadInput_idx + input_pos < inputfilesize && input_pos < THREAD_ELEMENT)
				{
					code = shared_codetable[GET_CHAR(input, input_pos & 3)];
					input_pos++;
					if ((input_pos & 3) == 0)
						input = threadInput[input_pos / 4];
				}
			}

			output_bits += code.length;
			if (output_bits / SEGMENT_SIZE != (output_bits - code.length) / SEGMENT_SIZE)
			{
				const int gap_pos = output_pos / (SEGMENT_SIZE / MAX_BITS);
				unsigned int gap_elements = output_bits & (MAX_CODEWORD_LENGTH - 1);
				gap_array_bytes[gap_pos] = gap_elements;
			}

			const int diff = window_pos + code.length - MAX_BITS;
			last_out_flag = diff;

			if (diff >= 0)
			{
				window <<= code.length - diff;
				window += (code.code >> diff);

				if (first_flag)
				{
					atomicOr(&outputfile[output_pos++], window);
					first_flag = 0;
				}
				else
				{
					outputfile[output_pos++] = window;
				}

				window = code.code & ~(~0 << diff);
				window_pos = diff;
			}
			else
			{
				window <<= code.length;
				window |= code.code;

				const int shift = MAX_BITS - (window_pos + code.length);
				window <<= shift;
				atomicOr(&outputfile[output_pos++], window);
				window_pos = 0;
				last_out_flag = 0;
			}
		}
		// Output remained bits
		if (last_out_flag != 0)
		{
			window <<= (MAX_BITS - last_out_flag);
			atomicOr(&outputfile[output_pos++], window);
		}
		// assign segments to blocks
		if (threadIdx.x == 0)
			shared_block_idx = atomicAdd(counter, 1);
		__syncthreads();
		block_idx = shared_block_idx;
		threadInput_idx = (block_idx * blockDim.x + threadIdx.x) * THREAD_ELEMENT;
		// /------------------------------------------------------------/
	}
}

// Kernel for gathering elements for gap array
__global__ void cuGetGapArray( uint *gap_array_bytes, uint *gap_array, uint gap_array_elements_num, uint gap_array_size ){
	__shared__ uint shared_gap_array[GAP_BLOCK_RANGE];

	int block_start_pos = blockIdx.x*GAP_BLOCK_RANGE;
	int thread_start_pos = threadIdx.x*GAP_ELEMENTS_NUM;
	int gap_array_idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint gap_element=0;

	for(int i=threadIdx.x; i<GAP_BLOCK_RANGE && block_start_pos+i < gap_array_elements_num; i+=blockDim.x){
		shared_gap_array[i] = gap_array_bytes[ block_start_pos + i ];
	}
	__syncthreads();

	for(int i=0; i<GAP_ELEMENTS_NUM && block_start_pos+thread_start_pos+i < gap_array_elements_num; i++){
		const int gap_shift = GAP_LENGTH_MAX*i;
		gap_element |= (shared_gap_array[ thread_start_pos + i ]<<gap_shift);

	}
	if( gap_array_idx < gap_array_size )
		gap_array[gap_array_idx] = gap_element;

}


void encode(
	uint *output_file, 
	uint output_file_size,
	uint *d_input_file,
	uint input_file_size,
	uint gap_array_elements_num,
	struct Codetable* codetable
){
	uint *d_output_file, *d_gap_array_bytes, *d_gap_array, *d_counter;
	unsigned long long int *d_inclusive_sum;
	struct Codetable *d_codetable;

	uint *gap_array;
	gap_array = output_file + output_file_size;
	uint gap_array_size = 0;
	gap_array_size = (gap_array_elements_num + GAP_ELEMENTS_NUM - 1) / GAP_ELEMENTS_NUM;

	puts("Mallocing...");
	uint block_num = (input_file_size + THREAD_ELEMENT * THREAD_NUM-1)/(THREAD_ELEMENT*THREAD_NUM);
	cudaMalloc(&d_output_file, sizeof(int)*output_file_size);
	cudaMalloc(&d_codetable, sizeof(struct Codetable)*MAX_CODE_NUM);
	cudaMalloc(&d_inclusive_sum, sizeof(unsigned long long int) * block_num);
	cudaMalloc(&d_gap_array_bytes, sizeof(unsigned long long int) * gap_array_elements_num);
	cudaMalloc(&d_gap_array, sizeof(unsigned int) * gap_array_size);
	cudaMalloc(&d_counter, sizeof(int));
	CUERROR
	puts("DONE - Mallocing...");

	puts("Memsetting");
	cudaMemset(d_output_file, 0, sizeof(int) * output_file_size);
	cudaMemset(d_inclusive_sum, 0, sizeof(unsigned long long int) * block_num);
	cudaMemset(d_gap_array_bytes, 0, sizeof(int) * gap_array_elements_num);
	cudaMemset(d_counter, 0, sizeof(int));
	CUERROR
	puts("Done - Memsetting");

	puts("Memcpying");
	cudaMemcpy(d_codetable, codetable, sizeof(struct Codetable)*MAX_CODE_NUM, cudaMemcpyHostToDevice);
	CUERROR
	puts("Done - Memcpying");

	puts("Calling kernel");
	cuEncoder<<<BLOCK_NUM, THREAD_NUM>>>(
		d_output_file,
		output_file_size,
		d_input_file,
		input_file_size, 
		d_codetable,
		d_inclusive_sum, 
		d_gap_array_bytes,
		gap_array_elements_num,
		d_counter
	);
	cudaDeviceSynchronize();
	CUERROR
	puts("Done - calling kernel");

	block_num = gap_array_size;
	block_num = (BLOCK_NUM+GAP_THREADS-1) / GAP_THREADS;

	puts("Calling kernel 2");
	cuGetGapArray<<<block_num, GAP_THREADS>>>(d_gap_array_bytes, d_gap_array, gap_array_elements_num, gap_array_size);
	CUERROR
	puts("Done - Calling kernel 2");

	puts("Memcpying back");
	cudaMemcpy(output_file, d_output_file, sizeof(int)*output_file_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(gap_array, d_gap_array, sizeof(int)*gap_array_size, cudaMemcpyDeviceToHost);
	puts("Done - Memcpying back");

	cudaFree(d_input_file);
	cudaFree(d_output_file);
	cudaFree(d_codetable);
	cudaFree(d_inclusive_sum);
	cudaFree(d_gap_array_bytes);
	cudaFree(d_gap_array);
}