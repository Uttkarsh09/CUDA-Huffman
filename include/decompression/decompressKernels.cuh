#pragma once
#ifndef DECOMPRESS_KERNELS
#define DECOMPRESS_KERNELS

#include <iostream>
#include "macros.hpp"

void decoder_l1_l2(
	unsigned int *input,
	int input_file_size,
	unsigned int *output,
	int output_file_size,
	int gap_element_num,
	void *dectable,
	int table_size,
	unsigned int prefix_bit,
	unsigned int symbol_count,
	struct TableInfo table_info
);

void cu_make_table_call(
	struct Symbol *symbols,
	struct TableInfo *table_info,
	int symbol_count,
	int prefix_bit,
	void *decode_table 
);

#endif