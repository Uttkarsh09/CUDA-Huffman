#include "decompress.hpp"

void fatal(const char* str){
	fprintf( stderr, "%s! at %s in %d\n", str, __FILE__, __LINE__);
	exit(EXIT_FAILURE);
}


unsigned int getTableInfo(
		struct Symbol* symbols,
		int symbol_count,
		int prefix_bit,
		struct TableInfo &table_info){

	unsigned int code=0;
	unsigned int previous_prefix_code=0;
	unsigned int prefix_code = 0;
	unsigned int ptrtable_size=0;
	unsigned int l2table_size=0;
	unsigned int l1table_size=0;
	unsigned int prefix_mask = ~ ( (~(0))<<prefix_bit );

	for(int i=0; i<symbol_count; i++){
		int current_length = symbols[i].length;
		int next_length = (i+1 == symbol_count) ? current_length : symbols[i+1].length;
		code = (code+1) << (next_length - current_length);

		if( current_length > prefix_bit ){
			unsigned int prefix_shift = current_length - prefix_bit;
			prefix_code = ((code>>prefix_shift)&prefix_mask);
			if( l1table_size == 0 ){
				l1table_size = (code>>prefix_shift);
			}

			if( previous_prefix_code != prefix_code ){
				l2table_size += 1<<prefix_shift;
				ptrtable_size++;
			}
			previous_prefix_code = prefix_code;
		}
	}

	table_info.l1table_size = l1table_size;
	table_info.l2table_size = l2table_size;
	table_info.ptrtable_size = ptrtable_size;
	if( l1table_size == 0 ){
		int bit = symbols[symbol_count-1].length;
		table_info.l1table_size = (1<<bit);
		return bit;
	}
	return prefix_bit;
}


unsigned int getTwolevelTable(
		void *stage_table,
		int prefix_bit,
		struct Symbol* symbols, 
		int symbol_count,
		struct TableInfo table_info){
	
	unsigned int code = 0;
	unsigned int boundary_code = 0;
	unsigned char *l1_table;
	unsigned char *l2_table;
	unsigned char *length_table;
	unsigned int *ptr_table;

	unsigned int ptrIdx = 0;
	unsigned char ptr_counter=0;

	ptr_table = (unsigned int *)stage_table;
	length_table = (unsigned char *)stage_table + table_info.ptrtable_size*sizeof(unsigned int);
	l1_table = length_table + MAX_CODE_NUM;
	l2_table = l1_table + table_info.l1table_size;

	for( int i=0; i<symbol_count; ){
		int current_length = symbols[i].length;
		unsigned char current_symbol = symbols[i].symbol;
		length_table[current_symbol] = current_length;

		if( current_length <= prefix_bit ){
			const int prefix_shift = prefix_bit - current_length;
			const int num_elements = 1<<prefix_shift;
			for( int elementsIdx = 0; elementsIdx < num_elements; elementsIdx++ ){
				l1_table[ (code<<prefix_shift) + elementsIdx ] = current_symbol;
			}
			int next_length = (i==symbol_count-1) ? current_length : symbols[i+1].length;
			code = ((code+1) << (next_length - current_length));
			i++;
		}else{
			int prefix_shift = current_length - prefix_bit;
			int prefix_shift_tmp = prefix_shift;
			unsigned int prefix_mask = ~( (~0)<<prefix_bit );

			unsigned int code_tmp = code;
			unsigned int last_length=current_length - prefix_bit;
			int counter = 0;

			if( boundary_code == 0 ){
				boundary_code = (code>>prefix_shift);
			}

			ptr_counter++;

			code_tmp = code;
			while( ((code_tmp>>prefix_shift_tmp)&prefix_mask) == ((code>>prefix_shift)&prefix_mask) ){
				if( i+counter==symbol_count ) break;
				counter++;
				current_length = symbols[i + counter-1].length;
				int next_length = (i+counter==symbol_count-1) ? current_length : symbols[i+counter].length;
				prefix_shift_tmp = next_length - prefix_bit;
				last_length = current_length - prefix_bit;
				code_tmp = ((code_tmp+1) << (next_length - current_length));
			}
			code_tmp = code;

			const unsigned int table_width = last_length;
			unsigned char* this_table = l2_table + ptrIdx;
			for(int symbolIdx=0; symbolIdx<counter; symbolIdx++){
				current_length = symbols[ i + symbolIdx ].length;
				current_symbol = symbols[ i + symbolIdx ].symbol;
				length_table[current_symbol] = current_length;
				const unsigned int suffix_shift = current_length - prefix_bit;
				const unsigned int suffix_mask = ~ ( (~(0)) << suffix_shift );
				const unsigned int suffix_code = (code_tmp&suffix_mask)<<(table_width-suffix_shift);
				const int elementsNum = 1<<(table_width-suffix_shift);


				for(int elementsIdx=0; elementsIdx<elementsNum; elementsIdx++){
					this_table[ suffix_code + elementsIdx ] = current_symbol;
				}
				int next_length = (i+symbolIdx==symbol_count-1) ? current_length : symbols[i+symbolIdx+1].length;
				code_tmp = ((code_tmp+1) << (next_length - current_length));
			}
			
			unsigned int ptr_value = (table_width<<16) | ptrIdx;
			ptr_table[ ptr_counter-1 ] = ptr_value;

			i += counter;
			ptrIdx += (1<<table_width);
			code = code_tmp;
		}
	}
	return boundary_code;
}


void Decompress::decompressController(){
	FILE *input_file_ptr, *output_file_ptr;
	size_t tmp_st;
	int symbol_count;
	struct Symbol *symbols;
	uint *input, *output;
	Logger *l = Logger::GetInstance("");

	l->logIt(l->INFO, "Creating io pointers");
	openFile(&input_file_ptr, input_file_name, "rb");
	openFile(&output_file_ptr, output_file_name, "wb");
	l->logIt(l->INFO, "output file name = %s<-", output_file_name.c_str());
	l->logIt(l->INFO, "DONE - Creating io pointers");

	if(1!=fread(&tmp_st, sizeof(size_t), 1, input_file_ptr)){
		l->logIt(Logger::ERROR, "Cannot read %s at %s:%d", input_file_name, __FILE__, __LINE__-1);
		l->logIt(Logger::ERROR, "Exiting");
		l->~Logger();
		exit(EXIT_FAILURE);
	}

	symbol_count = tmp_st;
	
	cudaMallocHost(&symbols, sizeof(struct Symbol)*symbol_count);

	for(int i=0 ; i<symbol_count ; i++){
		unsigned char tmp_symbol;
		unsigned char tmp_length;
		int result = 2;
		
		result = fread(&tmp_symbol, sizeof(tmp_symbol), 1, input_file_ptr);
		if(result != 1)
			l->logIt(l->ERROR, "Error in reading file @ %s line %d", __FILE__, __LINE__);

		result = fread(&tmp_length, sizeof(tmp_length), 1, input_file_ptr);
		if(result != 1)
			l->logIt(l->ERROR, "Error in reading file @ %s line %d", __FILE__, __LINE__);
		
		symbols[i].symbol = tmp_symbol;
		symbols[i].length = tmp_length;		
	}

	uint prefix_bit = FIXED_PREFIX_BIT;
	struct TableInfo h_table_info;
	void* stage_table;
	int stage_table_size;

	for(int i=0 ; i<LOOP ; i++){
		prefix_bit = getTableInfo(symbols, symbol_count, prefix_bit, h_table_info);
	}
	
	stage_table_size =  sizeof(int)*h_table_info.ptrtable_size + sizeof(char)*(MAX_CODE_NUM + h_table_info.l1table_size + h_table_info.l2table_size);
	cudaMallocHost(	&stage_table, stage_table_size);

	getTwolevelTable(
		stage_table,
		prefix_bit,
		symbols, 
		symbol_count,
		h_table_info 
	);
	
	int compressed_size, original_size, gap_array_size, gap_elements_num;
	if( 1 != fread( &original_size, sizeof(int), 1, input_file_ptr ) ) fatal("File read error 1");
	if( 1 != fread( &compressed_size, sizeof(int), 1, input_file_ptr ) ) fatal("File read error 2");
	if( 1 != fread( &gap_elements_num, sizeof(int), 1, input_file_ptr ) ) fatal("File read error 3");

	gap_array_size = (gap_elements_num + GAP_FAC_NUM-1)/GAP_FAC_NUM;

	cudaMallocHost( &input, sizeof(int)*(gap_array_size + compressed_size) );
	cudaMallocHost( &output, sizeof(int)*((original_size+3)/4) );

	if((size_t)(gap_array_size+compressed_size) != fread(input, sizeof(unsigned int), gap_array_size+compressed_size, input_file_ptr))
		fatal("File read error");

	printf("file,%s,", output_file_name.c_str());

	decoder_l1_l2(
		input,
		compressed_size,
		output,
		original_size,
		gap_elements_num,
		stage_table,
		stage_table_size,
		prefix_bit,
		symbol_count,
		h_table_info	
	);

	fwrite(output, sizeof(char), original_size, output_file_ptr);

	fclose(input_file_ptr);

	cudaFreeHost(symbols);
	cudaFreeHost(input);
	cudaFreeHost(output);
	cudaFreeHost(stage_table);
}