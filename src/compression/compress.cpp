#include "compress.hpp"
#include "compressKernels.cuh"

int compare(const void *p, const void *q){
	struct Symbol *P = (struct Symbol *)p;
	struct Symbol *Q = (struct Symbol *)q;
	return P->num - Q->num;
}

void Compress::compressController(){
	Logger* l = Logger::GetInstance("./logs/logs.log");
	FILE *input_ptr, *output_ptr;
	uint *input_file, *output_file, *gap_array, *num_of_symbols;
	unsigned long long int output_file_size_bits = 0;
	uint gap_array_size=0, gap_array_elements_num=0;
	num_of_symbols = (uint*)malloc(sizeof(int) * MAX_CODE_NUM);
	struct Symbol symbols[MAX_CODE_NUM] = {};
	struct Codetable *codetable;

	openFile(&input_ptr, input_file_name, "rb");
	openFile(&output_ptr, output_file_name, "wb");
	
	input_file_size_bytes = getFileSize(input_ptr);
	
	l->logIt(Logger::INFO, "Allocating Host mem");
	cudaMallocHost(&codetable, sizeof(struct Codetable) * MAX_CODE_NUM);
	CUERROR
	cudaMallocHost(&input_file, sizeof(int) * ((input_file_size_bytes+3)/4));
	CUERROR
	l->logIt(Logger::INFO, "Done - Allocated host memory");

	l->logIt(l->INFO, "reading input file");
	l->logIt(l->INFO, "size of input file = %lld bytes", input_file_size_bytes);
	fread(input_file, sizeof(char), input_file_size_bytes, input_ptr);
	fsync(input_ptr->_fileno);
	l->logIt(l->INFO, "DONE - reading input file");
	
	
	int symbol_count = 0;

	uint *d_num_of_symbols;
	uint *d_inputfile;

	l->logIt(l->INFO, "Allocating CUDA memory for d_inputfile[%lld] and d_num_of_symbols[%lld]", (input_file_size_bytes+3)/4 + 1, sizeof(int)*MAX_CODE_NUM);
	l->logIt(l->INFO, "MAX_CODE_NUM = %lld", MAX_CODE_NUM);
	cudaMalloc((void**)&d_inputfile, sizeof(int)*((input_file_size_bytes+3)/4 + 1));
	CUERROR
	cudaMalloc((void**)&d_num_of_symbols, sizeof(int)*MAX_CODE_NUM);
	CUERROR
	l->logIt(l->INFO, "Allocated CUDA memory");

	l->logIt(l->INFO, "Copying input file from Host->Device ");
	cudaMemcpy(d_inputfile, input_file, sizeof(int)*((input_file_size_bytes+3)/4), cudaMemcpyHostToDevice);
	CUERROR
	l->logIt(l->INFO, "Done Copying input file ");
	
	l->logIt(l->INFO, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
	l->logIt(l->INFO, "Generating char histogram");
	generateCharHistogram(d_num_of_symbols, d_inputfile, input_file_size_bytes);
	cudaMemcpy(num_of_symbols, d_num_of_symbols, sizeof(int)*MAX_CODE_NUM, cudaMemcpyDeviceToHost);
	l->logIt(l->INFO, "Printing HISTOGRAM");
	for(int i=0 ; i<MAX_CODE_NUM ; i++){
		// printf("%d(%c) = %lld\n", i, (char)i, num_of_symbols[i]);
		if(num_of_symbols[i]>0)
			l->logIt(l->INFO, "%d(%c) = %lld", i, (char)i, num_of_symbols[i]);
	}
	cudaFree(d_num_of_symbols);
	l->logIt(l->INFO, "Generated Symbols");
	l->logIt(l->INFO, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");

	// ? Store the symbol count in symbol struct which has its respective count and length
	symbol_count = storeSymbols(num_of_symbols, symbols);
	// ? store_symbols will filter out the symbols without any occourance
	qsort(symbols, symbol_count, sizeof(struct Symbol), compare);

	boundaryPackageMerge(symbols, symbol_count, codetable);
	
	output_file_size_bits = getOutputFileSize(symbols, symbol_count);
	gap_array_elements_num = (output_file_size_bits + SEGMENT_SIZE-1) / SEGMENT_SIZE;
	gap_array_size = gap_array_elements_num/GAP_ELEMENTS_NUM + ((gap_array_elements_num%GAP_ELEMENTS_NUM) != 0);

	output_file_size_bytes = (output_file_size_bits + MAX_BITS-1) / MAX_BITS;

	cudaMallocHost(&output_file, sizeof(uint) * (output_file_size_bytes + gap_array_size));
	CUERROR

	gap_array = output_file + output_file_size_bytes;

	encode(
		output_file,
		output_file_size_bytes,
		d_inputfile,
		input_file_size_bytes,
		gap_array_elements_num,
		codetable
	);
	CUERROR

	printf("File: %s\n", input_file_name.c_str());
	printf("Segment Size: %d\n", SEGMENT_SIZE);
	printf("Thread Num: %d\n", THREAD_NUM);
	printf("Bytes per Thread: %d\n", THREAD_ELEMENT);
	printf("\n");

	size_t tmp_symbol_count = symbol_count;
	fwrite(&tmp_symbol_count, sizeof(size_t), 1, output_ptr);

	for(int i=symbol_count-1 ; i>=0 ; i--){
		unsigned char tmp_symbol = symbols[i].symbol;
		unsigned char tmp_length = symbols[i].length;
		fwrite(&tmp_symbol, sizeof(tmp_symbol), 1, output_ptr);
		fwrite(&tmp_length, sizeof(tmp_length), 1, output_ptr);
	}

	fwrite(&input_file_size_bytes, sizeof(input_file_size_bytes), 1, output_ptr);
	fwrite(&output_file_size_bytes, sizeof(output_file_size_bytes), 1, output_ptr);
	fwrite(&gap_array_elements_num, sizeof(gap_array_elements_num), 1, output_ptr);
	fwrite(gap_array, sizeof(int), gap_array_size, output_ptr);
	fwrite(output_file, sizeof(unsigned int), output_file_size_bytes, output_ptr);

	fdatasync(output_ptr->_fileno);

	int outsize = ftell(output_ptr);
	printf("Ratio = output file size / input file size => %lf\n", (double)outsize/input_file_size_bytes);
	printf("Output file size: %d\n", outsize);
	printf("input file size: %d\n", input_file_size_bytes);
	
	fclose(input_ptr);
	fclose(output_ptr);

	cudaFreeHost(input_file);
	cudaFreeHost(output_file);
	cudaFreeHost(codetable);
}

int Compress::storeSymbols(uint *num_of_symbols, struct Symbol *symbols){
	int symbol_count = 0;
	for(int i=0 ; i<MAX_CODE_NUM ; i++){
		if(num_of_symbols[i] != 0){
			symbols[symbol_count].symbol = i;
			symbols[symbol_count].num = num_of_symbols[i];
			symbol_count++;
		}
	}

	return symbol_count;
}

void add_node(
	struct Symbol *symbols,
	int symbol_count,
	struct NodePackList &pack,
	int list_pos,
	int symbols_length
){
	struct NodePack *next_pack;
	struct NodePack *current_pack;
	struct NodePack *before_pack;
	int max_list_length = 2 * (symbol_count - 1);
	int above_pack_pos = 0;
	
	if(list_pos > 0){
		above_pack_pos = pack.current_pos[list_pos-1];
	}

	current_pack = &pack.list[max_list_length * list_pos + pack.current_pos[list_pos]];
	next_pack = &pack.list[max_list_length * list_pos + pack.current_pos[list_pos] + 1];
	int symbols_idx = current_pack->counter;
	next_pack->counter = current_pack->counter;

	if(list_pos > 0){
		// left
		{
			above_pack_pos = pack.current_pos[list_pos-1];
			before_pack = &pack.list[ max_list_length*(list_pos-1) + above_pack_pos];
			if( before_pack->weight <= symbols[symbols_idx].num || symbols_idx >= symbols_length){
				next_pack->left_weight = before_pack->weight;
				next_pack->chain = before_pack;
				add_node( symbols, symbol_count, pack, list_pos-1 , symbols_length);
			}
			else{
				next_pack->left_weight = symbols[symbols_idx++].num;
				next_pack->counter++;
				next_pack->chain = current_pack->chain;
			}
		}
		// right
		{
			above_pack_pos = pack.current_pos[list_pos-1];
			before_pack = &pack.list[ max_list_length*(list_pos-1) + above_pack_pos];
			if( before_pack->weight <= symbols[symbols_idx].num || symbols_idx >= symbols_length ){
				next_pack->right_weight = before_pack->weight;
				next_pack->chain = before_pack;
				add_node( symbols, symbol_count, pack, list_pos-1, symbols_length );
			}
			else{
				next_pack->right_weight = symbols[symbols_idx++].num;
				next_pack->counter++;
			}
		}
	}
	else{
		if(symbols_idx < symbols_length){
			next_pack->left_weight = symbols[symbols_idx++].num;
			next_pack->counter++;
		}

		if(symbols_idx < symbols_length){
			next_pack->right_weight = symbols[symbols_idx++].num;
			next_pack->counter++;
		}
		else{
			next_pack->right_weight = 0;
		}
		next_pack->chain = NULL;
	}

	pack.current_pos[list_pos] += 1;
	next_pack->weight = next_pack->left_weight + next_pack->right_weight;
}


void Compress::boundaryPackageMerge(struct Symbol *symbols, uint symbol_count, struct Codetable *codetable){
	struct NodePackList pack;
	int max_list_length = 2 * (symbol_count - 1);
	pack.list = (struct NodePack *) malloc(sizeof(struct NodePack)*max_list_length*MAX_CODEWORD_LENGTH);
	
	for(int i=0 ; i<MAX_CODEWORD_LENGTH ; i++){
		pack.current_pos[i] = 0;
	}

	int num_of_symbols = symbol_count;
	int looptime = (num_of_symbols - 1) * 2;

	// initalize NodePack
	for(int i=0 ; i<MAX_CODEWORD_LENGTH ; i++){
		struct NodePack init_node;
		init_node.left_weight = symbols[0].num;
		init_node.right_weight = symbols[1].num;
		init_node.weight = init_node.left_weight + init_node.right_weight;
		init_node.counter = 2;
		init_node.LR = LEFT;
		init_node.chain = NULL;
		init_node.pack_pos = 0;
		pack.list[i*max_list_length] = init_node;
	}

	int listpos = MAX_CODEWORD_LENGTH - 1;
	ull last_counter = 2;
	struct NodePack *before_pack;
	struct NodePack *chain_head;
	chain_head = NULL;
	before_pack = &pack.list[max_list_length * (listpos - 1)];

	for(int i=2; i<looptime ; i++){
		if(last_counter < symbol_count){
			if(before_pack->weight <= symbols[last_counter].num){
				add_node(symbols, symbol_count, pack, listpos-1, symbol_count);
				chain_head = before_pack;
				before_pack++;
			}
			else{
				last_counter++;
			}
		}
		else{
			add_node(symbols, symbol_count, pack, listpos-1, symbol_count);
			chain_head = before_pack;
			before_pack++;
		}
	}

	for(unsigned int i=0; i<last_counter; i++) symbols[i].length++;
	struct NodePack *chain;
	chain = chain_head;
	while( chain != NULL ){
		int counter = chain->counter;
		chain = chain->chain;
		for( int j=0; j<counter; j++ ) symbols[j].length++;
	}

	unsigned int code=0;
	unsigned int current_length=0;
	unsigned int next_length=0;

	current_length = symbols[symbol_count-1].length;
	for(int i=symbol_count-1; i>=0; i--){
		codetable[symbols[i].symbol].code = code;
		codetable[symbols[i].symbol].length = current_length;

		next_length = (i==0) ? current_length : symbols[i-1].length;

		code = (code + 1) << (next_length - current_length);
		current_length = next_length;
	}

}