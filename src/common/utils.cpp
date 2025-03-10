#include "utils.hpp"

void openFile(FILE **fptr, string file_name, string access_modifier){
	*fptr = fopen(file_name.c_str(), access_modifier.c_str());
	if(*fptr == NULL){
		printf("CANNOT OPEN FILE %s <-\n", file_name.c_str());
		exit(EXIT_FAILURE);
	}
}

uint getFileSize(FILE *fptr){
	uint size;
	fseek(fptr, 0L, SEEK_END);
	size = ftell(fptr);
	fseek(fptr, 0L, SEEK_SET);

	return size;
}

unsigned long long int getOutputFileSize(struct Symbol *symbols, uint symbols_count ){
	unsigned long long int output_file_size = 0;

	for(uint i=0 ; i<symbols_count ; i++){
		struct Symbol symbol = symbols[i];
		// ? Length of codewoard * num of occourances
		output_file_size += symbol.length * symbol.num;
	}
	
	return output_file_size;
}