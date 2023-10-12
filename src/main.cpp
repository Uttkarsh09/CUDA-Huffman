#include <iostream>
#include "compress.hpp"
#include "decompress.hpp"
#include "logger.hpp"

using namespace std;

int main(int argc, char**argv){
	if(argc != 3){
		cout << "Invalid command line arguments for huffman cpu" << endl;
		cout << "expected exe -c input_file" << endl;
		cout << "OR" << endl;
		cout << "expected exe -dc input_file(with .huff extension)" << endl;
		exit(EXIT_FAILURE);
	}
	string mode(argv[1]);
	string input_file(argv[2]);
	string output_file;
	Logger* l = Logger::GetInstance("./logs/logs.log");

	if(mode == "-c"){
		output_file = input_file+".huff";
		l->logIt(Logger::INFO, "Starting compression");
		Compress c(input_file, output_file);
		c.compressController();
		l->logIt(Logger::INFO, "Done - Compression");
	}
	else {
		output_file = input_file.substr(0, input_file.size()-5);
		l->logIt(Logger::INFO, "Starting Decompression");
		Decompress dc(input_file, output_file);
		dc.decompressController();
		l->logIt(Logger::INFO, "Done - Decompression");		
	}

	l->deleteInstance();

	return 0;

	
}