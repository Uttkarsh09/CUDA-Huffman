cpp_files='./src/common/*.cpp ./src/compression/compress.cpp ./src/decompression/decompress.cpp'
cu_files='./src/compression/compressKernels.cu ./src/decompression/decompressKernels.cu'
include_file='-I./include/common/ -I./include/compression/ -I./include/decompression'
nvcc -o bin/main ${cpp_files} ${cu_files} ${include_file} ./src/main.cpp  