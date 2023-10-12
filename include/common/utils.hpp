#pragma once
#include <iostream>
#include "macros.hpp"

using namespace std;

void openFile(FILE **fptr, string file_name, string access_modifier);

uint getFileSize(FILE *fptr);

unsigned long long int getOutputFileSize(struct Symbol *symbols, uint symbols_count );