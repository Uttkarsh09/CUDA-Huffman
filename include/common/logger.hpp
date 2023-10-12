#pragma once
#include <iostream>
#include <stdarg.h>

using namespace std;

class Logger{
public:
	enum LogLevel{
		OFF,
		CRITICAL,
		ERROR,
		WARNING,
		INFO,
		DEBUG,
		ALL
	};

	void operator=(const Logger&) = delete;
	static Logger *GetInstance(const string value);
	void deleteInstance();
	void logIt(LogLevel level, const char * format, ...);
	void setLogLevel(LogLevel level);
	string getLogLevelName(LogLevel level);
	
	Logger() = delete;

protected:
	static Logger* logger_;
	LogLevel logLevel;
	string value_;
	FILE *fptr;
	
	// ? inaccessable constructor
	Logger(const string file_name){
		fptr = fopen(file_name.c_str(), "w");
		logLevel = LogLevel::ALL;

		if(fptr == NULL){
			printf("ERROR - Cannot open file %s\n", file_name.c_str());
			exit(EXIT_FAILURE);
		}
	}
	
};
