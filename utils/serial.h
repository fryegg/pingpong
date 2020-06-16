#pragma once
#include <sys/types.h>
#include <windows.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <math.h>
#include <float.h>
#include <cmath>
#include <sstream>

class CSerial_ctr
{
public:
	CSerial_ctr(std::string port);
	~CSerial_ctr();
	HANDLE hSerial;
	DCB dcbSerialParams = { 0 };
	COMMTIMEOUTS timeouts = { 0 };

	void write_data(std::string buf);
	std::string read_data();
	std::wstring s2ws(const std::string& s);
};

