#pragma once
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace std;


class makecom
{
public:

	const char* sendcom(const char*command);
	const char* command1;
	

	// make_signal_msg2(std::to_string(send_x), std::to_string(send_y), std::to_string(send_z), std::to_string(0), std::to_string(0.1));

};