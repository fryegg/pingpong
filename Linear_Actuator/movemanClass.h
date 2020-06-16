#pragma once
#include <stdio.h>
#include <math.h>
#include <string>
#include <windows.h>	// Library described above


using namespace::std;

class moveman
{



public:
	int move_manipulator(float send_x, float send_y, float send_z, float send_vx, float send_vy, float send_vz, float send_t); // manipulator class - movemanip.cpp ÂüÁ¶
	const char* command1;
	
};