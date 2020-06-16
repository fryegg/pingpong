#pragma once
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace std;
class moveman
{


public: 
	void move_manipulator(float send_x, float send_y, float send_z, float send_vx, float send_vy, float send_vz, float send_t);
	float send_x;
	float send_y;
	float send_z;
	const char* command1;
};	