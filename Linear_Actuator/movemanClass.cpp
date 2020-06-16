#pragma once
#include <tchar.h> 
#include <iostream>
#include <math.h>
#include <iostream>
#include <SerialClass.h>
#include "makecommandClass.h"
#include "movamanClass.h"

void moveman::move_manipulator(float send_x, float send_y, float send_z, float send_vx, float send_vy, float send_vz, float send_t) //원래 manipulator 움직이는 함수
{
	float Ovx = 1;
	float Ovy = 0;
	float Ovz = 0;
	float Ix = send_x; float Iy = send_y; float Iz = send_z; float Ivx = send_vx; float Ivy = send_vy; float Ivz = send_vz;
	float lv[3];
	lv[0] = Ivx; lv[1] = Ivy; lv[2] = Ivz;  // input position& velocity
	float Ov[3];
	Ov[0] = Ovx; Ov[1] = Ovy; Ov[2] = Ovz; // output position& velocity
	float N[3];
	N[0] = lv[0] + Ov[0];
	N[1] = lv[1] + Ov[1];
	N[2] = lv[2] + Ov[2]; // 탁구채의 Normal vector
	float norm = sqrt(pow(N[0], 2) + pow(N[1], 2) + pow(N[2], 2));
	float rad = 0.015 * acos(N[0] / norm); //gripper angle
	float send = acos(N[0] / norm); //send angle

	float sendxposition = 0.274 * cos(send); // send x point = Ix
	float sendyposition = 0.274 * sin(send); //send y point = Iy
	float limit;
	float Sx; float Sy; float Sz; float Ex; float Ey; float Ez;

	if (Iy > 0)
	{
		Sx = Ix - 0.05 * cos(send); Sy = 0.274 + 0.05 * sin(send); Sz = Iz; // swing start point
		Ex = Ix + 0.05 * cos(send); Ey = 0.274 - 0.05 * sin(send); Ez = Iz; // swing end point
	}
	else
	{
		Sx = Ix - 0.05 * cos(send); Sy = -(0.274 + 0.05 * sin(send)); Sz = Iz; // swing start point
		Ex = Ix + 0.05 * cos(send); Ey = -(0.274 - 0.05 * sin(send)); Ez = Iz; // swing end point
	}

	
	//Sleep(1);
	//cout << Sx << endl;
	//cout << Ex << endl;
   // Serial open
	//const char* command = make_signal_msg2.c_str();
	//make_signal_msg2(std::to_string(send_x), std::to_string(send_y), std::to_string(send_z), std::to_string(0), std::to_string(0.1));

	//std::string signal_msg1 = makecommand(std::to_string(send_x,send_y,send_z,float(0),float(1)));
	//std::string signal_msg1 = make_signal_msg2(abcdef);

	//std::string command = "joint," + std::to_string(0) + "," + std::to_string(-0.785) + "," + std::to_string(0) + +"," + std::to_string(0);
	//std::string signal_msg1 = "joint,0,0,0,0";
	//Serial_ptr->write_data()
	//const char* command1 = signal_msg1.c_str();
	//cout << signal_msg1 << endl;
	//const char* pcom = command1;
	//cout << command1 << endl;	
	makecom makecom;
	Serial* SP = new Serial("\\\\.\\COM3");
	SP->WriteData(command1,1024);  // serial write - main 함수 안으로 옮겨야 되나?
	Sleep(1200); //time sleep - 1200 이하는 command가 씹힘
	//Sleep(1);

		//limit
	/*if ((Iz < 0.08) || (Iz > 0.305) || (Iy > 0.274) || (Iy < -0.274))
	{

	}
	else
	{*/
	//String signal_msg1 = make_signal_msg(std::to_string(Sx), std::to_string(Sy), std::to_string(Sz), std::to_string(0), std::to_string(float(0.1)));
	//Serial_ptr->write_data(signal_msg1);  // ready
	//String signal_msg2 = make_signal_msg(std::to_string(Ex), std::to_string(Ey), std::to_string(Ez), std::to_string(rad), std::to_string(float(0.1)));
	//Serial_ptr->write_data(signal_msg2);  // swing
	//Sleep(11);
//}
// openmanipulator command
//if (limit == 1) // 칠 수 없을때
//{
//	//Serial_ptr->write_data("pos,0.224,-0.005,0.105,0,1");
//}
//else// 칠 수 있을때
//{
//	String signal_msg1 = make_signal_msg(std::to_string(Sx), std::to_string(Sy), std::to_string(Sz), std::to_string(0), std::to_string(float(0.1)));
//	Serial_ptr->write_data(signal_msg1);  // ready
//	Sleep(11);
//	String signal_msg2 = make_signal_msg(std::to_string(Ex), std::to_string(Ey), std::to_string(Ez), std::to_string(rad), std::to_string(float(0.1)));
//	Serial_ptr->write_data(signal_msg2);  // swing
//	Sleep(11);

//	//Serial_ptr->write_data("jointandtool,0,0.274,0.205,0,0.5"); // go home
//	//Serial_ptr->write_data("jointandtool,0,0.274,0.205,0,0.5"); // go home
//}
	//return;
}
