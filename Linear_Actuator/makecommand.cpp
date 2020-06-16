#pragma once
#include <tchar.h> 
#include <iostream>
#include <iostream>
#include <string.h>
#include "makecommandClass.h"
#include "movamanClass.h"

using namespace std;
const char* makecom::sendcom(const char* command)
{
	/*std::string make_signal_msg2(string x_value, string y_value, string z_value, string rad, string num)
	{*/
		//return "jointandtool," + x_value + "," + y_value + "," + z_value + +"," + rad + +"," + num;
	moveman moveman;
		string p1 = to_string(moveman.send_x);
		string p2 = to_string(moveman.send_y);
		string p3 = to_string(moveman.send_z);
		string r = to_string(moveman.rad);
		string n = to_string(moveman.send_t);;
		char com[50] = "joint";
		const char* po1 = p1.c_str();
		const char* po2 = p2.c_str();
		const char* po3 = p3.c_str();
		const char* ra = r.c_str();
		const char* nu = n.c_str();

		strcat_s(com, ",");
		strcat_s(com, po1);
		strcat_s(com, ",");
		strcat_s(com, po2);
		strcat_s(com, ",");
		strcat_s(com, po3);
		strcat_s(com, ",");
		strcat_s(com, ra);
		strcat_s(com, ",");
		strcat_s(com, nu);

		cout << com << endl;
		const char* command1 = com;
	
}


//const char* makecom::sendcom(const char* command)
//{
//	return command1;
//
//}