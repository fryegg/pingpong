#pragma once

#define NOMINMAX
#include <iostream>
#include <thread>
#include <future>
#include <WinSock2.h>
#include <Ws2tcpip.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <math.h>
#include "CCamera_container.h"
#include <rs.hpp>
#include <hpp/rs_options.hpp>
#include <h/rs_option.h>
#include <rs_advanced_mode.hpp>
#include <rs_advanced_mode.h>
#include "Mod_CMatrix.h"
#include "CLinear_actu.h"
#include "serial.h"
#include "SerialClass.h"
#include "crtdbg.h"
#include <list>
#include <chrono>
#include <C:\Users\admin\Desktop\vision\boost_1_70_0\boost\math\special_functions/lambert_w.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include "movamanClass.h" // 참조 헤더 파일
#include "SerialClass.h" //새로 만든 Serial
#include "serial.h"
#include <tchar.h> 
#include <engine.h>
#include <cmath>
#include <stdlib.h>
#include <mutex>
#include "polyfit.hpp"

#define LOOPMAX 800
#define PI 3.14159265
///////////////////////////////////////////////////////////////////
using boost::math::lambert_w0;
using namespace std;
using namespace cv;
std::mutex mtx;

#pragma comment (lib, "libmat.lib")
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")
#pragma comment (lib, "libeng.lib")

list<int> l;
//---- control factor------//
//CSerial_ctr* Serial_ptr;
int ilowh = 9;
int ihighh = 31;

int ilows = 128;
int ihighs = 255;

int ilowv = 114;
int ihighv = 255;

int thresh1 = 100;
int thresh2 = 200;

/////////////////////////////////////////////////////////////////

int direction = 0;
int ball_num = 0; // 1번 던질 때 지나가면서 찍히는 공의 개수
float mola1;
float linear_y_ans;
// 1번
//int ilowh = 0;  
//int ihighh = 75;
//
//int ilows = 255;  
//int ihighs = 255;
//
//int ilowv = 74; 
//int ihighv = 255;

int width = 848; // 960 * 540
int height = 480;
int fps = 60;
float linear_y0;
float linear_y;
Point3f data_point;

// --------------------///
CLinear_actu* LM_G;

Mat* imghsv;
Mat* imghsv_;
Mat* imgthresholded;
Mat* imgthresholded_;
Mat imgthresholded_gray;
Mat* imgLines;
Mat image;
Mat* image_;
Mat* Depth_image;
vector<vector<Point> > contours1;
vector<Vec4i> hierarchy1;

int iLastX, iLastY, posX, posY;
rs2::config cfg;
rs2::pointcloud pc;
rs2::points points;
rs2::pipeline_profile selection;
rs2::context ctx;
rs2::pipeline pipe;
rs2::device_list devices;

bool save_flag = 0;
bool save_start_signal = 0;
bool save_end_signal = 0;
bool save_start_signal_once = 0;
bool save_end_signal_once = 0;

int time_data = 0;


Mod_CMatrix Mat_cal;
Mod_S3DCoord plane_point[4];

double rot_mat[9];
Mod_S3DCoord zero_point;
Mod_S3DCoord object_point, temp_point[2];

rs2::device device;
vector<rs2::sensor> color_sensor;
#define BUFSIZE 1024;

float predicted;
int port_num_to_Recognizer = 8888;
int port_num_to_Arm = 10048;
int port_num_from_Recognizer = 3001;

float speed_of_arm = 0.5;

std::string local_ip = "127.0.0.1";
std::string arm_ip = "192.168.0.12";
///////////////////////////////////////////////////////////////
float now;
float arm_yvalue;
float arm_xvalue;
float arm_zvalue;

int distance_thresh = 5;
int move_count1 = 0;
int total_angle = 0;
float first[3];
float second[3];
int firsttime = 0;
int secondtime = 0;

bool trueorfalse = false;
std::chrono::system_clock::time_point StartTime;
int count_flag = 0;
std::vector<float> arrx; // linear actuator를 이동시키기 위한 벡터
std::vector<float> arry;
float Sleep_time = 0;
float total_time = 0;
//////////////////////////////////////////////////////fitting 변수
const int vecsize = 5;
const int vecsize1 = 250;
int funcdegree = 1;
float timeint;
float fittime1;
std::vector<float> finalxA;
std::vector<float> finaltA;
float finalxsum;
float finaltsum;
float finalx;
float finalt;
float finalz;
float endx;
float endt;
float t11;
float t22;
int fixed_flag = 0;
std::vector<float> Bz(vecsize);
std::vector<float> realtime(vecsize);
std::vector<float> fittime(vecsize1);
std::vector<float> result(funcdegree);
std::vector<float> resultz(vecsize1);
int largest_area = 0;
int largest_contour_index = 0;
Scalar color(255, 255, 255);

//////////////////////////////////////////글로벌 설정////////


void move_ac(float position)
{
	LM_G->move_actu(position);
	//cout << "real position:"<< position << endl;
}

void move_ac2(float position, float fitting_position)
{
	//Sleep(time * 1000);
	float final_position;
	final_position = position + 100 * fitting_position;
	if (final_position > 55)
	{
		final_position = 55;
	}
	else if (final_position < -55)
	{
		final_position = -55;
	}
	move_ac(final_position);
	//cout << "real position:"<< position << endl;
}

void initialize()
{
	iLastX = -1;
	iLastY = -1;
	posX = -1;
	posY = -1;

	imghsv = new Mat();
	imghsv_ = new Mat();
	imgthresholded = new Mat();
	imgthresholded_ = new Mat();
	imgLines = new Mat();
	
	image_ = new Mat();
	Depth_image = new Mat();
	*Depth_image = Mat(Size(width, height), CV_8UC3);

	devices = ctx.query_devices();
	device = devices.front();

	LM_G = new CLinear_actu();


	rs2::device device = devices.front();
	auto color_sensor = device.query_sensors();
	//color_sensor.at(1).set_option(RS2_OPTION_BRIGHTNESS, 70);
	color_sensor.at(1).set_option(RS2_OPTION_GAIN, 30);
	//color_sensor.at(1).set_option(RS2_OPTION_EXPOSURE, 25);

	//color_sensor.at(0).set_option(RS2_OPTION_EXPOSURE, 3350);
	//if (color_sensor.at(0).supports(RS2_OPTION_EMITTER_ENABLED))
	//{
	//	color_sensor.at(0).set_option(RS2_OPTION_EMITTER_ENABLED, 1.f); // Enable emitter
	////	//color_sensor.at(0).set_option(RS2_OPTION_EMITTER_ENABLED, 0.f); // Disable emitter
	//}
	//if (color_sensor.at(0).supports(RS2_OPTION_LASER_POWER))
	//{
	//	auto range = color_sensor.at(0).get_option_range(RS2_OPTION_LASER_POWER);
	//	color_sensor.at(0).set_option(RS2_OPTION_LASER_POWER, range.max); // Set max power
	////	//color_sensor.at(0).set_option(RS2_OPTION_LASER_POWER, 0.f); // Disable laser
	//}

	//color_sensor.at(0).set_option(RS2_OPTION_DEPTH_UNITS, 0.0001); // Disable laser

	cfg.enable_device(devices[0].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
	cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
	cfg.enable_stream(RS2_STREAM_INFRARED, width, height, RS2_FORMAT_Y8, fps);
	cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);

	selection = pipe.start(cfg);
}


Point3f get_3D_coor(Point coor, const rs2::points& points, const rs2::video_frame& color)
{
	auto Vertex = points.get_vertices();
	int width = color.get_width();

	return Point3f(Vertex[coor.x + (coor.y * width)].x, Vertex[coor.x + (coor.y * width)].y, Vertex[coor.x + (coor.y * width)].z);
}

void move_manipulator(float send_x, float send_y, float send_z, float send_vx, float send_vy, float send_vz, float send_t) //원래 manipulator 움직이는 함수
{
	engine* pEngine;
	pEngine = engOpen("null"); // opens matlab engine
	float Ovx = 1;
	float Ovy = 0;
	float Ovz = 0;

	float Ix = send_x;
	float Iy = send_y;
	float Iz = send_z;
	float Ivx = send_vx;
	float Ivy = send_vy;
	float Ivz = send_vz;
	float delay_t = send_t;
	//////////////////////////////////////////////////////////////////////////
	float lv[3];
	lv[0] = Ivx; lv[1] = Ivy; lv[2] = Ivz;  // input position& velocity
	float Ov[3];
	Ov[0] = Ovx; Ov[1] = Ovy; Ov[2] = Ovz; // output position& velocity
	float N[3];
	N[0] = -lv[0] + Ov[0];
	N[1] = -lv[1] + Ov[1];
	N[2] = -lv[2] + Ov[2]; // 탁구채의 Normal vector
	float norm = sqrt(pow(N[0], 2) + pow(N[1], 2) + pow(N[2], 2));
	float rad = 0.015 * acos(N[0] / norm); //gripper angle
	float fake_rad = 0;
	float send = acos(N[0] / norm); //send angle
	//////////////////////////////////////////////
	//cout << rad << endl;
	float limit;
	float Sx; float Sy; float Sz; float Ex; float Ey; float Ez;
	float SSx; float SSy;
	float hitxposition;
	float hityposition;
	float Mx;
	float My;
	float Mz;
	//if (Iz > 0.23)
	//{
	//	Iz = 0.23;
	//}
	if (Iz > 0.235)
	{
		hitxposition = (0.126 + 0.024 + sqrt(pow(0.124, 2) - pow((Iz - 0.235), 2))) * sin(send);
		hityposition = -(0.126 + 0.024 + sqrt(pow(0.124, 2) - pow((Iz - 0.235), 2))) * cos(send);
		if (Iy > 0)
		{
			Mx = (0.126 + 0.024 + sqrt(pow(0.124, 2) - pow((Iz - 0.235), 2))) * sin(send + 0.785);
			My = -(0.126 + 0.024 + sqrt(pow(0.124, 2) - pow((Iz - 0.235), 2))) * cos(send + 0.785);
			Mz = (0.194 + Iz) / 2;
			Sx = (0.126 + 0.024 + sqrt(pow(0.124, 2) - pow((Iz - 0.235), 2))) * sin(send - 0.523);
			Sy = -(0.126 + 0.024 + sqrt(pow(0.124, 2) - pow((Iz - 0.235), 2))) * cos(send + 0.523);
			Sz = Iz; // swing start point
			Ex = (0.126 + 0.024 + sqrt(pow(0.124, 2) - pow((Iz - 0.235), 2))) * sin(send + 0.785);
			Ey = -(0.126 + 0.024 + sqrt(pow(0.124, 2) - pow((Iz - 0.235), 2))) * cos(send - 0.785);
			Ez = Iz; // swing start point
		}
		else
		{
			Mx = (0.126 + 0.024 + sqrt(pow(0.124, 2) - pow((Iz - 0.235), 2))) * sin(send + 0.785);
			My = (0.126 + 0.024 + sqrt(pow(0.124, 2) - pow((Iz - 0.235), 2))) * cos(send + 0.785);
			Mz = (0.194 + Iz) / 2;
			Sx = (0.126 + 0.024 + sqrt(pow(0.124, 2) - pow((Iz - 0.235), 2))) * sin(send - 0.523);
			Sy = (0.126 + 0.024 + sqrt(pow(0.124, 2) - pow((Iz - 0.235), 2))) * cos(send + 0.523);
			Sz = Iz; // swing start point
			Ex = (0.126 + 0.024 + sqrt(pow(0.124, 2) - pow((Iz - 0.235), 2))) * sin(send + 0.785);
			Ey = (0.126 + 0.024 + sqrt(pow(0.124, 2) - pow((Iz - 0.235), 2))) * cos(send - 0.785);
			Ez = Iz; // swing start point
		}
	}
	else
	{
		hitxposition = 0.274 * sin(send);
		hityposition = -0.274 * cos(send);


		if (Iy > 0)
		{
			Mx = 0.274 * sin(send + 0.785); My = 0.274 * cos(send + 0.785); Mz = (0.194 + Iz) / 2;
			Sx = 0.274 * sin(send - 0.523); Sy = 0.274 * cos(send + 0.523); Sz = Iz;// swing start point
			Ex = 0.274 * sin(send + 0.523); Ey = 0.274 * cos(send - 0.523); Ez = Iz;// swing end point
			//fake_rad = 0.00393/2;
		}
		else
		{
			Mx = 0.274 * sin(send + 0.785); My = -0.274 * cos(send + 0.785); Mz = (0.194 + Iz) / 2;
			Sx = 0.274 * sin(send - 0.523); Sy = -0.274 * cos(send + 0.523); Sz = Iz;// swing start point
			Ex = 0.274 * sin(send + 0.523); Ey = -0.274 * cos(send - 0.523); Ez = Iz;// swing end point
			//fake_rad = -0.00393/2;
		}
	}
	if (Iz < -0.1)
	{
		Iz = -0.1;
	}
	//limit

	if (Iz < -0.1)
	{
		limit = 1;
	}
	else
	{
		limit = 0;
	}


	if (limit == 1) // 칠 수 없을때
	{
		cout << "limit" << limit << endl;
	}
	else
	{
		//Sleep(delay_t*1000);
		string p1 = to_string(Mx);
		string p2 = to_string(My);
		string p3 = to_string(Mz);
		string r1 = to_string(fake_rad);
		string n1 = to_string(0.1);
		char com1[200] = "fprintf(s,'";
		const char* po1 = p1.c_str();
		const char* po2 = p2.c_str();
		const char* po3 = p3.c_str();
		const char* ra1 = r1.c_str();
		const char* nu1 = n1.c_str();
		strcat_s(com1, "jointandtool");
		strcat_s(com1, ",");
		strcat_s(com1, po1);
		strcat_s(com1, ",");
		strcat_s(com1, po2);
		strcat_s(com1, ",");
		strcat_s(com1, po3);
		strcat_s(com1, ",");
		strcat_s(com1, ra1);
		strcat_s(com1, ",");
		strcat_s(com1, nu1);
		strcat_s(com1, "');");
		engEvalString(pEngine, com1);
		engEvalString(pEngine, "pause(0.13);");

		string p4 = to_string(Sx);
		string p5 = to_string(Sy);
		string p6 = to_string(Sz);
		string r2 = to_string(fake_rad);
		string n2 = to_string(0.1);
		char com2[200] = "fprintf(s,'";
		const char* po4 = p4.c_str();
		const char* po5 = p5.c_str();
		const char* po6 = p6.c_str();
		const char* ra2 = r2.c_str();
		const char* nu2 = n2.c_str();
		strcat_s(com2, "jointandtool");
		strcat_s(com2, ",");
		strcat_s(com2, po4);
		strcat_s(com2, ",");
		strcat_s(com2, po5);
		strcat_s(com2, ",");
		strcat_s(com2, po6);
		strcat_s(com2, ",");
		strcat_s(com2, ra2);
		strcat_s(com2, ",");
		strcat_s(com2, nu2);
		strcat_s(com2, "');");
		engEvalString(pEngine, com2);
		engEvalString(pEngine, "pause(0.12);");
		//string dt;
		//if (delay_t > 0.12 * 9 / 2) {
		//	dt = to_string(delay_t * 2 / 9);
		//}
		//else
		//{
		//	dt = to_string(0.12);
		//}
		//char delay[50] = "pause(";
		//const char* det = dt.c_str();
		//strcat_s(delay, det);
		//strcat_s(delay, ");");
		//engEvalString(pEngine, delay); //delay_t

		string p7 = to_string(Ex);
		string p8 = to_string(Ey);
		string p9 = to_string(Ez);
		string r3 = to_string(fake_rad);
		string n3 = to_string(0.1);
		char com3[200] = "fprintf(s,'";
		const char* po7 = p7.c_str();
		const char* po8 = p8.c_str();
		const char* po9 = p9.c_str();
		const char* ra3 = r3.c_str();
		const char* nu3 = n3.c_str();
		strcat_s(com3, "jointandtool");
		strcat_s(com3, ",");
		strcat_s(com3, po7);
		strcat_s(com3, ",");
		strcat_s(com3, po8);
		strcat_s(com3, ",");
		strcat_s(com3, po9);
		strcat_s(com3, ",");
		strcat_s(com3, ra3);
		strcat_s(com3, ",");
		strcat_s(com3, nu3);
		strcat_s(com3, "');");
		engEvalString(pEngine, com3);
		engEvalString(pEngine, "pause(0.3);"); //0.3

		engEvalString(pEngine, "fprintf(s,'joint,0,-0.885,0.785,0');");
	}
}

float move_linear2(float first_0, float first_1, float angle) // 이 부분이 linear guide 움직이게 하는 부분
{

	// 두점을 잡고 그 기울기를 계산

	float mola0 = (angle * (0 - first_0) + (first_1)); //  mola0 는 직선식이예요

	mola1 = (-1) * (0 + mola0); //mola1은 이제 linear guide가 닿는 곳입니다. 100곱한 거는 스케일 맞춰 준 거예요 cm 로


	return (100 * mola1);
	//didk
}


double slope(std::vector<float>& x, std::vector<float>& y) {
	const auto n = x.size();
	const auto s_x = std::accumulate(x.begin(), x.end(), 0.0);
	const auto s_y = std::accumulate(y.begin(), y.end(), 0.0);
	const auto s_xx = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
	const auto s_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
	const auto a = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
	return a;
}
Point rotation(float angle, Point point) {

	float s = sin(angle * PI / 180); // angle is in radians
	float c = cos(angle * PI / 180); // angle is in radians
	float xnew = point.x * c - point.y * s;
	float ynew = point.x * s + point.y * c;
	return point;
}
double mean(std::vector<float>& z) {
	const auto n = z.size();
	const auto s_x = std::accumulate(z.begin(), z.end(), 0.0);
	const auto ans = s_x / n;
	return ans;
}

void Real_initialize()
{
	///////////////////////////////////////////////
	engine* pEngine;
	pEngine = engOpen("null"); // opens matlab engine
	/////////////////////////////////////////////////
	int frame_number = 0;
	////////////////////////////////////////////
	rs2::align align_to_depth(RS2_STREAM_COLOR);
	// csv 파일 설정
	initialize();
	int data_num = 1;

	//////////////////////////////////
	int framecount = 0; // 처음 프레임을 잡아줌
	//Serial* TP = new Serial("\\\\.\\COM3");
	//Serial_ptr = new CSerial_ctr("COM3");
	//moveman moveman;
	float average_vel = 0;

	float* linear_z;
	float distance = 0; // 처음 찍힌 좌표와 나중에 찍히는 좌표들 사이의 거리
	int timescount = 0;
	float move_lineary = 0;
	float mean_z;

	///////////////////////////////////////


	////////////////////////////////////////////
	double* positionandt1;
	positionandt1 = (double*)calloc(10, sizeof(double));
	double* positionandt2;
	positionandt2 = (double*)calloc(10, sizeof(double));
	double* tp;
	tp = (double*)calloc(10, sizeof(double));
	///////////////////////////////////////////////////
	int time = 0;
	rs2::frameset data = pipe.wait_for_frames();
	data = align_to_depth.process(data);

	rs2::frame color = data.get_color_frame();
	rs2::frame depth = data.get_depth_frame();

	/////////////////////////////////////////////////////////////
	namedWindow("Control1", CV_WINDOW_AUTOSIZE);
	cvCreateTrackbar("LowH", "Control1", &ilowh, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control1", &ihighh, 179);

	cvCreateTrackbar("LowS", "Control1", &ilows, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control1", &ihighs, 255);

	cvCreateTrackbar("LowV", "Control1", &ilowv, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Control1", &ihighv, 255);

	namedWindow("Control2", CV_WINDOW_AUTOSIZE);
	cvCreateTrackbar("LowH", "Control2", &thresh1, 300); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control2", &thresh2, 300);
	/////////////////////////////////////////////////////////////////////////
	engEvalString(pEngine, "clc; clear all;close all;");
	engEvalString(pEngine, "delete(instrfindall);");
	engEvalString(pEngine, "s = serial('COM3');");
	engEvalString(pEngine, "set(s,'BaudRate',100000);");
	engEvalString(pEngine, "fopen(s);");
	engEvalString(pEngine, "fprintf(s,'joint,0,0,0,0');");
	engEvalString(pEngine, "pause(1);");
	engEvalString(pEngine, "fprintf(s,'joint,0,-0.885,0.785,0');");
	//////////////////////////////////////////////////////////////////////
}


void streaming()
{

	rs2::align align_to_depth(RS2_STREAM_COLOR);

	///////////////////////////////////////////////
	float angle = 0;
	/////////////////////////////////////////////////

	second[0] = first[0];
	second[1] = first[1];
	second[2] = first[2];
	secondtime = firsttime;


	rs2::frameset data = pipe.wait_for_frames();
	data = align_to_depth.process(data);
	//*Prev = imgthresholded->clone(); // 여기가 문제임
	rs2::frame color = data.get_color_frame();
	rs2::frame depth = data.get_depth_frame();

	const int w = color.as<rs2::video_frame>().get_width();
	const int h = color.as<rs2::video_frame>().get_height();

	image = Mat(Size(w, h), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);
	Rect rect(40, 5, 807, 464);
	image = image(rect);
	*imgthresholded = Mat::zeros(Size(w, h), CV_8UC3);
	cvtColor(image, *imghsv, COLOR_BGR2HSV);
	inRange(*imghsv, Scalar(ilowh, ilows, ilowv), Scalar(ihighh, ihighs, ihighv), *imgthresholded);
	erode(*imgthresholded, *imgthresholded, getStructuringElement(MORPH_ELLIPSE, Size(3, 3))); //침식연산 (낮은-어두운) : 작은노이즈 제거
	dilate(*imgthresholded, *imgthresholded, getStructuringElement(MORPH_ELLIPSE, Size(3, 3))); //팽창연산 (높은-밝은) : 큰객체로 결합

	dilate(*imgthresholded, *imgthresholded, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	erode(*imgthresholded, *imgthresholded, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

	/*Canny(*imgthresholded, *imgthresholded, thresh1, thresh2);

	findContours(*imgthresholded, contours1, hierarchy1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));*/

	//////////////////////////////////////////////////////////////
	//Point centroid;
	//for (size_t i = 0; i < contours1.size(); i++) {
	//	Moments m = moments(contours1.at(i));
	//	float area = contourArea(contours1.at(i));

	//	centroid.x = (float(m.m10) / float(m.m00));
	//	centroid.y = (float(m.m10) / float(m.m00));
	//	std::chrono::system_clock::time_point EndTime = std::chrono::system_clock::now();
	//	std::chrono::milliseconds mili = std::chrono::duration_cast<std::chrono::milliseconds>(EndTime - StartTime);
	//	if (area > 5)
	//	{
	//		if (area > largest_area) {
	//			largest_area = area;
	//			//cout << i << " area  " << area << endl;
	//			// Store the index of largest contour
	//			largest_contour_index = i;
	//			if (centroid.x > 0 || centroid.y > 0)
	//			{
	//				first[0] = centroid.x;
	//				first[1] = centroid.y;
	//				firsttime = mili.count();
	//				cout << "x:" << centroid.x << "\t\t" << "y:" << centroid.y << "\t\t" << "time:" << firsttime - secondtime << endl;
	//			}
	//		}

	//		// sometimes the results are impossible values, so we ignore them
	//		if (centroid.x < 0 || centroid.x >= imgthresholded->cols || centroid.y < 0 || centroid.y > imgthresholded->rows)
	//			continue;


	//		// we correct the center position (we computed the position in the ROI,
	//		// we want the position in the full frame)
	//		/*centroid.x += roi_rect.x;
	//		centroid.y += roi_rect.y;

	//		positions.push_back(centroid);*/
	//	}
	//}

	Moments oMoments = moments(*imgthresholded);

	double dM01 = oMoments.m01;
	double dM10 = oMoments.m10;
	double dArea = oMoments.m00;
	//cout << dArea << endl;
	std::string XYZ_Data;

	if ((dArea > 5000) && (dArea < 150000)) // 잡히는 area의 크기 필터
	{
		posX = dM10 / dArea;
		posY = dM01 / dArea;



		if (depth)
		{
			trueorfalse = true;
			pc.map_to(color);
			points = pc.calculate(depth);
			data_point = get_3D_coor(Point(posX, posY), points, color);
			/////////////////////////////////////////////////////////////

			//////////////////////////////////////////////////////////////////
			/*data_point.x = -(data_point.x - 1.16937);
			data_point.x = data_point.x + 0.065;
			data_point.y = data_point.y - 0.017704;
			data_point.z = -(data_point.z - 1.70);
			data_point.z = data_point.z - (0.065 / 1.23) * data_point.y + 0.034;
			data_point.z = data_point.z - data_point.x * (0.06 / 2);
			data_point.z = data_point.z + (0.04 / 1.23) * data_point.y;*/
			////////////////////Calibration 부분/////////////////////////

			//시간 기록
			//예는 밀리세컨드 단위로 시간을 측정

			std::chrono::system_clock::time_point EndTime = std::chrono::system_clock::now();
			std::chrono::milliseconds mili = std::chrono::duration_cast<std::chrono::milliseconds>(EndTime - StartTime);

			if ((data_point.x != 0) || (data_point.y != 0) || (data_point.z != 0))
			{
				first[0] = data_point.x;
				first[1] = data_point.y;
				first[2] = data_point.z;
				firsttime = mili.count();
			}
			/////////////////////////////////////////////////////////////
			first[0] = first[0]+1.149;
			first[1] =-(first[1]+0.015);
			first[2] = first[2];
		}
	}
	
 // (frame_number <= 1000);

}

void streaming2()
{
	//////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////

	//Vx(delta x) 기준으로 방향을 알 수 있다 //요거 만들던 중이었음
	if (trueorfalse)
	{	

		std::cout << "temp : " << first[0] << "\t\t" << first[1] << endl;

		if (first[0] > 0) {
			Bz[move_count1] = 2.4 - first[0];
		}

		if ((first[0] - second[0] < 0))
		{
			direction = 1;
		}
		else
		{
			direction = -1;
		}

		if (count_flag == 0 && direction == 1)
		{
			if (first[0] > 0) {
				//std::cout << "temp : " << first[0] << "\t\t" << first[1] << "time:" << "\t\t" << firsttime - secondtime << std::endl;
				move_count1++;
				arrx.push_back(first[0]);
				arry.push_back(first[1]);
			}
		}

		cout << "firsttime-secondtime" << firsttime-secondtime << endl;
		//cout << "secondtime" << secondtime << endl;

		if (firsttime - secondtime > 750)
		{
			move_count1 = 0;// 종료 and 다음공

			arrx.clear();
			arry.clear();
			if (first[0] > 0)
			{
				//std::cout << "temp : " << first[0] << "\t\t" << first[1] << "time:" << "\t\t" << firsttime - secondtime << std::endl;
				arrx.push_back(first[0]);
				arry.push_back(first[1]);
			}
			//cout << point.x << endl;
			count_flag = 0; //move_count를 더해라
		}


		if (move_count1 == 5 && count_flag == 0)
		{
			linear_y = move_linear2(first[0], first[1], slope(arrx, arry)); //실제 예상 좌표 값
			fixed_flag = 1;
			count_flag = 1; // 더 이상 move_count를 더하지 않음
			finalxsum = 0;
			finaltsum = 0;
			if (first[0] > 0)
			{
				float timeint = (firsttime - secondtime);
				//cout << "realtime = " << timeint << endl;

				for (int i = 1; i < move_count1; i++)
				{
					realtime[0] = 0;
					float timeint = (firsttime - secondtime);
					//cout << "realtime = "<<timeint << endl;
					realtime[i] = realtime[i - 1] + timeint;

				}

				for (int i = 0; i < move_count1; i++)
				{
					realtime[i] = realtime[i] / 1000;
				}

				result = polyfit(realtime, Bz, funcdegree);
				//cout <<"result0" << result[0]<< "result1" <<result[1] << "result2" << result[2] << "result3" << result[3] << "result4" << result[4] << "result5" << result[5] << endl;
				//cout << realtime[9] << endl;
				//cout << move_count2 << endl;

				for (int i = 0; i < vecsize1; i++)
				{
					fittime1 = i * 0.004;
					fittime[i] = fittime1;
				}

				resultz = polyval(result, fittime);


				for (int i = 0; i < vecsize1; i++)
				{

					float cond = resultz[0] + 2.74;
					if (resultz[i] >= cond && resultz[i] < (cond + 0.05))
					{

						finalxA.push_back(resultz[i]);
						finaltA.push_back(fittime[i] * 1.18071429);

						//endx = resultz[i];
						//endt = fittime[i]*1.18071429;
					}
				}
				int resultnum = finalxA.size();
				int resultnum2 = finaltA.size();

				for (int j = 0; j < resultnum; j++)
				{
					finalxsum = finalxsum + finalxA[j];
					finaltsum = finaltsum + finaltA[j];
				}
				finalx = finalxsum / resultnum;
				finalt = finaltsum / resultnum2;

				finalxA.clear();
				finaltA.clear();
				//finalx = endx;
				//finalt = endt;
				cout << "finalx = " << finalx << endl;
				cout << "finalt = " << finalt << endl;

				if (finalt <= 0.35)
				{
					t11 = 0.75 * finalt;
					t22 = 0.25 * finalt;
					finalz = (0.8 * 9.8 * t11 * t22 - 0.5 * 9.8 * t22 * t22) * 0.85 + 0.21;
				}
				if (0.35 < finalt && finalt <= 0.5)
				{
					t11 = 0.6 * finalt;
					t22 = 0.4 * finalt;
					finalz = (0.8 * 9.8 * t11 * t22 - 0.5 * 9.8 * t22 * t22) * 0.85 + 0.16;
				}
				if (0.5 < finalt && finalt <= 0.65)
				{
					t11 = 5 * finalt / 9;
					t22 = 4 * finalt / 9;
					finalz = (0.8 * 9.8 * t11 * t22 - 0.5 * 9.8 * t22 * t22) * 0.85 + 0.085;
				}
				if (0.65 < finalt && finalt < 0.75)
				{
					t11 = 0.5 * finalt;
					t22 = 0.5 * finalt;
					finalz = (0.8 * 9.8 * t11 * t22 - 0.5 * 9.8 * t22 * t22) * 0.85 - 0.01;
				}
				if (0.75 < finalt)
				{
					t11 = 0.5 * finalt;
					t22 = 0.5 * finalt;
					finalz = (0.8 * 9.8 * t11 * t22 - 0.5 * 9.8 * t22 * t22) * 0.85 - 0.09;
					finalt = 1.2;
				}
				if (finalz > 0.35)
				{
					finalz = 0.35*0.35/finalz;
				}
				cout << "finalz = " << finalz << endl;

			}
			cout << "linear_y:" << linear_y << endl;
			//cout << "fixed_flag:" << fixed_flag << endl;
			if (fixed_flag == 1) {

				if (linear_y > 55)
				{
					linear_y0 = -0.23; //-0.274
					cout << "estimated position:" << linear_y << endl;
					std::thread thread1(std::bind(&move_manipulator, 0, linear_y0 + 0.02, finalz - 0.12, -1, 0, 0, finalt));
					std::thread thread2(std::bind(&move_ac2, linear_y, linear_y0));
					////cout << linear_y << endl;
					thread1.join();
					thread2.join();
					move_ac(0);
				}
				else if (55 > linear_y && linear_y > 35)
				{
					linear_y0 = -0.27; //-0.274
					cout << "estimated position:" << linear_y << endl;
					std::thread thread1(std::bind(&move_manipulator, 0, linear_y0 + 0.06, finalz - 0.12, -1, 0, 0, finalt));
					std::thread thread2(std::bind(&move_ac2, linear_y, linear_y0));
					////cout << linear_y << endl;
					thread1.join();
					thread2.join();
					move_ac(0);
				}
				else if (linear_y < 35 && linear_y > 20)
				{
					linear_y0 = -0.26; //-0.274
					cout << "estimated position:" << linear_y << endl;
					std::thread thread1(std::bind(&move_manipulator, 0, linear_y0 + 0.02, finalz - 0.12, -1, 0, 0, finalt));
					std::thread thread2(std::bind(&move_ac2, linear_y, linear_y0));
					////cout << linear_y << endl;
					thread1.join();
					thread2.join();
					move_ac(0);
				}
				else if (linear_y > 0 && linear_y < 20)
				{
					linear_y0 = -0.28; //-0.274
					cout << "estimated position:" << linear_y << endl;
					std::thread thread1(std::bind(&move_manipulator, 0, linear_y0 + 0.06, finalz - 0.12, -1, 0, 0, finalt));
					std::thread thread2(std::bind(&move_ac2, linear_y, linear_y0));
					////cout << linear_y << endl;
					thread1.join();
					thread2.join();
					move_ac(0);
				}
				else if (linear_y < 0 && linear_y > -20)
				{
					linear_y0 = 0.28; //-0.274
					cout << "estimated position:" << linear_y << endl;
					std::thread thread1(std::bind(&move_manipulator, 0, linear_y0 - 0.06, finalz - 0.12, -1, 0, 0, finalt));
					std::thread thread2(std::bind(&move_ac2, linear_y, linear_y0));
					////cout << linear_y << endl;
					thread1.join();
					thread2.join();
					move_ac(0);
				}
				else if (linear_y < -20 && linear_y> -35)
				{
					linear_y0 = 0.26; // 0.274
					cout << "estimated position:" << linear_y << endl;
					std::thread thread1(std::bind(&move_manipulator, 0, linear_y0 - 0.02, finalz - 0.12, -1, 0, 0, finalt));
					std::thread thread2(std::bind(&move_ac2, linear_y, linear_y0));
					////cout << linear_y << endl;
					thread1.join();
					thread2.join();
					move_ac(0);
				}
				else if (-55 < linear_y && linear_y < -35)
				{
					linear_y0 = 0.27; // 0.274
					cout << "estimated position:" << linear_y << endl;
					std::thread thread1(std::bind(&move_manipulator, 0, linear_y0 - 0.06, finalz - 0.12, -1, 0, 0, finalt));
					std::thread thread2(std::bind(&move_ac2, linear_y, linear_y0));
					////cout << linear_y << endl;
					thread1.join();
					thread2.join();
					move_ac(0);
				}
				else if (linear_y < -55)
				{
					linear_y0 = 0.23; // 0.274
					cout << "estimated position:" << linear_y << endl;
					std::thread thread1(std::bind(&move_manipulator, 0, linear_y0 - 0.02, finalz - 0.12, -1, 0, 0, finalt));
					std::thread thread2(std::bind(&move_ac2, linear_y, linear_y0));
					////cout << linear_y << endl;
					thread1.join();
					thread2.join();
					move_ac(0);
				}
				else;
				//}
			}
			arrx.clear();
			arry.clear();
			fixed_flag = 0;
		}
	}
}




int main()
{
	first[0] = 0;
	first[1] = 0;
	first[2] = 0;
	Real_initialize();
	StartTime = std::chrono::system_clock::now();
	while (true) {
		thread th1(streaming);
		th1.join();
		//imshow("realsense_data_threshold", *imgthresholded);
		//imshow("realsense_data", image);
		//int c = cvWaitKey(1);
		//cout << first[2] << endl;
		//cout << firsttime << endl;
		if (firsttime != secondtime)
		{
			thread th2(streaming2);
			th2.join();
			/*streaming2();*/
		}
	}
}