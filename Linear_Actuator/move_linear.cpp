//#pragma once
//#include <iostream>
//#include <vector>
//#include <numeric>
//#include <algorithm>
//#include <Windows.h>
//#include "CLinear_actu.h"
//#include "move_linear.h"
//
//using namespace std;
//
//float first_0;
//float first_1;
//std::vector<float>& x_;
//std::vector<float>& y_;
//CLinear_actu* LM_G;
//double slope(std::vector<float>& x, std::vector<float>& y) {
//	const auto n = x.size();
//	const auto s_x = std::accumulate(x.begin(), x.end(), 0.0);
//	const auto s_y = std::accumulate(y.begin(), y.end(), 0.0);
//	const auto s_xx = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
//	const auto s_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
//	const auto a = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
//	return a;
//	}
//void move_ac(int position)
//{
//	LM_G->move_actu(position);
//}
//void move_linear3(float first0, float first1, float angle) // �� �κ��� linear guide �����̰� �ϴ� �κ�
//{
//
//	// ������ ��� �� ���⸦ ���
//
//	float mola0 = (angle * (0 - first0) + first1); //  mola0 �� �������̿���
//
//	float mola1 = (-1) * (0 + mola0); //mola1�� ���� linear guide�� ��� ���Դϴ�. 100���� �Ŵ� ������ ���� �� �ſ��� cm ��
//	float linear_y = 0;
//	if (mola1 > 0)
//	{
//		// �������� 0.274 ��ŭ ������ ���ܳ������� ����
//		linear_y = mola1 + 0.274;
//	}
//	else if (mola1 < 0)
//	{
//		linear_y = mola1 - 0.274;
//	}
//	if (100 * mola1 > 55)
//	{
//		move_ac(55);
//	}
//	else if (100 * mola1 < -55)
//	{
//		move_ac(-55);
//	}
//	else
//	{
//		move_ac(100 * mola1);
//	}
//		cout << 100 * mola1 << endl;
//		//move_ac(0);
//
//	}
//void move_linear_class::move_linear_real(float first_0, float first_1, std::vector<float>& x_, std::vector<float>& y_, int move_count, int once2) {
//	if (move_count == 3)
//	{
//		move_linear2(first_0, first_1, slope(x_, y_));
//	}
//
//
//	if ((move_count == 12) && (once2 == 0))
//	{
//		move_linear3(first_0, first_1, slope(x_, y_));
//		// ���缭 ĥ �� �ֵ���
//		x_.clear();
//		y_.clear();
//		once2 = 1;
//		move_count = 0;
//	}
//	Sleep(250);
//	move_ac(0);
//}
//
//	void move_linear2(float first0, float first1, float angle) // �� �κ��� linear guide �����̰� �ϴ� �κ�
//	{
//
//		// ������ ��� �� ���⸦ ���
//
//		float mola0 = (angle * (0 - first0) + first1); //  mola0 �� �������̿���
//
//		float mola1 = (-1) * (0 + mola0); //mola1�� ���� linear guide�� ��� ���Դϴ�. 100���� �Ŵ� ������ ���� �� �ſ��� cm ��
//		float linear_y = 0;
//		if (mola1 > 0)
//		{
//			// �������� 0.274 ��ŭ ������ ���ܳ������� ����
//			linear_y = mola1 + 0.274;
//		}
//		else if (mola1 < 0)
//		{
//			linear_y = mola1 - 0.274;
//		}
//		if (100 * mola1 > 55)
//		{
//			move_ac(55);
//		}
//		else if (100 * mola1 < -55)
//		{
//			move_ac(-55);
//		}
//		else
//		{
//			move_ac(100 * mola1);
//		}
//		cout << 100 * mola1 << endl;
//		//move_ac(0);
//
//	}