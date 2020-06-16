#pragma once
// Mod_CMatrix.h: interface for the Mod_CMatrix class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_MOD_CMATRIX_H__87515FD3_2DCC_473A_B256_28F864450414__INCLUDED_)
#define AFX_MOD_CMATRIX_H__87515FD3_2DCC_473A_B256_28F864450414__INCLUDED_

//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include <math.h>
#include <vector>

//�������
#include <opencv/cv.h>
#include <opencv/highgui.h>

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

//////////////////////////////////////////////////////////////////////////
#define Exist_Matrix	1

//using namespace std;

typedef struct Mod_CVector
{
	double X;
	double Y;
	double Z;

	void init() { X = 0;	Y = 0;	Z = 0; }
	~Mod_CVector() {}
	Mod_CVector operator +(Mod_CVector inVec)
	{
		Mod_CVector outVec;
		outVec.X = X + inVec.X;		outVec.Y = Y + inVec.Y;		outVec.Z = Z + inVec.Z;
		return	outVec;
	}

	Mod_CVector operator -(Mod_CVector inVec)
	{
		Mod_CVector outVec;
		outVec.X = X - inVec.X;		outVec.Y = Y - inVec.Y;		outVec.Z = Z - inVec.Z;
		return	outVec;
	}

	Mod_CVector operator *(double inValue)
	{
		Mod_CVector outVec;
		outVec.X = X * inValue;		outVec.Y = Y * inValue;		outVec.Z = Z * inValue;
		return	outVec;
	}

	Mod_CVector operator /(double inValue)
	{
		Mod_CVector outVec;
		if (inValue != 0)
		{
			outVec.X = X / inValue;			outVec.Y = Y / inValue;			outVec.Z = Z / inValue;
		}

		else
		{
			outVec.X = X;			outVec.Y = Y;			outVec.Z = Z;
		}
		return outVec;
	}

}Mod_CVector;

typedef struct Mod_C2DVector
{
	double X;
	double Y;
	void init() { X = 0;	Y = 0; }
	~Mod_C2DVector() {}
	Mod_CVector operator +(Mod_CVector inVec)
	{
		Mod_CVector outVec;
		outVec.X = X + inVec.X;		outVec.Y = Y + inVec.Y;
		return	outVec;
	}

	Mod_CVector operator -(Mod_CVector inVec)
	{
		Mod_CVector outVec;
		outVec.X = X - inVec.X;		outVec.Y = Y - inVec.Y;
		return	outVec;
	}

	Mod_CVector operator *(double inValue)
	{
		Mod_CVector outVec;
		outVec.X = X * inValue;		outVec.Y = Y * inValue;
		return	outVec;
	}

	Mod_CVector operator /(double inValue)
	{
		Mod_CVector outVec;
		if (inValue != 0)
		{
			outVec.X = X / inValue;			outVec.Y = Y / inValue;
		}

		else
		{
			outVec.X = X;			outVec.Y = Y;
		}
		return outVec;
	}
}Mod_C2DVector;


typedef struct Mod_S3DCoord
{
	double x;
	double y;
	double z;

	void init() { x = 0; y = 0; z = 0; }
	~Mod_S3DCoord() {}
	Mod_S3DCoord operator +(Mod_S3DCoord inPoint)
	{
		Mod_S3DCoord outPoint;
		outPoint.x = x + inPoint.x;		outPoint.y = y + inPoint.y;		outPoint.z = z + inPoint.z;
		return	outPoint;
	}

	Mod_S3DCoord operator -(Mod_S3DCoord inPoint)
	{
		Mod_S3DCoord outPoint;
		outPoint.x = x - inPoint.x;		outPoint.y = y - inPoint.y;		outPoint.z = z - inPoint.z;
		return	outPoint;
	}

	Mod_S3DCoord operator *(double inValue)
	{
		Mod_S3DCoord outPoint;
		outPoint.x = x * inValue;		outPoint.y = y * inValue;		outPoint.z = z * inValue;
		return	outPoint;
	}

	Mod_S3DCoord operator /(double inValue)
	{
		Mod_S3DCoord outPoint;
		if (inValue != 0)
		{
			outPoint.x = x / inValue;			outPoint.y = y / inValue;			outPoint.z = z / inValue;
		}

		else
		{
			outPoint.x = x;			outPoint.y = y;			outPoint.z = z;
		}
		return outPoint;
	}

}Mod_S3DCoord;

typedef struct Mod_SUVCoord
{
	int u; int v;
	~Mod_SUVCoord() {}
}Mod_SUVCoord;


typedef struct Mod_S2DCoord
{
	double x;
	double y;
	~Mod_S2DCoord() {}
}Mod_S2DCoord;


typedef struct Mod_SColor
{
	float r;
	float g;
	float b;
	~Mod_SColor() {}
}Mod_SColor;


typedef struct Mod_S3DPoint
{
	Mod_S3DCoord pcoord;
	Mod_SColor pcolor;
	~Mod_S3DPoint() {}
}Mod_S3DPoint;

typedef struct Mod_UVPoint
{
	Mod_S3DCoord pcoord;
	Mod_SColor pcolor;
	int U;
	int V;
	~Mod_UVPoint(){}
}Mod_UVPoint;


typedef struct Mod_SMeshModel
{
	std::vector <Mod_S3DCoord> Out_cir_Vertex;  //Mod_S3dCoord�� �������� �޸𸮸� ���Ѵ��� ������� �Ҵ�.
	std::vector <Mod_S3DCoord> in_cir_Vertex;
	std::vector <Mod_S3DCoord> Handle_Vertex;
	~Mod_SMeshModel(){}
}Mod_SMeshModel;


typedef struct Mod_SCircle
{
	double a;
	double b;
	double r;
	//	int count_vote; 						// Circle Eq: (x-a)^2 + (y-b)^2 = r^2
	~Mod_SCircle(){}
}Mod_SCircle;


typedef struct Mod_SCircle3D
{
	double a;
	double b;
	double r;
	double r2;
	double h;
	//	int count_vote; 						// Circle Eq: (x-a)^2 + (y-b)^2 = r^2  with height 'h'
	~Mod_SCircle3D(){}
}Mod_SCircle3D;

typedef struct Mod_S2DPoint
{
	Mod_S2DCoord pcoord;
	Mod_SColor pcolor;
	~Mod_S2DPoint(){}
}Mod_S2DPoint;


typedef struct Mod_SEigenv
{
	Mod_CVector e1;
	Mod_CVector e2;
	Mod_CVector e3;
	~Mod_SEigenv(){}
}Mod_SEigenv;

typedef struct Mod_SPlane
{
	Mod_CVector norm;
	Mod_S3DCoord norm_point;
	Mod_S3DCoord plane_point2;
	Mod_S3DCoord plane_point3;
	~Mod_SPlane(){}
}Mod_SPlane;

typedef struct Mod_SVotingVec
{
	Mod_CVector vec;
	int count;
	//	bool flag;
	~Mod_SVotingVec(){}
}Mod_SVotingVec;

typedef struct Mod_Radius
{
	double distace;
	~Mod_Radius(){}
}Mod_Radius;

typedef struct PrimitivePoint
{
	int Num_Primitive;
	std::vector<int> PointIndex;
	~PrimitivePoint()
	{
		if (PointIndex.size() > 0) PointIndex.clear();
	}
}PrimitivePoint;

typedef struct ImageData
{
	int img_width;
	int img_height;
	std::vector<Mod_S3DPoint> rangedata;
	~ImageData()
	{
		if (rangedata.size() > 0) rangedata.clear();
	}
}ImageData;

typedef struct Index
{
	std::vector<int> PointIndex;
	~Index()
	{
		if (PointIndex.size() > 0) PointIndex.clear();
	}
}Index;

typedef struct Connection
{
	std::vector<int> connect_pri;
	~Connection()
	{
		if (connect_pri.size() > 0) connect_pri.clear();
	}
}Connection;

typedef struct Mod_SCylind
{
	std::vector<double> Radius;			//��ü ���� 1mm�� ������
	std::vector<Mod_S3DCoord> WPoints;	//ī�޶� ��ǥ�迡���� Primitive Point
	std::vector<Mod_S3DCoord> CPoints;	//Y��� ȸ������ �����ϰ� ȸ���� Primitive Point
	std::vector<Mod_SColor> PColor;		//Primitive�� Color ��
	Mod_CVector MainAxis;			//ī�޶� ��ǥ�迡���� ȸ���� ������ ����
	Mod_S3DCoord meanWpoint;		//��ü�� �߽���(ī�޶� ��ǥ��)
	Mod_S3DCoord High_CPoint;		//��ü�� �߽ɿ��� �ְ���� Point(��ü ��ǥ��)
	Mod_S3DCoord Low_CPoint;		//��ü�� �߽ɿ��� �ּҳ��� Point(��ü ��ǥ��)
	int num_Primitive;
	~Mod_SCylind()
	{
		if (WPoints.size() >0) WPoints.clear();
		if (CPoints.size() >0) CPoints.clear();
		if (PColor.size() >0) PColor.clear();
		if (Radius.size() >0) Radius.clear();
	}
}Mod_SCylind;


typedef struct Mod_BKPlane
{
	std::vector<Mod_S3DCoord> WPoints;	//ī�޶� ��ǥ�迡���� Primitive Point
	std::vector<int> PointIndex;
	Mod_CVector Normal_Vector;		//ī�޶� ��ǥ�迡���� ��� ��ֺ����� ����
	Mod_S3DCoord meanWpoint;		//��ü�� �߽���(ī�޶� ��ǥ��)
	int Num_Primitive;
	~Mod_BKPlane()
	{
		if (WPoints.size() >0) WPoints.clear();
		if (PointIndex.size() >0) PointIndex.clear();
	}
}Mod_BKPlane;

typedef struct Mod_BKKTable
{
	Mod_S3DCoord point1;
	Mod_S3DCoord point2;
	Mod_S3DCoord point3;
	Mod_S3DCoord point4;
	std::vector<Mod_S3DCoord> T_point;
	~Mod_BKKTable()
	{
		if (T_point.size() >0) T_point.clear();
	}
}Mod_BKKTable;


typedef struct Mod_BKMesh
{
	std::vector<Mod_S3DCoord> WPoints;	//ī�޶� ��ǥ�迡���� Primitive Point
	Mod_CVector Normal_Vector;		//ī�޶� ��ǥ�迡���� ��� ��ֺ����� ����
	Mod_S3DCoord meanWpoint;		//��ü�� �߽���(ī�޶� ��ǥ��)
	int Num_Primitive;
	double A;
	double B;
	double C;
	double D;

	~Mod_BKMesh()
	{
		if (WPoints.size() >0) WPoints.clear();
	}
}Mod_BKMesh;


typedef struct Mod_BK3DLine
{
	Mod_CVector Direction_Vector;		//ī�޶� ��ǥ�迡���� ��� ��ֺ����� ����
	Mod_S3DCoord Point;		//��ü�� �߽���(ī�޶� ��ǥ��)	
	~Mod_BK3DLine() {}
}Mod_BK3DLine;


typedef struct Mod_BKEllipse
{
	double Angle;
	double Width;
	double Height;
	Mod_S2DCoord CenterPoint;
	~Mod_BKEllipse(){}
}Mod_BKEllipse;

//struct SSS
//{
//	int *A;
//	SSS()
//	{
//		A = NULL;
//    }
//	~SSS()
//	{
//		delete [] A;
//	}
//
//};

typedef struct Object_Pose
{
	double pose[9];
	~Object_Pose(){}
}Object_Pose;

typedef struct Object_DB
{
	int kind_primitive1;
	int kind_primitive2;
	double angle;
	double distance;
	~Object_DB(){}
}Object_DB;

typedef struct Categori_DB
{
	int kind_categori1;
	int kind_categori2;
	double angle;
	~Categori_DB(){}
}Categori_DB;

typedef struct Mod_CMilkPack
{
	int Center_Plane;
	int Plane1;
	int Plane2;
	~Mod_CMilkPack(){}
}Mod_CMilkPack;

typedef struct Mod_Panta
{
	int Hexa_Num;
	int Center_Plane;
	int Haxa_Plane;
	int Plane;
	Mod_S2DCoord CHPoint1;
	Mod_S2DCoord CHPoint2;
	Mod_S2DCoord CPPoint1;
	Mod_S2DCoord CPPoint2;
	~Mod_Panta(){}
}Mod_Panta;

typedef struct Mod_PantaPoint
{
	Mod_S3DCoord Point1;
	Mod_S3DCoord Point2;
	Mod_S3DCoord Point3;
	Mod_S3DCoord Point4;
	Mod_S3DCoord Point5;
	Mod_S3DCoord Point6;
	~Mod_PantaPoint(){}
}Mod_PantaPoint;

typedef struct Mod_Cup
{
	int Number;  //�Ǹ��� ��ü�� �� ����
	int Plane;
	int Cylinder;
	~Mod_Cup(){}
}Mod_Cup;

typedef struct Mod_MilkPoint
{
	Mod_S3DCoord Point1;
	Mod_S3DCoord Point2;
	Mod_S3DCoord Point3;
	Mod_S3DCoord Point4;
	Mod_S3DCoord Point5;
	Mod_S3DCoord Point6;
	Mod_S3DCoord Point7;
	Mod_S3DCoord Point8;
	Mod_S3DCoord Point9;
	Mod_S3DCoord Point10;
	~Mod_MilkPoint(){}
}Mod_MilkPoint;

typedef struct Mod_Cube
{
	std::vector<int> include_Pri;
	double pose[9];
	double distance[3];
	Mod_S3DCoord origin_Point;
	~Mod_Cube()
	{
		if (include_Pri.size() >0) include_Pri.clear();
	}
}Mod_Cube;

typedef struct Mod_HexaPoint
{
	Mod_S3DCoord Point1;
	Mod_S3DCoord Point2;
	Mod_S3DCoord Point3;
	Mod_S3DCoord Point4;
	Mod_S3DCoord Point5;
	Mod_S3DCoord Point6;
	Mod_S3DCoord Point7;
	Mod_S3DCoord Point8;
	~Mod_HexaPoint(){}
}Mod_HexaPoint;

typedef struct Mod_Line
{
	double a;
	double b;
	~Mod_Line(){}
}Mod_Line;

typedef struct Mod_S2DLine
{
	double a;
	double b;
	std::vector<Mod_S2DCoord> points;	//ī�޶� ��ǥ�迡���� Primitive Point
	~Mod_S2DLine()
	{
		if (points.size() >0) points.clear();
	}

}Mod_S2DLine;

typedef struct Mod_Para
{
	double a;
	double b;
	double c;
	~Mod_Para(){}
}Mod_Para;

typedef struct Mod_Conic
{
	double A;
	double B;
	double C;
	double D;
	double E;
	double F;
	~Mod_Conic() {}
}Mod_Conic;

typedef struct Mod_Poly
{
	double a;
	double b;
	double c;
	~Mod_Poly(){}
}Mod_Poly;

typedef struct Mod_TrendPara
{
	Mod_Para Before_Para;
	Mod_Para Current_Para;
	Mod_Para After_Para;
	~Mod_TrendPara(){}
}Mod_TrendPara;

typedef struct Mod_TrendLine
{
	Mod_Line Before_Line;
	Mod_Line Current_Line;
	Mod_Line After_Line;
	~Mod_TrendLine(){}
}Mod_TrendLine;

typedef struct Mod_TrendPoly
{
	Mod_Poly Before_Poly;
	Mod_Poly Current_Poly;
	Mod_Poly After_Poly;
	~Mod_TrendPoly(){}
}Mod_TrendPoly;

typedef struct Mod_Vectortrend
{
	Mod_CVector Before_Vector;
	Mod_CVector Current_Vector;
	Mod_CVector After_Vector;
	~Mod_Vectortrend(){}
}Mod_Vectortrend;

typedef struct Recog_Result
{
	std::vector<int> include_Pri;
	std::vector<Mod_S3DCoord> wpoint;
	double pose[9];
	Mod_S3DCoord origin_Point;
	~Recog_Result()
	{
		if (include_Pri.size() >0) include_Pri.clear();
		if (wpoint.size() >0) wpoint.clear();
	}
}Recog_Result;

typedef struct Categori_Result
{
	std::vector<Mod_Cube> cube;
	std::vector<Mod_SCylind> r_s_o;
	double pose[9];
	Mod_S3DCoord origin_Point;
	~Categori_Result()
	{
		if (cube.size() >0) cube.clear();
		if (r_s_o.size() >0) r_s_o.clear();
	}
}Categori_Result;


//////////////////////////////////////////////////////////////////////////
class Mod_CMatrix
{
public:
	Mod_CMatrix();
	virtual ~Mod_CMatrix();

	Mod_CVector InitVec(Mod_CVector inputVec);
	Mod_CVector MakeDirVec_norm(Mod_S3DCoord start, Mod_S3DCoord dest);
	Mod_CVector MakeDirVec(Mod_S3DCoord start, Mod_S3DCoord dest);
	Mod_CVector Cal_MeshNorm(Mod_CVector *norm_Vector, PrimitivePoint mesh);
	double CalDist(Mod_S3DCoord p1, Mod_S3DCoord p2);
	Mod_CVector MakeUnitVec(Mod_CVector inputVector);
	void TranposedMat3x3(double input_mat[9], double out_T_mat[9]);
	double DotProduct(Mod_CVector v1, Mod_CVector v2);
	Mod_CVector CrossProduct(Mod_CVector start, Mod_CVector dest);
	bool Inverse3x3(double input_mat[9], double Inv_Mat[9]);
	double det2x2(double a, double b, double c, double d);
	double det3x3(double a1, double a2, double a3, double b1, double b2, double b3, double c1, double c2, double c3);
	void MutiplyMatrix3x3(double* A, double* B, double* answer); // answer = AB
	void MutiplyMatrix4x4(double* A, double* B, double* answer); // answer = AB


	int CalCovMat_Mean(std::vector<Mod_S3DCoord> input, Mod_S3DCoord &mean, double* out_cov);
	int Cal2DCovMat_Mean(std::vector<Mod_S2DCoord> input, Mod_S2DCoord &mean, double* out_cov);
	void CalSubMeanMatrix(std::vector <Mod_S3DCoord> input, Mod_S3DCoord &mean, Mod_S3DCoord* outmat);
	void Cal2DSubMeanMatrix(std::vector <Mod_S2DCoord> input, Mod_S2DCoord &mean, Mod_S2DCoord* outmat);
	Mod_S3DCoord CalMatrixMean(std::vector <Mod_S3DCoord> input);
	Mod_S2DCoord Cal2DMatrixMean(std::vector <Mod_S2DCoord> input);
	void Display2D(Mod_SColor* Color);
	int BoundaryClassification(bool* Exist_Point, bool* Boundary, int* Classcification);
	//double* SVDwithCov(std::vector<Mod_S3DCoord> input, Mod_S3DCoord &mean, Mod_CVector eigen_Vector[3]);
	int SVDwithCov(std::vector<Mod_S3DCoord> input, Mod_CVector* eigen_Vector, double* eigen_Value);
	int SVDwithCovIn2D(std::vector<Mod_S2DCoord> input, Mod_C2DVector* eigen_Vector, double *eigen_Value);
	//double* SVDwithCov(std::vector<Mod_CVector> input, Mod_CVector eigen_Vector[3]);
	void CalSVD(double* covMat, Mod_CVector* eigen_Vector, double* eigen_Value);

	void Sort_double(double array_of_ints[], const int array_size);
	double Median_Double(int sizofArr, double *inputArr);
	double Median_Double(double *inputArr, int size_rangemask);
	double Minimum_Double(const double array_of_ints[], const int array_size);
	double Maximum_Double(const double array_of_ints[], const int array_size);
	double Average_Double(std::vector<double>inputArr);

	void Sort_Int(int array_of_ints[], const int array_size);
	int Median_Int(int sizofArr, int *inputArr);
	int Median_Int(std::vector<int>inputArr);
	int Minimum_Int(const int array_of_ints[], const int array_size);
	int Maximum_Int(const int array_of_ints[], const int array_size);
	int Average_Int(std::vector<int>inputArr);

	void CalRotationMat(Mod_CVector input_vector, double *out_Mat);
	void CalRotationMat2(Mod_CVector input_vector, double *out_Mat);
	void Cal_inversematrix(int Num, double* Matrix, double* InverseMatrix); //������� ���ϴ� �Լ� Num = ����� ��� ���� �� 
	double cal_3Ddistance(Mod_S3DCoord Point1, Mod_S3DCoord Point2);
	double cal_2Ddistance(Mod_S2DCoord Point1, Mod_S2DCoord Point2);
	double cal_2DdistanceLine_Point(Mod_S2DCoord Point, Mod_Line Line);
	double cal_3DdistanceLine_Point(Mod_S3DCoord Point, Mod_BK3DLine Line);
	double cal_3DdistancePlane_Point(double A, double B, double C, double D, Mod_S3DCoord Point);
	void Cal_CannyEdge(Mod_SColor* Image, bool* CannyEdge);
	Mod_BK3DLine Cal_CrossLine(Mod_BKPlane Plane1, Mod_BKPlane Plane2); //�� ����� �����ϴ� ������ ������
	Mod_S3DCoord Cal_IntersectionPointPlane_Line(double A, double B, double C, double D, Mod_BK3DLine Line);
	//Fitting �˰���//////////////////////////////////////////////////////////////
	void Ca_PlaneEquation(Mod_CVector Normal_Vector, Mod_S3DCoord Mean_Point, double &A, double &B, double &C, double &D);
	void Ellipsefitting(std::vector <Mod_S2DCoord> Ellipse_Point, Mod_BKEllipse &Ellipse);
	double EllipseErrorDistance(Mod_S2DCoord Point, Mod_BKEllipse Ellipse);
	//Mod_S2DCoord Circlefitting(double &Radius, double &LAvr, vector <Mod_S2DCoord> Circle_Point); //2DCircle Fitting ���ϰ��� ���� �߽���, �Ű����� 1: ���� ������ ���, 2: Fitting�� ���� 2D ����Ʈ
	Mod_Line Linefitting(std::vector <Mod_S2DCoord> Line_Point); //2D Line Fitting ���ϰ��� 2D ������ ������ ���, �Ű����� : Fitting�� ���� 2D ����Ʈ	
	Mod_Conic ConicFitting(std::vector <Mod_S2DCoord> Conic_Point);
	Mod_S2DCoord Circlefitting(double &Radius, std::vector <Mod_S2DCoord> Circle_Point);

	//3D Point ȸ��
	Mod_CVector RoatedVector(double *r_mat, Mod_CVector vec);
	Mod_S3DCoord RoatedPoint(double *r_mat, Mod_S3DCoord point);
	Mod_CVector TransPosedVector(double *r_mat, Mod_CVector vec);
	Mod_S3DCoord TransPosedPoint(double *T_mat, Mod_S3DCoord point);
	Mod_S3DCoord RoatedPointAxisY(double Degree, Mod_S3DCoord Point);
	Mod_S3DCoord RoatedPointAxisZ(double Degree, Mod_S3DCoord Point);
	Mod_CVector RoatedVecAxisX(double Degree, Mod_CVector vec);
	Mod_CVector RoatedVecAxisY(double Degree, Mod_CVector vec);
	Mod_CVector RoatedVecAxisZ(double Degree, Mod_CVector vec);
	void Cal_ObjectRotation(Mod_CVector x_axis, Mod_CVector y_axis, Mod_CVector z_axis, double *r_mat);

};

#endif // !defined(AFX_MOD_CMATRIX_H__87515FD3_2DCC_473A_B256_28F864450414__INCLUDED_)
