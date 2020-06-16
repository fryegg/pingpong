// Mod_CMatrix.cpp: implementation of the Mod_CMatrix class.
//
//////////////////////////////////////////////////////////////////////

#include "Mod_CMatrix.h"
//#include "Conicfit.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
#define TRUE		1
#define FALSE		0
#define ERR			-30000
#define Not_ERR		1
#define IMG_WIDTH	640
#define IMG_HEIGHT	480


Mod_CMatrix::Mod_CMatrix()
{

}

Mod_CMatrix::~Mod_CMatrix()
{
	
}

double Mod_CMatrix::det2x2(double a, double b, double c, double d)
{
	return
		a * d - b * c;
}

double Mod_CMatrix::det3x3(double a1, double a2, double a3, double b1, double b2, double b3, double c1, double c2, double c3)
{
	double out_det;

	out_det = a1 * det2x2(b2, b3, c2, c3) - b1 * det2x2(a2, a3, c2, c3) + c1 * det2x2(a2, a3, b2, b3);

	return
		out_det;
}

void Mod_CMatrix::MutiplyMatrix3x3(double* A, double* B, double* answer)
{
	answer[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
	answer[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
	answer[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
	answer[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
	answer[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
	answer[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
	answer[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
	answer[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
	answer[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}

void Mod_CMatrix::MutiplyMatrix4x4(double* A, double* B, double* answer)
{
	answer[0] = A[0] * B[0] + A[1] * B[4] + A[2] * B[8] + A[3] * B[12];
	answer[1] = A[0] * B[1] + A[1] * B[5] + A[2] * B[9] + A[3] * B[13];
	answer[2] = A[0] * B[2] + A[1] * B[6] + A[2] * B[10] + A[3] * B[14];
	answer[3] = A[0] * B[3] + A[1] * B[7] + A[2] * B[11] + A[3] * B[15];
	answer[4] = A[4] * B[0] + A[5] * B[4] + A[6] * B[8] + A[7] * B[12];
	answer[5] = A[4] * B[1] + A[5] * B[5] + A[6] * B[9] + A[7] * B[13];
	answer[6] = A[4] * B[2] + A[5] * B[6] + A[6] * B[10] + A[7] * B[14];
	answer[7] = A[4] * B[3] + A[5] * B[7] + A[6] * B[11] + A[7] * B[15];
	answer[8] = A[8] * B[0] + A[9] * B[4] + A[10] * B[8] + A[11] * B[12];
	answer[9] = A[8] * B[1] + A[9] * B[5] + A[10] * B[9] + A[11] * B[13];
	answer[10] = A[8] * B[2] + A[9] * B[6] + A[10] * B[10] + A[11] * B[14];
	answer[11] = A[8] * B[3] + A[9] * B[7] + A[10] * B[11] + A[11] * B[15];
	answer[12] = A[12] * B[0] + A[13] * B[4] + A[14] * B[8] + A[15] * B[12];
	answer[13] = A[12] * B[1] + A[13] * B[5] + A[14] * B[9] + A[15] * B[13];
	answer[14] = A[12] * B[2] + A[13] * B[6] + A[14] * B[10] + A[15] * B[14];
	answer[15] = A[12] * B[3] + A[13] * B[7] + A[14] * B[11] + A[15] * B[15];
}

double Mod_CMatrix::CalDist(Mod_S3DCoord p1, Mod_S3DCoord p2)
{
	double resultDist;
	resultDist = sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) + (p1.z - p2.z)*(p1.z - p2.z));
	return

		resultDist;
}

double Mod_CMatrix::DotProduct(Mod_CVector v1, Mod_CVector v2)
{
	double outValue;

	outValue = v1.X*v2.X + v1.Y*v2.Y + v1.Z*v2.Z;

	return
		outValue;
}

void Mod_CMatrix::TranposedMat3x3(double input_mat[9], double out_T_mat[9])
{
	//	double *out_T_mat =new double[9];

	out_T_mat[0] = input_mat[0];
	out_T_mat[1] = input_mat[3];
	out_T_mat[2] = input_mat[6];

	out_T_mat[3] = input_mat[1];
	out_T_mat[4] = input_mat[4];
	out_T_mat[5] = input_mat[7];

	out_T_mat[6] = input_mat[2];
	out_T_mat[7] = input_mat[5];
	out_T_mat[8] = input_mat[8];

	// 	return
	// 		out_T_mat;
}

Mod_CVector Mod_CMatrix::MakeUnitVec(Mod_CVector inputVector)
{
	Mod_CVector outputVector;
	double amp, outX, outY, outZ;

	outX = pow(inputVector.X * 1000, 2);
	outY = pow(inputVector.Y * 1000, 2);
	outZ = pow(inputVector.Z * 1000, 2);

	amp = sqrt(outX + outY + outZ) / 1000;
	if (amp != 0)
	{
		outputVector.X = inputVector.X / amp;
		outputVector.Y = inputVector.Y / amp;
		outputVector.Z = inputVector.Z / amp;

	}
	else
	{
		outputVector.init();
	}
	return
		outputVector;
}

Mod_CVector Mod_CMatrix::CrossProduct(Mod_CVector start, Mod_CVector dest)
{
	Mod_CVector C;
	C.X = start.Y * dest.Z - start.Z * dest.Y;
	C.Y = start.Z * dest.X - start.X * dest.Z;
	C.Z = start.X * dest.Y - start.Y * dest.X;
	return C;
}

Mod_CVector Mod_CMatrix::MakeDirVec_norm(Mod_S3DCoord start, Mod_S3DCoord dest)
{
	Mod_CVector out_vector;
	out_vector.X = dest.x - start.x;
	out_vector.Y = dest.y - start.y;
	out_vector.Z = dest.z - start.z;
	out_vector = MakeUnitVec(out_vector);
	return
		out_vector;
}

Mod_CVector Mod_CMatrix::MakeDirVec(Mod_S3DCoord start, Mod_S3DCoord dest)
{
	Mod_CVector out_vector;
	out_vector.X = dest.x - start.x;
	out_vector.Y = dest.y - start.y;
	out_vector.Z = dest.z - start.z;
	return		out_vector;
}

Mod_CVector Mod_CMatrix::Cal_MeshNorm(Mod_CVector *norm_Vector, PrimitivePoint mesh)
{
	int i, index, num_Points = 0;
	Mod_CVector mean_Vector;

	mean_Vector.X = 0;
	mean_Vector.Y = 0;
	mean_Vector.Z = 0;

	for (i = 0; i< (int)mesh.PointIndex.size(); i++)
	{
		index = mesh.PointIndex[i];
		if (norm_Vector[index].X != 0 && norm_Vector[index].Y != 0 && norm_Vector[index].Z != 0)
		{
			mean_Vector.X += norm_Vector[index].X;
			mean_Vector.Y += norm_Vector[index].Y;
			mean_Vector.Z += norm_Vector[index].Z;
			num_Points++;
		}
	}

	mean_Vector.X /= num_Points;
	mean_Vector.Y /= num_Points;
	mean_Vector.Z /= num_Points;

	mean_Vector = MakeUnitVec(mean_Vector);

	return mean_Vector;
}

int Mod_CMatrix::CalCovMat_Mean(std::vector<Mod_S3DCoord> input, Mod_S3DCoord &mean, double* out_cov)
{
	int i;
	Mod_S3DCoord* submean = new Mod_S3DCoord[input.size()];
	if (submean == NULL) return ERR;

	CalSubMeanMatrix(input, mean, submean);
	for (i = 0; i<9; i++)
	{
		out_cov[i] = 0;
	}

	for (i = 0; i<(int)input.size(); i++)
	{
		out_cov[0] = out_cov[0] + submean[i].x * submean[i].x;
	}

	for (i = 0; i<(int)input.size(); i++)
	{
		out_cov[1] = out_cov[1] + submean[i].x * submean[i].y;
	}

	for (i = 0; i<(int)input.size(); i++)
	{
		out_cov[2] = out_cov[2] + submean[i].x * submean[i].z;
	}

	for (i = 0; i<(int)input.size(); i++)
	{
		out_cov[3] = out_cov[3] + submean[i].y * submean[i].x;
	}

	for (i = 0; i<(int)input.size(); i++)
	{
		out_cov[4] = out_cov[4] + submean[i].y * submean[i].y;
	}

	for (i = 0; i<(int)input.size(); i++)
	{
		out_cov[5] = out_cov[5] + submean[i].y * submean[i].z;
	}

	for (i = 0; i<(int)input.size(); i++)
	{
		out_cov[6] = out_cov[6] + submean[i].z * submean[i].x;
	}

	for (i = 0; i<(int)input.size(); i++)
	{
		out_cov[7] = out_cov[7] + submean[i].z * submean[i].y;
	}

	for (i = 0; i<(int)input.size(); i++)
	{
		out_cov[8] = out_cov[8] + submean[i].z * submean[i].z;
	}

	if (submean) { delete[] submean; submean = NULL; }
	return	Not_ERR;
}

int Mod_CMatrix::Cal2DCovMat_Mean(std::vector<Mod_S2DCoord> input, Mod_S2DCoord &mean, double* out_cov)
{
	int i;
	Mod_S2DCoord* submean = new Mod_S2DCoord[input.size()];
	if (submean == NULL) return ERR;

	Cal2DSubMeanMatrix(input, mean, submean);
	for (i = 0; i<4; i++)
	{
		out_cov[i] = 0;
	}

	for (i = 0; i<(int)input.size(); i++)
	{
		out_cov[0] = out_cov[0] + submean[i].x * submean[i].x;
	}

	for (i = 0; i<(int)input.size(); i++)
	{
		out_cov[1] = out_cov[1] + submean[i].x * submean[i].y;
	}

	for (i = 0; i<(int)input.size(); i++)
	{
		out_cov[2] = out_cov[2] + submean[i].x * submean[i].y;
	}

	for (i = 0; i<(int)input.size(); i++)
	{
		out_cov[3] = out_cov[3] + submean[i].y * submean[i].y;
	}

	if (submean) { delete[] submean; submean = NULL; }
	return	Not_ERR;
}


void Mod_CMatrix::CalSubMeanMatrix(std::vector <Mod_S3DCoord> input, Mod_S3DCoord &mean, Mod_S3DCoord* outmat)
{
	int i;
	mean = CalMatrixMean(input);

	for (i = 0; i<(int)input.size(); i++)
	{
		outmat[i].x = input[i].x - mean.x;
		outmat[i].y = input[i].y - mean.y;
		outmat[i].z = input[i].z - mean.z;
	}
}

void Mod_CMatrix::Cal2DSubMeanMatrix(std::vector <Mod_S2DCoord> input, Mod_S2DCoord &mean, Mod_S2DCoord* outmat)
{
	int i;
	mean = Cal2DMatrixMean(input);

	for (i = 0; i<(int)input.size(); i++)
	{
		outmat[i].x = input[i].x - mean.x;
		outmat[i].y = input[i].y - mean.y;
	}
}

Mod_S3DCoord Mod_CMatrix::CalMatrixMean(std::vector <Mod_S3DCoord> input)
{
	int i;
	Mod_S3DCoord mean;
	mean.x = 0;	mean.y = 0; mean.z = 0;

	for (i = 0; i<(int)input.size(); i++)
	{
		mean.x = mean.x + input[i].x;
		mean.y = mean.y + input[i].y;
		mean.z = mean.z + input[i].z;
	}

	mean.x = mean.x / input.size();
	mean.y = mean.y / input.size();
	mean.z = mean.z / input.size();

	return
		mean;
}

Mod_S2DCoord Mod_CMatrix::Cal2DMatrixMean(std::vector <Mod_S2DCoord> input)
{
	int i;
	Mod_S2DCoord mean;
	mean.x = 0;	mean.y = 0;

	for (i = 0; i<(int)input.size(); i++)
	{
		mean.x = mean.x + input[i].x;
		mean.y = mean.y + input[i].y;
	}

	mean.x = mean.x / input.size();
	mean.y = mean.y / input.size();

	return
		mean;
}

/*
double* Mod_CMatrix::SVDwithCov(std::vector<Mod_S3DCoord> input, Mod_S3DCoord &mean, Mod_CVector eigen_Vector[3])
{
double *eigen_Value= new double[3];
double *covMat = new double [9];
covMat = CalCovMat_Mean(input, mean);
eigen_Value = CalSVD(covMat, eigen_Vector);
delete[] covMat;
return
eigen_Value;
}
*/
int Mod_CMatrix::SVDwithCov(std::vector<Mod_S3DCoord> input, Mod_CVector* eigen_Vector, double *eigen_Value)
{
	Mod_S3DCoord mean;
	double covMat[9];
	if (CalCovMat_Mean(input, mean, covMat) == ERR) return ERR;
	CalSVD(covMat, eigen_Vector, eigen_Value);
	return Not_ERR;
}

int Mod_CMatrix::SVDwithCovIn2D(std::vector<Mod_S2DCoord> input, Mod_C2DVector* eigen_Vector, double *eigen_Value)
{
	Mod_S2DCoord mean;
	double covMat[4];
	if (Cal2DCovMat_Mean(input, mean, covMat) == ERR) return ERR;

	CvMat cov_mat; CvMat SVD_mat; CvMat SVD_V_mat;
	double SVD[4];
	double SVD_V[4];

	cvInitMatHeader(&cov_mat, 2, 2, CV_64FC1, covMat);
	cvInitMatHeader(&SVD_mat, 2, 2, CV_64FC1, SVD);
	cvInitMatHeader(&SVD_V_mat, 2, 2, CV_64FC1, SVD_V);

	cvSVD(&cov_mat, &SVD_mat, NULL, &SVD_V_mat, CV_SVD_V_T);

	eigen_Value[0] = SVD_mat.data.db[0];
	eigen_Value[1] = SVD_mat.data.db[3];


	eigen_Vector[0].X = SVD_V_mat.data.db[0];
	eigen_Vector[0].Y = SVD_V_mat.data.db[1];

	eigen_Vector[1].X = SVD_V_mat.data.db[2];
	eigen_Vector[1].Y = SVD_V_mat.data.db[3];

	return Not_ERR;
}

/*
double* Mod_CMatrix::SVDwithCov(vector<Mod_CVector> input, Mod_CVector eigen_Vector[3])
{
int i;
vector <Mod_S3DCoord> InputCoord;
for(i=0; i<(int)input.size(); i++)
{
Mod_S3DCoord temp_value;
temp_value.x = input[i].X;
temp_value.y = input[i].Y;
temp_value.z = input[i].Z;
InputCoord.push_back(temp_value);
}

Mod_S3DCoord mean;
double *eigen_Value= new double[3];

double *covMat = new double [9];
covMat = CalCovMat_Mean(InputCoord, mean);
eigen_Value = CalSVD(covMat, eigen_Vector);
InputCoord.clear();
delete[] covMat;
return
eigen_Value;
}
*/
void Mod_CMatrix::CalSVD(double* covMat, Mod_CVector* eigen_Vector, double* eigen_Value)
{
	//CvMat mean_mat;
	CvMat cov_mat; CvMat SVD_mat; CvMat SVD_V_mat;
	double SVD[9];
	double SVD_V[9];

	cvInitMatHeader(&cov_mat, 3, 3, CV_64FC1, covMat);
	cvInitMatHeader(&SVD_mat, 3, 3, CV_64FC1, SVD);
	cvInitMatHeader(&SVD_V_mat, 3, 3, CV_64FC1, SVD_V);

	cvSVD(&cov_mat, &SVD_mat, NULL, &SVD_V_mat, CV_SVD_V_T);

	eigen_Value[0] = SVD_mat.data.db[0];
	eigen_Value[1] = SVD_mat.data.db[4];
	eigen_Value[2] = SVD_mat.data.db[8];

	eigen_Vector[0].X = SVD_V_mat.data.db[0];
	eigen_Vector[0].Y = SVD_V_mat.data.db[1];
	eigen_Vector[0].Z = SVD_V_mat.data.db[2];

	eigen_Vector[1].X = SVD_V_mat.data.db[3];
	eigen_Vector[1].Y = SVD_V_mat.data.db[4];
	eigen_Vector[1].Z = SVD_V_mat.data.db[5];

	eigen_Vector[2].X = SVD_V_mat.data.db[6];
	eigen_Vector[2].Y = SVD_V_mat.data.db[7];
	eigen_Vector[2].Z = SVD_V_mat.data.db[8];

}
//////////////////////////////////////////////////////////////////////////

void Mod_CMatrix::Sort_double(double array_of_ints[], const int array_size)
{
	int i = 0;
	int j = 0;
	int increment = 3;
	double temp = 0;

	while (increment > 0)
	{
		for (i = 0; i < array_size; i++)
		{
			j = i;
			temp = array_of_ints[i];
			while ((j >= increment) && (array_of_ints[j - increment] > temp))
			{
				array_of_ints[j] = array_of_ints[j - increment];
				j = j - increment;
			}
			array_of_ints[j] = temp;
		}
		if (increment / 2 != 0)
		{
			increment = (increment / 2);
		}
		else if (increment == 1)
		{
			increment = 0;
		}
		else
		{
			increment = 1;
		}
	}
}

void Mod_CMatrix::Sort_Int(int array_of_ints[], const int array_size)
{
	int i = 0;
	int j = 0;
	int increment = 3;
	int temp = 0;

	while (increment > 0)
	{
		for (i = 0; i < array_size; i++)
		{
			j = i;
			temp = array_of_ints[i];
			while ((j >= increment) && (array_of_ints[j - increment] > temp))
			{
				array_of_ints[j] = array_of_ints[j - increment];
				j = j - increment;
			}
			array_of_ints[j] = temp;
		}
		if (increment / 2 != 0)
		{
			increment = (increment / 2);
		}
		else if (increment == 1)
		{
			increment = 0;
		}
		else
		{
			increment = 1;
		}
	}
}

double Mod_CMatrix::Median_Double(double *inputArr, int size_rangemask)
{
	int i = size_rangemask;
	int j;
	double* pInts = new double[i];
	double temp = 0;
	if (i > 0)
	{
		for (j = 0; j<i; j++)
		{
			pInts[j] = inputArr[j];
		}

		Sort_double(pInts, i);
		temp = pInts[((int)i / 2)];
	}

	else
	{
		printf("usage error: parameter array_size must be greater than zero!\n");
	}
	if (pInts)
		delete[] pInts;
	return temp;
}

int Mod_CMatrix::Median_Int(std::vector<int>inputArr)
{
	int i = inputArr.size();
	int j;
	int* pInts = new int[i];
	int temp = 0;
	if (i > 0)
	{
		for (j = 0; j<i; j++)
		{
			pInts[j] = inputArr[j];
		}

		Sort_Int(pInts, i);
		temp = pInts[((int)i / 2)];
	}

	else
	{
		printf("usage error: parameter array_size must be greater than zero!\n");
	}

	delete[] pInts;
	return temp;
}

double Mod_CMatrix::Median_Double(int sizofArr, double *inputArr)
{
	int i = sizofArr;
	double* pInts = 0;
	double temp = 0;
	if (i > 0)
	{
		memcpy(pInts, inputArr, (i * sizeof(double)));
		Sort_double(pInts, i);
		temp = pInts[((int)i / 2)];
	}

	else
	{
		printf("usage error: parameter array_size must be greater than zero!\n");
	}

	delete[] pInts;
	return temp;
}

int Mod_CMatrix::Median_Int(int sizofArr, int *inputArr)
{
	int i = sizofArr;
	int* pInts = 0;
	int temp = 0;
	if (i > 0)
	{
		memcpy(pInts, inputArr, (i * sizeof(int)));
		Sort_Int(pInts, i);
		temp = pInts[((int)i / 2)];
	}

	else
	{
		printf("usage error: parameter array_size must be greater than zero!\n");
	}

	delete[] pInts;
	return temp;
}

double Mod_CMatrix::Minimum_Double(const double array_of_ints[], const int array_size)
{
	int i = array_size;
	double temp = array_of_ints[0];
	if (i > 0)
	{
		do
		{
			temp = std::min(temp, array_of_ints[-1 + i]);
		} while ((--i) > 0);
	}
	else
	{
		printf("usage error: parameter array_size must be greater than zero!\n");
	}
	return temp;
}

int Mod_CMatrix::Minimum_Int(const int array_of_ints[], const int array_size)
{
	int i = array_size;
	int temp = array_of_ints[0];
	if (i > 0)
	{
		do
		{
			temp = std::min(temp, array_of_ints[-1 + i]);
		} while ((--i) > 0);
	}
	else
	{
		printf("usage error: parameter array_size must be greater than zero!\n");
	}
	return temp;
}

double Mod_CMatrix::Maximum_Double(const double array_of_ints[], const int array_size)
{
	int i = array_size;
	double temp = array_of_ints[0];
	if (i > 0)
	{
		do
		{
			temp = std::max(temp, array_of_ints[-1 + i]);
		} while ((--i) > 0);
	}
	else
	{
		printf("usage error: parameter array_size must be greater than zero!\n");
	}
	return temp;
}

int Mod_CMatrix::Maximum_Int(const int array_of_ints[], const int array_size)
{
	int i = array_size;
	int temp = array_of_ints[0];
	if (i > 0)
	{
		do
		{
			temp = std::max(temp, array_of_ints[-1 + i]);
		} while ((--i) > 0);
	}
	else
	{
		printf("usage error: parameter array_size must be greater than zero!\n");
	}
	return temp;
}

double Mod_CMatrix::Average_Double(std::vector<double>inputArr)
{
	int i = 0;
	int numofArr = inputArr.size();
	double sum = 0;
	double out_aver = 0;

	for (i = 0; i<numofArr; i++)
	{
		sum = sum + inputArr[i];
	}

	out_aver = sum / numofArr;
	return
		out_aver;
}

int Mod_CMatrix::Average_Int(std::vector<int>inputArr)
{
	int i = 0;
	int numofArr = inputArr.size();
	int sum = 0;
	int out_aver = 0;

	for (i = 0; i<numofArr; i++)
	{
		sum = sum + inputArr[i];
	}

	out_aver = sum / numofArr;
	return
		out_aver;
}

void Mod_CMatrix::CalRotationMat(Mod_CVector input_vector, double *out_Mat)
{
	double theta_x, theta_z;
	double tempyn;

	theta_x = atan(-input_vector.Z / input_vector.Y);
	tempyn = cos(theta_x)*input_vector.Y - sin(theta_x)*input_vector.Z;
	theta_z = atan(input_vector.X / tempyn);

	out_Mat[0] = cos(theta_z);
	out_Mat[1] = -sin(theta_z)*cos(theta_x);
	out_Mat[2] = sin(theta_z)*sin(theta_x);

	out_Mat[3] = sin(theta_z);
	out_Mat[4] = cos(theta_z)*cos(theta_x);
	out_Mat[5] = -cos(theta_z)*sin(theta_x);

	out_Mat[6] = 0;
	out_Mat[7] = sin(theta_x);
	out_Mat[8] = cos(theta_x);
	//printf("theta_X1 : %lf\n",theta_x*57.295);
	//printf("theta_Z1 : %lf\n", theta_z*57.295);
}


void Mod_CMatrix::CalRotationMat2(Mod_CVector input_vector, double *out_Mat)
{
	double theta_x, theta_z;

	theta_x = asin(input_vector.Z);
	theta_z = asin(-input_vector.X / cos(theta_x));

	out_Mat[0] = cos(theta_z);
	out_Mat[1] = -sin(theta_z)*cos(theta_x);
	out_Mat[2] = sin(theta_z)*sin(theta_x);

	out_Mat[3] = sin(theta_z);
	out_Mat[4] = cos(theta_z)*cos(theta_x);
	out_Mat[5] = -cos(theta_z)*sin(theta_x);

	out_Mat[6] = 0;
	out_Mat[7] = sin(theta_x);
	out_Mat[8] = cos(theta_x);
	printf("theta_X : %lf\n", theta_x*57.295);
	printf("theta_Z : %lf\n", theta_z*57.295);
}

/*
void Mod_CMatrix::CalRotationMat2(Mod_CVector input_vector, double *out_Mat)
{
double theta_x, theta_z;
double tempyn, tempyn2;

theta_x = -atan2(input_vector.Z, input_vector.Y);
tempyn = cos(theta_x);
tempyn2 = -sin(theta_x);

printf("Trans_Y : %lf\n", tempyn);
theta_z = atan2(input_vector.X, -input_vector.Y);


printf("Recog_Vec.X : %lf\n", input_vector.X);
printf("Recog_Vec.Y : %lf\n", input_vector.Y);
printf("Recog_Vec.Z : %lf\n", input_vector.Z);
printf("theta_X : %lf\n",theta_x*57.295);
printf("theta_Z : %lf\n", theta_z*57.295);

out_Mat[0] = cos(theta_z);
out_Mat[1] = -sin(theta_z)*cos(theta_x);
out_Mat[2] = sin(theta_z)*sin(theta_x);

out_Mat[3] = sin(theta_z);
out_Mat[4] = cos(theta_z)*cos(theta_x);
out_Mat[5] = -cos(theta_z)*sin(theta_x);

out_Mat[6] = 0;
out_Mat[7] = sin(theta_x);
out_Mat[8] = cos(theta_x);
}
*/
//
//void Mod_CMatrix::CalRotationMat2(Mod_CVector input_vector, double *out_Mat)
//{
//	double theta_x, theta_z;
//	double tempyn;
//	
//	tempyn = sqrt(pow(input_vector.Z,2) + (input_vector.X,2));
//	theta_x = acos(tempyn);
//	theta_z = atan2(input_vector.X, -input_vector.Y);
//
//	printf("tempyn : %lf\n", tempyn);
//	printf("Recog_Vec.X : %lf\n", input_vector.X);
//	printf("Recog_Vec.Y : %lf\n", input_vector.Y);
//	printf("Recog_Vec.Z : %lf\n", input_vector.Z);
//	printf("theta_X : %lf\n",theta_x*57.295);
//	printf("theta_Z : %lf\n", theta_z*57.295);
//
//	out_Mat[0] = cos(theta_z);
//	out_Mat[1] = -sin(theta_z)*cos(theta_x);
//	out_Mat[2] = sin(theta_z)*sin(theta_x);
//
//	out_Mat[3] = sin(theta_z);
//	out_Mat[4] = cos(theta_z)*cos(theta_x);
//	out_Mat[5] = -cos(theta_z)*sin(theta_x);
//
//	out_Mat[6] = 0;
//	out_Mat[7] = sin(theta_x);
//	out_Mat[8] = cos(theta_x);	
//}

bool Mod_CMatrix::Inverse3x3(double input_mat[9], double Inv_Mat[9])
{
	double det_mat;
	det_mat = det3x3(
		input_mat[0], input_mat[1], input_mat[2],
		input_mat[3], input_mat[4], input_mat[5],
		input_mat[6], input_mat[7], input_mat[8]);

	if (det_mat)
	{
		Inv_Mat[0] = det2x2(input_mat[4], input_mat[5], input_mat[7], input_mat[8]) / det_mat;
		Inv_Mat[3] = -det2x2(input_mat[3], input_mat[5], input_mat[6], input_mat[8]) / det_mat;
		Inv_Mat[6] = det2x2(input_mat[3], input_mat[4], input_mat[6], input_mat[7]) / det_mat;

		Inv_Mat[1] = -det2x2(input_mat[1], input_mat[2], input_mat[7], input_mat[8]) / det_mat;
		Inv_Mat[4] = det2x2(input_mat[0], input_mat[2], input_mat[6], input_mat[8]) / det_mat;
		Inv_Mat[7] = -det2x2(input_mat[0], input_mat[1], input_mat[6], input_mat[7]) / det_mat;

		Inv_Mat[2] = det2x2(input_mat[1], input_mat[2], input_mat[4], input_mat[5]) / det_mat;
		Inv_Mat[5] = -det2x2(input_mat[0], input_mat[2], input_mat[3], input_mat[5]) / det_mat;
		Inv_Mat[8] = det2x2(input_mat[0], input_mat[1], input_mat[3], input_mat[4]) / det_mat;
		return
			TRUE;
	}
	return
		FALSE;
}


double Mod_CMatrix::cal_3Ddistance(Mod_S3DCoord Point1, Mod_S3DCoord Point2)
{
	double distance;
	distance = pow((Point1.x - Point2.x) * 1000, 2) + pow((Point1.y - Point2.y) * 1000, 2) + pow((Point1.z - Point2.z) * 1000, 2);
	distance = sqrt(distance) / 1000;
	return distance;
}

double Mod_CMatrix::cal_2Ddistance(Mod_S2DCoord Point1, Mod_S2DCoord Point2)
{
	double distance;
	distance = pow((Point1.x - Point2.x) * 1000, 2) + pow((Point1.y - Point2.y) * 1000, 2);
	distance = sqrt(distance) / 1000;
	return distance;
}

double Mod_CMatrix::cal_2DdistanceLine_Point(Mod_S2DCoord Point, Mod_Line Line)
{
	double distance;
	double variance[2];

	variance[0] = Line.b*Point.x - Point.y + Line.a;
	variance[0] = fabs(variance[0]);
	variance[1] = pow(Line.b, 2) + 1;
	variance[1] = sqrt(variance[1]);

	distance = variance[0] / variance[1];
	return distance;
}

double Mod_CMatrix::cal_3DdistanceLine_Point(Mod_S3DCoord Point, Mod_BK3DLine Line)
{
	double Distance;
	double T, Variance[4];
	Mod_CVector Point_To_Line;

	Variance[0] = Line.Direction_Vector.X*(Point.x - Line.Point.x);
	Variance[1] = Line.Direction_Vector.Y*(Point.y - Line.Point.y);
	Variance[2] = Line.Direction_Vector.Z*(Point.z - Line.Point.z);
	Variance[3] = pow(Line.Direction_Vector.X, 2) + pow(Line.Direction_Vector.Y, 2) + pow(Line.Direction_Vector.Z, 2);
	T = (Variance[0] + Variance[1] + Variance[2]) / Variance[3];

	Point_To_Line.X = Point.x - (T*Line.Direction_Vector.X + Line.Point.x);
	Point_To_Line.Y = Point.y - (T*Line.Direction_Vector.Y + Line.Point.y);
	Point_To_Line.Z = Point.z - (T*Line.Direction_Vector.Z + Line.Point.z);

	Variance[0] = pow(Point_To_Line.X, 2);
	Variance[1] = pow(Point_To_Line.Y, 2);
	Variance[2] = pow(Point_To_Line.Z, 2);

	Distance = sqrt(Variance[0] + Variance[1] + Variance[2]);
	return Distance;
}

void Mod_CMatrix::Cal_CannyEdge(Mod_SColor* Image, bool* CannyEdge)
{
	IplImage *RGB_Image = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);
	IplImage *Gray_Image = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 1);
	IplImage *Edge_Image = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 1);
	for (int i = 0; i<640 * 480; i++)
	{
		RGB_Image->imageData[3 * i] = char(Image[i].b);
		RGB_Image->imageData[3 * i + 1] = char(Image[i].g);
		RGB_Image->imageData[3 * i + 2] = char(Image[i].r);
	}

	cvCvtColor(RGB_Image, Gray_Image, CV_BGR2GRAY);
	cvCanny(Gray_Image, Edge_Image, 50, 200, 3);
	for (int i = 0; i<640 * 480; i++) if (Edge_Image->imageData[i] != 0) CannyEdge[i] = 1;
	if (RGB_Image) { cvReleaseImage(&RGB_Image), RGB_Image = NULL; }
	if (Gray_Image) { cvReleaseImage(&Gray_Image), Gray_Image = NULL; }
	if (Edge_Image) { cvReleaseImage(&Edge_Image), Edge_Image = NULL; }
}

Mod_BK3DLine Mod_CMatrix::Cal_CrossLine(Mod_BKPlane Plane1, Mod_BKPlane Plane2)
{
	double x, y, z;
	double A1, B1, C1, D1, A2, B2, C2, D2;
	Mod_CVector  Cross_Vector;
	Mod_BK3DLine out_Line;

	A1 = Plane1.Normal_Vector.X; B1 = Plane1.Normal_Vector.Y; C1 = Plane1.Normal_Vector.Z;
	A2 = Plane2.Normal_Vector.X; B2 = Plane2.Normal_Vector.Y; C2 = Plane2.Normal_Vector.Z;
	D1 = -(A1*Plane1.meanWpoint.x + B1*Plane1.meanWpoint.y + C1*Plane1.meanWpoint.z);
	D2 = -(A2*Plane2.meanWpoint.x + B2*Plane2.meanWpoint.y + C2*Plane2.meanWpoint.z);

	if ((A2 / B2 - A1 / B1) != 0) x = (D1 / B1 - D2 / B2) / (A2 / B2 - A1 / B1);
	else x = 0;
	if ((B2 / A2 - B1 / A1) != 0) y = (D1 / A1 - D2 / A2) / (B2 / A2 - B1 / A1);
	else y = 0;
	z = 0;

	Cross_Vector = CrossProduct(Plane1.Normal_Vector, Plane2.Normal_Vector);
	Cross_Vector = MakeUnitVec(Cross_Vector);

	out_Line.Direction_Vector = Cross_Vector;
	out_Line.Point.x = x; out_Line.Point.y = y; out_Line.Point.z = z;

	return out_Line;
}

Mod_S3DCoord Mod_CMatrix::Cal_IntersectionPointPlane_Line(double A, double B, double C, double D, Mod_BK3DLine Line)
{
	double variance[3];
	Mod_S3DCoord temp_Point;

	variance[0] = -(A*Line.Point.x + B*Line.Point.y + C*Line.Point.z + D);
	variance[1] = A*Line.Direction_Vector.X + B*Line.Direction_Vector.Y + C*Line.Direction_Vector.Z;
	variance[2] = variance[0] / variance[1];

	temp_Point.x = Line.Direction_Vector.X*variance[2] + Line.Point.x;
	temp_Point.y = Line.Direction_Vector.Y*variance[2] + Line.Point.y;
	temp_Point.z = Line.Direction_Vector.Z*variance[2] + Line.Point.z;

	return temp_Point;
}

void Mod_CMatrix::Cal_inversematrix(int Num, double* Matrix, double* InverseMatrix)
{
	int i, j, m;
	double divide;
	double* temp_Matrix, *temp_InverseMatrix;


	temp_Matrix = new double[Num];
	temp_InverseMatrix = new double[Num];

	for (i = 0; i<Num*Num; i++) InverseMatrix[i] = 0;
	for (i = 0; i<Num; i++) InverseMatrix[i*Num + i] = 1;

	for (i = 0; i<Num; i++)
	{
		divide = Matrix[i*Num + i];
		for (j = 0; j<Num; j++) Matrix[i*Num + j] = Matrix[i*Num + j] / divide;
		for (j = 0; j<Num; j++) InverseMatrix[i*Num + j] = InverseMatrix[i*Num + j] / divide;
		for (j = 0; j<Num; j++)	//행렬의 행
		{
			for (m = 0; m<Num; m++)
			{
				temp_Matrix[m] = Matrix[i*Num + m] * Matrix[j*Num + i];
				temp_InverseMatrix[m] = InverseMatrix[i*Num + m] * Matrix[j*Num + i];
			}
			for (m = 0; m<Num; m++)
			{
				if (i != j)
				{
					Matrix[j*Num + m] = Matrix[j*Num + m] - temp_Matrix[m];
					InverseMatrix[j*Num + m] = InverseMatrix[j*Num + m] - temp_InverseMatrix[m];
				}
			}
		}
	}

	//for(i=0; i<Num*Num; i++) printf("\nMatrix[%d] = %lf",i, InverseMatrix[i]);
	if (temp_Matrix) delete[] temp_Matrix;
	if (temp_InverseMatrix) delete[] temp_InverseMatrix;
}

void Mod_CMatrix::Ca_PlaneEquation(Mod_CVector Normal_Vector, Mod_S3DCoord Mean_Point, double &A, double &B, double &C, double &D)
{
	A = Normal_Vector.X;
	B = Normal_Vector.Y;
	C = Normal_Vector.Z;
	D = -(A*Mean_Point.x + B*Mean_Point.y + C*Mean_Point.z);
}

double Mod_CMatrix::cal_3DdistancePlane_Point(double A, double B, double C, double D, Mod_S3DCoord Point)
{
	double Variable[5];

	Variable[0] = A*Point.x + B*Point.y + C*Point.z + D;
	Variable[1] = pow(A, 2) + pow(B, 2) + pow(C, 2);
	Variable[2] = fabs(Variable[0]);
	Variable[3] = sqrt(Variable[1]);
	Variable[4] = Variable[2] / Variable[3];

	return Variable[4];
}

/*void Mod_CMatrix::Ellipsefitting(vector <Mod_S2DCoord> Ellipse_Point, Mod_BKEllipse &Ellipse)
{
int size = (int) Ellipse_Point.size();

CvBox2D32f FitEllipse;
CvPoint2D32f* PointArray2D32f;
PointArray2D32f = new CvPoint2D32f [size];

if(size > 6)
{
for(int i=0; i<size; i++)
{
PointArray2D32f[i].x = (float) Ellipse_Point[i].x;
PointArray2D32f[i].y = (float) Ellipse_Point[i].y;
}
//cvFitEllipse(PointArray2D32f, size, &FitEllipse);


Ellipse.Angle = (double) FitEllipse.angle;
Ellipse.CenterPoint.x = (double) FitEllipse.center.x;
Ellipse.CenterPoint.y = (double) FitEllipse.center.y;
Ellipse.Height = (double) FitEllipse.size.height/2;
Ellipse.Width = (double) FitEllipse.size.width/2;
}

if(PointArray2D32f) delete [] PointArray2D32f;
}
*/

double Mod_CMatrix::EllipseErrorDistance(Mod_S2DCoord Point, Mod_BKEllipse Ellipse)
{
	Mod_S3DCoord Temp_Point[2];
	double Variance, Fit_Z[2], Distance[2];

	Temp_Point[0].x = Point.x - Ellipse.CenterPoint.x;
	Temp_Point[0].y = 0;
	Temp_Point[0].z = Point.y - Ellipse.CenterPoint.y;

	Temp_Point[1] = RoatedPointAxisY(Ellipse.Angle, Temp_Point[0]);
	Variance = pow(Ellipse.Height, 2)*(1 - (pow(Temp_Point[1].x, 2) / pow(Ellipse.Width, 2)));

	if (Variance > 0)
	{
		Fit_Z[0] = sqrt(Variance);
		Fit_Z[1] = -sqrt(Variance);

		Distance[0] = abs(Fit_Z[0] - Temp_Point[1].z);
		Distance[1] = abs(Fit_Z[1] - Temp_Point[1].z);

		if (Distance[0] < Distance[1]) return Distance[0];
		else return Distance[1];
	}

	else return 100000;
}

Mod_Line Mod_CMatrix::Linefitting(std::vector <Mod_S2DCoord> Line_Point)	//y= a + bx
{
	int i, Num_Point;
	double Matrix[4];
	double desteny_num[2];

	Mod_Line Coefficient;

	memset(Matrix, 0, sizeof(double) * 4);
	memset(desteny_num, 0, sizeof(double) * 2);

	Num_Point = (int)Line_Point.size();

	for (i = 0; i<Num_Point; i++)
	{
		Matrix[0] += 1;
		Matrix[1] += Line_Point[i].x;
		Matrix[2] += Line_Point[i].x;
		Matrix[3] += Line_Point[i].x * Line_Point[i].x;

		desteny_num[0] += Line_Point[i].y;
		desteny_num[1] += Line_Point[i].x * Line_Point[i].y;
	}

	CvMat *SrcMat1 = cvCreateMat(2, 2, CV_64FC1);
	CvMat *InvertSrcMat1 = cvCreateMat(2, 2, CV_64FC1);
	CvMat *SrcMat2 = cvCreateMat(2, 1, CV_64FC1);
	CvMat *Desteny_Mat = cvCreateMat(2, 1, CV_64FC1);

	cvInitMatHeader(SrcMat1, 2, 2, CV_64FC1, Matrix);
	cvInitMatHeader(SrcMat2, 2, 1, CV_64FC1, desteny_num);

	cvInvert(SrcMat1, InvertSrcMat1, CV_LU);
	cvMatMulAdd(InvertSrcMat1, SrcMat2, 0, Desteny_Mat);

	Coefficient.a = cvmGet(Desteny_Mat, 0, 0);
	Coefficient.b = cvmGet(Desteny_Mat, 1, 0);

	cvReleaseMat(&SrcMat1);
	cvReleaseMat(&InvertSrcMat1);
	cvReleaseMat(&SrcMat2);
	cvReleaseMat(&Desteny_Mat);

	return Coefficient;
}


//ConicFitting A+C = 1
/*
Mod_Conic Mod_CMatrix::ConicFitting(vector <Mod_S2DCoord> Conic_Point)
{
double X_2, Y_2, XY, X, Y;
double Variable[25] ,Desteny[5];
Mod_Conic Return_Vari;

memset(Variable, 0, sizeof(double)*25);
memset(Desteny, 0, sizeof(double)*5);

for(int i = 0; i< (int)Conic_Point.size(); i++)
{
X_2 = pow(Conic_Point[i].x, 2);
Y_2 = pow(Conic_Point[i].y, 2);
XY = Conic_Point[i].x * Conic_Point[i].y;
X = Conic_Point[i].x;
Y = Conic_Point[i].y;

Variable[0] += pow(X_2 - Y_2, 2);
Variable[1] += (X_2 - Y_2) * XY;
Variable[2] += (X_2 - Y_2) * X;
Variable[3] += (X_2 - Y_2) * Y;
Variable[4] += (X_2 - Y_2);
Desteny[0]  -= (X_2 - Y_2) * Y_2;

Variable[5] += XY * (X_2 - Y_2);
Variable[6] += pow(XY, 2);
Variable[7] += XY * X;
Variable[8] += XY * Y;
Variable[9] += XY;
Desteny[1]  -= XY * Y_2;

Variable[10] += X * (X_2 - Y_2);
Variable[11] += XY * X;
Variable[12] += X_2;
Variable[13] += XY;
Variable[14] += X;
Desteny[2]   -= X * Y_2;

Variable[15] += Y * (X_2 - Y_2);
Variable[16] += XY * Y;
Variable[17] += XY;
Variable[18] += Y_2;
Variable[19] += Y;
Desteny[3]   -= Y * Y_2;

Variable[20] += (X_2 - Y_2);
Variable[21] += XY;
Variable[22] += X;
Variable[23] += Y;
Variable[24] += 1;
Desteny[4]   -= Y_2;
}

CvMat *SrcMat = cvCreateMat(5, 5, CV_64FC1);
CvMat *InvertMat = cvCreateMat(5, 5, CV_64FC1);
CvMat *VariMat = cvCreateMat(5, 1, CV_64FC1);
CvMat *Desteny_Mat = cvCreateMat(5, 1, CV_64FC1);

cvInitMatHeader(SrcMat, 5, 5, CV_64FC1, Variable);
cvInitMatHeader(VariMat, 5, 1, CV_64FC1, Desteny);

cvInvert(SrcMat, InvertMat, CV_LU);
cvMatMulAdd(InvertMat, VariMat,0, Desteny_Mat);

Return_Vari.A = cvmGet(Desteny_Mat ,0,0);
Return_Vari.B = cvmGet(Desteny_Mat ,1,0);
Return_Vari.C = 1 - Return_Vari.A;
Return_Vari.D = cvmGet(Desteny_Mat ,2,0);
Return_Vari.E = cvmGet(Desteny_Mat ,3,0);
Return_Vari.F = cvmGet(Desteny_Mat ,4,0);

cvReleaseMat(&SrcMat);
cvReleaseMat(&InvertMat);
cvReleaseMat(&VariMat);
cvReleaseMat(&Desteny_Mat);

return Return_Vari;
}
//*/


//Conic Fitting F = 1 
/*
Mod_Conic Mod_CMatrix::ConicFitting(vector <Mod_S2DCoord> Conic_Point)
{
double X_2, Y_2, XY, X, Y;
double Variable[25] ,Desteny[5];
Mod_Conic Return_Vari;

memset(Variable, 0, sizeof(double)*25);
memset(Desteny, 0, sizeof(double)*5);

for(int i = 0; i< (int)Conic_Point.size(); i++)
{
X_2 = pow(Conic_Point[i].x, 2);
Y_2 = pow(Conic_Point[i].y, 2);
XY = Conic_Point[i].x * Conic_Point[i].y;
X = Conic_Point[i].x;
Y = Conic_Point[i].y;

Variable[0] += pow(X_2, 2);
Variable[1] += X_2 * XY;
Variable[2] += pow(XY, 2);
Variable[3] += X*X_2;
Variable[4] += X*XY;
Desteny[0]  -= X_2;

Variable[5] += XY * X_2;
Variable[6] += pow(XY, 2);
Variable[7] += XY * Y_2;
Variable[8] += XY * X;
Variable[9] += XY * Y;
Desteny[1]  -= XY;

Variable[10] += pow(XY, 2);
Variable[11] += XY * Y_2;
Variable[12] += pow(Y_2, 2);
Variable[13] += XY * Y;
Variable[14] += pow(Y, 3);
Desteny[2]   -= Y_2;

Variable[15] += pow(X, 3);
Variable[16] += XY * X;
Variable[17] += XY * Y;
Variable[18] += X_2;
Variable[19] += XY;
Desteny[3]   -= X;

Variable[20] += X_2 * Y;
Variable[21] += XY * Y;
Variable[22] += pow(Y, 3);
Variable[23] += XY;
Variable[24] += Y_2;
Desteny[4]   -= Y;
}

CvMat *SrcMat = cvCreateMat(5, 5, CV_64FC1);
CvMat *InvertMat = cvCreateMat(5, 5, CV_64FC1);
CvMat *VariMat = cvCreateMat(5, 1, CV_64FC1);
CvMat *Desteny_Mat = cvCreateMat(5, 1, CV_64FC1);

cvInitMatHeader(SrcMat, 5, 5, CV_64FC1, Variable);
cvInitMatHeader(VariMat, 5, 1, CV_64FC1, Desteny);

cvInvert(SrcMat, InvertMat, CV_LU);
cvMatMulAdd(InvertMat, VariMat,0, Desteny_Mat);

Return_Vari.A = cvmGet(Desteny_Mat ,0,0);
Return_Vari.B = cvmGet(Desteny_Mat ,1,0);
Return_Vari.C = cvmGet(Desteny_Mat ,2,0);
Return_Vari.D = cvmGet(Desteny_Mat ,3,0);
Return_Vari.E = cvmGet(Desteny_Mat ,4,0);
Return_Vari.F = 1;

cvReleaseMat(&SrcMat);
cvReleaseMat(&InvertMat);
cvReleaseMat(&VariMat);
cvReleaseMat(&Desteny_Mat);

return Return_Vari;
}
//*/

//Conic Fitting B^2-ac >= 0
/*
Mod_Conic Mod_CMatrix::ConicFitting(vector <Mod_S2DCoord> Conic_Point)
{
int SizeofPoint = (int) Conic_Point.size();
Mod_Conic Return_Vari;

double Variable[6];
double *Point_X = new double [SizeofPoint];
double *Point_Y = new double [SizeofPoint];

memset(Point_X, 0, sizeof(double)*SizeofPoint);
memset(Point_Y, 0, sizeof(double)*SizeofPoint);


for(int i=0; i<SizeofPoint; i++)
{
Point_X[i] = Conic_Point[i].x;
Point_Y[i] = Conic_Point[i].y;
}

mwArray P_X(SizeofPoint, 1, mxDOUBLE_CLASS, mxREAL);
mwArray P_Y(SizeofPoint, 1, mxDOUBLE_CLASS, mxREAL);
mwArray Conic(6, 1, mxDOUBLE_CLASS, mxREAL);

P_X.SetData(Point_X, SizeofPoint);
P_Y.SetData(Point_Y, SizeofPoint);

Conicfit(1, Conic, P_X, P_Y);
int size = Conic.NumberOfElements();
if(size == 6)
{
Return_Vari.A = Conic.Get(1, 1);
Return_Vari.B = Conic.Get(1, 2);
Return_Vari.C = Conic.Get(1, 3);
Return_Vari.D = Conic.Get(1, 4);
Return_Vari.E = Conic.Get(1, 5);
Return_Vari.F = Conic.Get(1, 6);
}

else
{
Return_Vari.A = 0;
Return_Vari.B = 0;
Return_Vari.C = 0;
Return_Vari.D = 0;
Return_Vari.E = 0;
Return_Vari.F = 0;
}

if(Point_X) delete [] Point_X;
if(Point_Y) delete [] Point_Y;
return Return_Vari;
}
*/

Mod_S2DCoord Mod_CMatrix::Circlefitting(double &Radius, std::vector <Mod_S2DCoord> Circle_Point)
{
	double X_2, Y_2, XY, X, Y;
	double Variable[9], Desteny[4], Distance;
	Mod_S2DCoord Circle_Center;

	memset(Variable, 0, sizeof(double) * 9);
	memset(Desteny, 0, sizeof(double) * 3);

	for (int i = 0; i< (int)Circle_Point.size(); i++)
	{
		X_2 = pow(Circle_Point[i].x, 2);
		Y_2 = pow(Circle_Point[i].y, 2);
		XY = Circle_Point[i].x * Circle_Point[i].y;
		X = Circle_Point[i].x;
		Y = Circle_Point[i].y;

		Variable[0] += X_2;
		Variable[1] += XY;
		Variable[2] += X;
		Desteny[0] -= ((X_2 + Y_2) *X);

		Variable[3] += XY;
		Variable[4] += Y_2;
		Variable[5] += Y;
		Desteny[1] -= ((X_2 + Y_2) *Y);

		Variable[6] += X;
		Variable[7] += Y;
		Variable[8] += 1;
		Desteny[2] -= (X_2 + Y_2);
	}

	CvMat *SrcMat = cvCreateMat(3, 3, CV_64FC1);
	CvMat *InvertMat = cvCreateMat(3, 3, CV_64FC1);
	CvMat *VariMat = cvCreateMat(3, 1, CV_64FC1);
	CvMat *Desteny_Mat = cvCreateMat(3, 1, CV_64FC1);

	cvInitMatHeader(SrcMat, 3, 3, CV_64FC1, Variable);
	cvInitMatHeader(VariMat, 3, 1, CV_64FC1, Desteny);

	cvInvert(SrcMat, InvertMat, CV_LU);
	cvMatMulAdd(InvertMat, VariMat, 0, Desteny_Mat);

	Circle_Center.x = -cvmGet(Desteny_Mat, 0, 0) / 2;
	Circle_Center.y = -cvmGet(Desteny_Mat, 1, 0) / 2;
	Radius = -pow(Circle_Center.x, 2) - pow(Circle_Center.y, 2) + cvmGet(Desteny_Mat, 2, 0);

	if (Radius < 0)
	{
		Radius = sqrt(-Radius);

		int NumofErrorPoint = 0;
		double mean_Distance = 0;

		for (int i = 0; i< (int)Circle_Point.size(); i++)
		{
			Distance = cal_2Ddistance(Circle_Center, Circle_Point[i]) - Radius;
			mean_Distance += Distance;
			if (Distance > 0.5) NumofErrorPoint++;
		}

		mean_Distance /= (int)Circle_Point.size();
		if (mean_Distance > 0.5 || NumofErrorPoint > (int)Circle_Point.size() / 5)
		{
			Circle_Center.x = 0; Circle_Center.y = 0; Radius = 0;
		}
	}

	else { Circle_Center.x = 0; Circle_Center.y = 0; Radius = 0; }

	cvReleaseMat(&SrcMat);
	cvReleaseMat(&InvertMat);
	cvReleaseMat(&VariMat);
	cvReleaseMat(&Desteny_Mat);

	return Circle_Center;
}


void Mod_CMatrix::Display2D(Mod_SColor* Color)
{
	IplImage *Image = NULL;
	Image = cvCreateImage(cvSize(IMG_WIDTH, IMG_HEIGHT), IPL_DEPTH_8U, 3);
	for (int i = 0; i<IMG_HEIGHT*IMG_WIDTH; i++)
	{
		Image->imageData[3 * i] = char(Color[i].b);
		Image->imageData[3 * i + 1] = char(Color[i].g);
		Image->imageData[3 * i + 2] = char(Color[i].r);
	}

	cvNamedWindow("2D_Image", 1);
	cvShowImage("2D_Image", Image);
	if (Image) { cvReleaseImage(&Image), Image = NULL; }
}

int Mod_CMatrix::BoundaryClassification(bool* Exist_Point, bool* Boundary, int* Classcification)
{
	int k, Num, Num_Object;

	bool *temp_Boundary = new bool[IMG_HEIGHT*IMG_WIDTH];
	int *Map = new int[IMG_HEIGHT*IMG_WIDTH];

	std::vector <int> Number;

	k = 1;

	memcpy(temp_Boundary, Boundary, sizeof(bool)*IMG_HEIGHT*IMG_WIDTH);
	memset(Map, 0, sizeof(int)*IMG_HEIGHT*IMG_WIDTH);
	memset(Classcification, 0, sizeof(int)*IMG_HEIGHT*IMG_WIDTH);

	for (int j = 1; j<IMG_HEIGHT - 1; j++)
	{
		for (int i = 1; i<IMG_WIDTH - 1; i++)
		{
			int p_index = j*IMG_WIDTH + i;
			if (Boundary[p_index] == 1)
			{
				if (Exist_Point[p_index + 1] == 1) temp_Boundary[p_index + 1] = 1;
				else if (Exist_Point[p_index - 1] == 1) temp_Boundary[p_index - 1] = 1;
				else if (Exist_Point[p_index + IMG_WIDTH] == 1) temp_Boundary[p_index + IMG_WIDTH] = 1;
				else if (Exist_Point[p_index - IMG_WIDTH] == 1) temp_Boundary[p_index - IMG_WIDTH] = 1;
			}
		}
	}

	memcpy(Boundary, temp_Boundary, sizeof(bool)*IMG_HEIGHT*IMG_WIDTH);

	for (int j = 1; j<IMG_HEIGHT - 1; j++)
	{
		for (int i = 1; i<IMG_WIDTH - 1; i++)
		{
			int p_index = j*IMG_WIDTH + i;
			if (Boundary[p_index] == 1 && Exist_Point[p_index] == 1)
			{
				if (Map[p_index - 640] != 0 && Map[p_index - 1] == Map[p_index - 640])
				{
					Map[p_index] = Map[p_index - 640];
				}

				if (Map[p_index - 640] != 0 && Map[p_index - 1] != 0 && Map[p_index - 1] != Map[p_index - 640])
				{
					for (int m = 0; m < j*IMG_WIDTH + i; m++)
					{
						if (Map[m] == Map[p_index - 1])	Map[m] = Map[p_index - 640];
					}
					Map[p_index] = Map[p_index - 640];
				}

				if (Map[p_index - 640] != 0 && Map[p_index - 1] == 0)
				{
					Map[p_index] = Map[p_index - 640];
				}

				if (Map[p_index - 640] == 0 && Map[p_index - 1] != 0)
				{
					Map[p_index] = Map[p_index - 1];
				}

				if (Map[p_index - 640] == 0 && Map[p_index - 1] == 0)
				{
					Map[p_index] = k;
					k++;
				}
			}
		}
	}

	int *Voting = new int[k];

	memset(Voting, 0, sizeof(int)*k);

	for (int i = 0; i<IMG_HEIGHT*IMG_WIDTH; i++)
	{
		Num = Map[i];
		if (Num != 0) Voting[Num]++;
	}

	for (int i = 0; i<k; i++)
	{
		if (Voting[i] > 0)	Number.push_back(i);
	}

	k = 1;
	Num_Object = (int)Number.size();

	for (int i = 0; i< (int)Number.size(); i++)
	{
		for (int j = 0; j<IMG_HEIGHT*IMG_WIDTH; j++)
		{
			if (Number[i] == Map[j])
			{
				Classcification[j] = k;
			}
		}
		k++;
	}

	if (Number.size() >0) Number.clear();
	if (temp_Boundary) { delete[] temp_Boundary;	temp_Boundary = NULL; }
	if (Voting) { delete[] Voting;	Voting = NULL; }
	if (Map) { delete[] Map; Map = NULL; }
	return Num_Object;
}

/*
Mod_S2DCoord Mod_CMatrix::Circlefitting(double &Radius, std::vector <Mod_S2DCoord> Circle_Point)
{
int i,j;
int size;
double X,Y,R, Temp_X, Temp_Y, xAvr,	yAvr, LAvr, LaAvr, LbAvr, dx, dy, L;
double disperse;
const double tolerance = 1e-06;
Mod_S2DCoord Center_Point;

xAvr = 0;
yAvr = 0;

size = (int)Circle_Point.size();

for(i = 0; i < size; i++)
{
xAvr += Circle_Point[i].x;
yAvr += Circle_Point[i].y;
}

xAvr /= i;
yAvr /= i;

X = xAvr;
Y = yAvr;

for(i = 0; i< 1000; i++)
{
Temp_X = X;
Temp_Y = Y;

LAvr = 0;
LaAvr = 0;
LbAvr = 0;

for (j = 0; j < size; j++)
{
dx = Circle_Point[j].x - X;
dy = Circle_Point[j].y - Y;
L = sqrt(dx * dx + dy * dy);
if(L == 0) break;
LAvr += L;
LaAvr -= (dx / L);
LbAvr -= (dy / L);
}

if(L == 0) break;
LAvr  /= size;
LaAvr /= size;
LbAvr /= size;

X = xAvr + (LAvr * LaAvr);
Y = yAvr + (LAvr * LbAvr);
R = LAvr;
if (fabs(X - Temp_X) <= tolerance && fabs(Y - Temp_Y) <= tolerance)
break;
}

LAvr = 0;
if(L != 0)
{
for(i=0; i<size;i++)
{
dx = Circle_Point[i].x - X;
dy = Circle_Point[i].y - Y;
L = sqrt(dx * dx + dy * dy);
disperse = L-R;
if(disperse < 0) disperse = -disperse;
LAvr +=disperse;
}
}

LAvr /= size;

if(i != 1000 && LAvr < 0.01 && R < 0.5 && L != 0)
{
Radius = R;
Center_Point.x = X;
Center_Point.y = Y;
return Center_Point;
}
else
{
Radius = 0;
Center_Point.x = 0;
Center_Point.y = 0;
return Center_Point;
}
}
*/

Mod_CVector Mod_CMatrix::RoatedVector(double *r_mat, Mod_CVector vec)
{
	Mod_CVector temp_Vec;

	temp_Vec.X = vec.X * r_mat[0] + vec.Y * r_mat[1] + vec.Z * r_mat[2];
	temp_Vec.Y = vec.X * r_mat[3] + vec.Y * r_mat[4] + vec.Z * r_mat[5];
	temp_Vec.Z = vec.X * r_mat[6] + vec.Y * r_mat[7] + vec.Z * r_mat[8];

	return temp_Vec;
}

Mod_S3DCoord Mod_CMatrix::RoatedPoint(double *r_mat, Mod_S3DCoord point)
{
	Mod_S3DCoord temp_point;

	temp_point.x = point.x * r_mat[0] + point.y * r_mat[1] + point.z * r_mat[2];
	temp_point.y = point.x * r_mat[3] + point.y * r_mat[4] + point.z * r_mat[5];
	temp_point.z = point.x * r_mat[6] + point.y * r_mat[7] + point.z * r_mat[8];

	return temp_point;
}

Mod_S3DCoord Mod_CMatrix::TransPosedPoint(double *T_mat, Mod_S3DCoord point)
{
	Mod_S3DCoord temp_point;

	temp_point.x = point.x * T_mat[0] + point.y * T_mat[1] + point.z * T_mat[2] + T_mat[3];
	temp_point.y = point.x * T_mat[4] + point.y * T_mat[5] + point.z * T_mat[6] + T_mat[7];
	temp_point.z = point.x * T_mat[8] + point.y * T_mat[9] + point.z * T_mat[10] + T_mat[11];

	return temp_point;
}

Mod_CVector Mod_CMatrix::TransPosedVector(double *T_mat, Mod_CVector vec)
{
	Mod_CVector temp_Vec;

	temp_Vec.X = vec.X * T_mat[0] + vec.Y * T_mat[1] + vec.Z * T_mat[2];
	temp_Vec.Y = vec.X * T_mat[4] + vec.Y * T_mat[5] + vec.Z * T_mat[6];
	temp_Vec.Z = vec.X * T_mat[8] + vec.Y * T_mat[9] + vec.Z * T_mat[10];

	return temp_Vec;
}



Mod_S3DCoord Mod_CMatrix::RoatedPointAxisY(double Degree, Mod_S3DCoord Point)
{
	double R_Mat[9];
	double Angle;
	Mod_S3DCoord Temp_Point;

	memset(R_Mat, 0, 9 * sizeof(double));

	Angle = Degree / 57.295;

	R_Mat[0] = cos(Angle);  R_Mat[2] = sin(Angle); R_Mat[4] = 1;
	R_Mat[6] = -sin(Angle); R_Mat[8] = cos(Angle);

	Temp_Point.x = Point.x * R_Mat[0] + Point.y * R_Mat[1] + Point.z * R_Mat[2];
	Temp_Point.y = Point.x * R_Mat[3] + Point.y * R_Mat[4] + Point.z * R_Mat[5];
	Temp_Point.z = Point.x * R_Mat[6] + Point.y * R_Mat[7] + Point.z * R_Mat[8];

	return Temp_Point;
}

Mod_S3DCoord Mod_CMatrix::RoatedPointAxisZ(double Degree, Mod_S3DCoord Point)
{
	double R_Mat[9];
	double Angle;
	Mod_S3DCoord Temp_Point;

	memset(R_Mat, 0, 9 * sizeof(double));

	Angle = Degree / 57.295;

	R_Mat[0] = cos(Angle);	R_Mat[1] = -sin(Angle);	R_Mat[2] = 0;
	R_Mat[3] = sin(Angle);	R_Mat[4] = cos(Angle);	R_Mat[5] = 0;
	R_Mat[6] = 0;			R_Mat[7] = 0;			R_Mat[8] = 1;

	Temp_Point.x = Point.x * R_Mat[0] + Point.y * R_Mat[1] + Point.z * R_Mat[2];
	Temp_Point.y = Point.x * R_Mat[3] + Point.y * R_Mat[4] + Point.z * R_Mat[5];
	Temp_Point.z = Point.x * R_Mat[6] + Point.y * R_Mat[7] + Point.z * R_Mat[8];

	return Temp_Point;
}

Mod_CVector Mod_CMatrix::RoatedVecAxisX(double Degree, Mod_CVector vec)
{
	double r_mat[9];
	double angle;
	Mod_CVector temp_Vec;

	angle = Degree / 57.295;

	r_mat[0] = 1;	r_mat[1] = 0;			r_mat[2] = 0;
	r_mat[3] = 0;	r_mat[4] = cos(angle);	r_mat[5] = -sin(angle);
	r_mat[6] = 0;	r_mat[7] = sin(angle);	r_mat[8] = cos(angle);

	temp_Vec.X = vec.X * r_mat[0] + vec.Y * r_mat[1] + vec.Z * r_mat[2];
	temp_Vec.Y = vec.X * r_mat[3] + vec.Y * r_mat[4] + vec.Z * r_mat[5];
	temp_Vec.Z = vec.X * r_mat[6] + vec.Y * r_mat[7] + vec.Z * r_mat[8];

	return temp_Vec;
}

Mod_CVector Mod_CMatrix::RoatedVecAxisY(double Degree, Mod_CVector vec)
{
	double r_mat[9];
	double angle;
	Mod_CVector temp_Vec;

	angle = Degree / 57.295;

	r_mat[0] = cos(angle);	r_mat[1] = 0;	r_mat[2] = sin(angle);
	r_mat[3] = 0;			r_mat[4] = 1;	r_mat[5] = 0;
	r_mat[6] = -sin(angle); r_mat[7] = 0;	r_mat[8] = cos(angle);

	temp_Vec.X = vec.X * r_mat[0] + vec.Y * r_mat[1] + vec.Z * r_mat[2];
	temp_Vec.Y = vec.X * r_mat[3] + vec.Y * r_mat[4] + vec.Z * r_mat[5];
	temp_Vec.Z = vec.X * r_mat[6] + vec.Y * r_mat[7] + vec.Z * r_mat[8];

	return temp_Vec;
}

Mod_CVector Mod_CMatrix::RoatedVecAxisZ(double Degree, Mod_CVector vec)
{
	double r_mat[9];
	double angle;
	Mod_CVector temp_Vec;

	angle = Degree / 57.295;

	r_mat[0] = cos(angle);	r_mat[1] = -sin(angle);	r_mat[2] = 0;
	r_mat[3] = sin(angle);	r_mat[4] = cos(angle);	r_mat[5] = 0;
	r_mat[6] = 0;			r_mat[7] = 0;			r_mat[8] = 1;

	temp_Vec.X = vec.X * r_mat[0] + vec.Y * r_mat[1] + vec.Z * r_mat[2];
	temp_Vec.Y = vec.X * r_mat[3] + vec.Y * r_mat[4] + vec.Z * r_mat[5];
	temp_Vec.Z = vec.X * r_mat[6] + vec.Y * r_mat[7] + vec.Z * r_mat[8];

	return temp_Vec;
}

void Mod_CMatrix::Cal_ObjectRotation(Mod_CVector x_axis, Mod_CVector y_axis, Mod_CVector z_axis, double *r_mat)
{
	double x_180[9], y_180[9], temp_r[9], first_r[9], second_r[9];
	Mod_CVector temp_Vec;

	double theta_y;
	CalRotationMat(y_axis, temp_r);

	temp_Vec = RoatedVector(temp_r, y_axis);
	if (temp_Vec.Y < 0)
	{
		x_180[0] = 1; x_180[1] = 0; x_180[2] = 0;
		x_180[3] = 0; x_180[4] = -1; x_180[5] = 0;
		x_180[6] = 0; x_180[7] = 0; x_180[8] = -1;
		MutiplyMatrix3x3(x_180, temp_r, first_r);
	}
	else memcpy(first_r, temp_r, sizeof(double) * 9);

	temp_Vec = RoatedVector(first_r, z_axis);
	theta_y = atan(-temp_Vec.X / temp_Vec.Z);

	temp_r[0] = cos(theta_y); temp_r[1] = 0; temp_r[2] = sin(theta_y);
	temp_r[3] = 0; temp_r[4] = 1; temp_r[5] = 0;
	temp_r[6] = -sin(theta_y); temp_r[7] = 0; temp_r[8] = cos(theta_y);

	temp_Vec = RoatedVector(temp_r, temp_Vec);

	if (temp_Vec.Z < 0)
	{
		y_180[0] = -1; y_180[1] = 0; y_180[2] = 0;
		y_180[3] = 0; y_180[4] = 1; y_180[5] = 0;
		y_180[6] = 0; y_180[7] = 0; y_180[8] = -1;
		MutiplyMatrix3x3(y_180, temp_r, second_r);
	}
	else memcpy(second_r, temp_r, sizeof(double) * 9);

	MutiplyMatrix3x3(second_r, first_r, r_mat);
}

