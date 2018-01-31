#include "stdafx.h"
#include<cmath>
#include<iostream>
#include<omp.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;


int Gx(Mat imageData, int x, int y)
{
	//get the Gx value with cconvolution applying the matrix [+1,0,-1;+2,0,-2;+1,0,-1]
	int gx = imageData.at<uchar>(y - 1, x - 1) +
		2 * imageData.at<uchar>(y, x - 1) +
		imageData.at<uchar>(y + 1, x - 1) -
		imageData.at<uchar>(y - 1, x + 1) -
		2 * imageData.at<uchar>(y, x + 1) -
		imageData.at<uchar>(y + 1, x + 1);

	//return the value
	return gx;
}

int Gy(Mat imageData, int x, int y)
{
	//get the Gy value with convolution applying the matrix [+1,+2,+1;0,0,0;+1,-1,+1]
	int gy = imageData.at<uchar>(y - 1, x - 1) +
		2 * imageData.at<uchar>(y - 1, x) +
		imageData.at<uchar>(y - 1, x + 1) -
		imageData.at<uchar>(y + 1, x - 1) -
		2 * imageData.at<uchar>(y + 1, x) -
		imageData.at<uchar>(y + 1, x + 1);

	//return the value
	return gy;
}

int main()
{
	//using Mat because is easy to use
	Mat sourceImage, greyImage, ouputImage;
	double startTime, endTime;
	int gx, gy, g;

	//the entire path of the image
	String imagePath = "C:/Users/xavi.portilla/Desktop/sobel/sobel_copia.png";

	//get the image
	sourceImage = imread(imagePath);

	//transform the RGB pixel matrix into grayScale matrix to get a better result
	cvtColor(sourceImage, greyImage, CV_BGR2GRAY);

	//in order to reuse the matrix, we clone the grey one into the output image
	ouputImage = greyImage.clone();

	//if there is no daata, return an error
	if (!greyImage.data)
	{
		//return error
		return -1;
	}

	//intialize the outputMatrix to 0
	#pragma omp parallel for  
	for (int y = 0; y < greyImage.rows; y++)
		for (int x = 0; x < greyImage.cols; x++)
			//set all values to zero
			ouputImage.at<uchar>(y, x) = 0;
	//get the start time to know how long it takes
	startTime = omp_get_wtime();

	//Sobel Operator, parallel for to calculate over the whole matrix
	#pragma omp parallel for private (gx, gy, sum) //num_threads(128)
	for (int y = 1; y < greyImage.rows - 1; y++) {
		for (int x = 1; x < greyImage.cols - 1; x++) {
			//convolution of the kernel with matrix [-1, 0, 1; -2, 0, 2; -1, 0, 1] in order to obtain Gx
			gx = Gx(greyImage, x, y);
			//convolution of the kernel with matrix [-1, -2, -1; 0, 0, 0; 1, 2, 1] in order to obtain Gy
			gy = Gy(greyImage, x, y);

			//to obtain G we have to calculate the square root of the Gx squared + Gy squared 	
			g = abs(gx) + abs(gy);

			//normalize values
			g = g > 255 ? 255 : g;
			g = g < 0 ? 0 : g;

			//set the G value
			ouputImage.at<uchar>(y, x) = g;
		}
	}
	//get the start time to know how long it takes
	endTime = omp_get_wtime();

	//create a window to show the result with openCV functions
	namedWindow("sobel");
	imshow("sobel", ouputImage);
	
	//show the time of the algorithm
	cout << "Time of the algorithm: " << (endTime - startTime) << " s." << endl;

	//waiting the user until he presses a key to end the execution
	waitKey();

	//return no errors
	return 0;
}
