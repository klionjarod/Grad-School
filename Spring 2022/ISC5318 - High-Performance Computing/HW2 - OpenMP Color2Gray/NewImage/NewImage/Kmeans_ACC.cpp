#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Jpegfile.h"



int main()
{


	int k = 4; //number of k-means
	int iterations = 1; //Number of iterations to run the algorithm
	double timeElapsed; //total time elapsed counter

	int totalPixels;
	UINT height, width; //height and width of image
	BYTE* dataBuf = NULL; //dataBuf initialization for image
	BYTE* generators = new BYTE[k * 3]; //dynamic array of 3 colors (RGB) for each generator

	//Do STEP 1: INITIALIZATION
	dataBuf = JpegFile::JpegFileToRGB("testcolor.jpg", &width, &height);
	totalPixels = (int)height * (int)width; //total pixels in image

	//choose k pixels to put into generators
	int chooser = totalPixels / k;
	BYTE* pRed, * pGrn, * pBlu;
	for (int i = 0; i < k; i++)
	{
		pRed = dataBuf + i * chooser * 3;
		pGrn = dataBuf + i * chooser * 3 + 1;
		pBlu = dataBuf + i * chooser * 3 + 2;
		generators[i * 3] = *pRed;
		generators[i * 3 + 1] = *pGrn;
		generators[i * 3 + 2] = *pBlu;
	}


	//Start the iterations of k-means clustering
	for (int iters = 0; iters < iterations; iters++)
	{
		/////////////////////////////////////////////////////////////
		//COMBINE STEP 2: GROUPING and STEP 3: GET NEW GENERATORS
		/////////////////////////////////////////////////////////////

		int* allClusterCount = new int[k] {}; //dynamic array for size of all clusters
		int* allClusterSum = new int[k * 3]{}; //dynamic array for sum of each color for eventual averaging


		//loop over all the pixels in dataBuf
		for (int i = 0; i < totalPixels; i++)
		{
			BYTE pRed = dataBuf[i * 3];
			BYTE pGrn = dataBuf[i * 3 + 1];
			BYTE pBlu = dataBuf[i * 3 + 2];

			double minDist = DBL_MAX;
			int minCluster; //closest old generator (L2)
			//loop over clusters
			for (int j = 0; j < k; j++)
			{
				double L2norm = 0;
				double L2dist = (pRed - generators[j * 3]) * (pRed - generators[j * 3]) +
					(pGrn - generators[j * 3 + 1]) * (pGrn - generators[j * 3 + 1]) +
					(pBlu - generators[j * 3 + 2]) * (pBlu - generators[j * 3 + 2]);
				//if the distance to the generator is the closest after iterating through all generators
				if (L2dist < minDist)
				{
					minDist = L2dist;
					minCluster = j;
				}
			}

			allClusterSum[minCluster * 3] += (int)(pRed);
			allClusterSum[minCluster * 3 + 1] += (int)(pGrn);
			allClusterSum[minCluster * 3 + 2] += (int)(pBlu);
			allClusterCount[minCluster]++;
		}


		//UPDATE GENERATORS
		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				//consider possible state of 0 cluster
				if (allClusterCount[i] != 0)
				{
					generators[i * 3 + j] = (BYTE)(allClusterSum[i * 3 + j] / allClusterCount[i]);
				}
			}
		}
		
		delete[] allClusterCount;
		allClusterCount = NULL;
		delete[] allClusterSum;
		allClusterSum = NULL;
	}

	//STEP 5: FINISH

	//Fill in the pixels with the color of its group's generator
	for (int i = 0; i < totalPixels; i++)
	{
		BYTE* pRed = &dataBuf[i * 3];
		BYTE* pGrn = &dataBuf[i * 3 + 1];
		BYTE* pBlu = &dataBuf[i * 3 + 2];

		double minDist = DBL_MAX;
		int minCluster; //closest old generator (L2)
		//loop over clusters
		for (int j = 0; j < k; j++)
		{
			double L2dist = (*pRed - generators[j * 3]) * (*pRed - generators[j * 3]) +
				(*pGrn - generators[j * 3 + 1]) * (*pGrn - generators[j * 3 + 1]) +
				(*pBlu - generators[j * 3 + 2]) * (*pBlu - generators[j * 3 + 2]);
			//if the distance to the generator is the closest after iterating through all generators
			if (L2dist < minDist)
			{
				minDist = L2dist;
				minCluster = j;
			}
		}
		*pRed = (BYTE)generators[minCluster * 3];
		*pGrn = (BYTE)generators[minCluster * 3 + 1];
		*pBlu = (BYTE)generators[minCluster * 3 + 2];
	}

	JpegFile::RGBToJpegFile("testcolor_kmeans_ACC.jpg", dataBuf, width, height, true, 75);
	delete dataBuf;
	std::cout << "Elapsed time: ";


	return 0;
}
