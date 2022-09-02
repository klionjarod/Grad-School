#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Jpegfile.h"
#include "mpi.h"


int Kmeans_MPI(int argc, char* argv[]) 
{
	//initialize needed variables before
	int id, p;

	//Initialize the MPI region
	MPI_Init(&argc, &argv);
	
	//Get the worker id and number of processes
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	int k = 4; //number of k-means
	int iterations = 1; //Number of iterations to run the algorithm
	double timeElapsed; //total time elapsed counter

	int numPixels, totalPixels;
	UINT height, width; //height and width of image
	BYTE* dataBuf = NULL; //dataBuf initialization for image
	BYTE* generators = new BYTE[k * 3]; //dynamic array of 3 colors (RGB) for each generator

	//Do STEP 1: INITIALIZATION entirely in Master process
	if (id == 0)
	{
		std::cout << "Reading the image..." << std::endl;
		dataBuf = JpegFile::JpegFileToRGB("bigimage.jpg", &width, &height);
		std::cout << "Finished reading the image" << std::endl;
		timeElapsed = MPI_Wtime();
		totalPixels = (int)height * (int)width; //total pixels in image
		numPixels = totalPixels / p; //number of pixels per process

		//choose k pixels to put into generators
		int chooser = totalPixels / k;
		BYTE *pRed, *pGrn, *pBlu;
		for (int i = 0; i < k; i++)
		{
			pRed = dataBuf + i * chooser * 3;
			pGrn = dataBuf + i * chooser * 3 + 1;
			pBlu = dataBuf + i * chooser * 3 + 2;
			generators[i * 3] = *pRed;
			generators[i * 3 + 1] = *pGrn;
			generators[i * 3 + 2] = *pBlu;
		}

	}

	//Broadcast the number of pixels to each worker
	std::cout << "BEFORE numpixels bcast" << std::endl;
	MPI_Bcast(&numPixels, 1, MPI_INT, 0, MPI_COMM_WORLD);
	std::cout << "AFTER numpixels bcast" << std::endl;

	BYTE* workerBuf = NULL;
	workerBuf = new BYTE[(numPixels) * 3]; //allocate buffer for pixels (RGB) sent to each worker

	//Scatter dataBuf into workerBuf (even load)
	std::cout << "BEFORE scatter" << std::endl;
	MPI_Scatter(dataBuf, (numPixels) * 3, MPI_BYTE, workerBuf, (numPixels) * 3, MPI_BYTE, 0, MPI_COMM_WORLD);
	std::cout << "AFTER scatter" << std::endl;

	
	//Start the iterations of k-means clustering
	for (int iters = 0; iters < iterations; iters++)
	{
		/////////////////////////////////////////////////////////////
		//COMBINE STEP 2: GROUPING and STEP 3: GET NEW GENERATORS
		/////////////////////////////////////////////////////////////

		//Broadcast the generators to the workers at beginning of each iteration
		std::cout << "BEFORE MPI BCAST" << std::endl;
		MPI_Bcast(generators, k * 3, MPI_INT, 0, MPI_COMM_WORLD);
		std::cout << "AFTER MPI BCAST" << std::endl;

		int* allClusterCount = new int[k] {}; //dynamic array for size of all clusters
		int* allClusterSum = new int[k * 3]{}; //dynamic array for sum of each color for eventual averaging

		//Initialize worker sums and counts at beginning of each iteration
		int* workerClusterCount = NULL;
		workerClusterCount = new int[k] {}; //initialize values to 0
		int* workerClusterSum = NULL;
		workerClusterSum = new int[k * 3]{}; //initialize values to 0 (3 for RGB)


		//loop over all the pixels in workerBuf
		for (int i = 0; i < (int)(numPixels); i++)
		{
			BYTE pRed = workerBuf[i * 3];
			BYTE pGrn = workerBuf[i * 3 + 1];
			BYTE pBlu = workerBuf[i * 3 + 2];

			double minDist = DBL_MAX;
			int minCluster; //closest old generator (L2)
			//loop over clusters
			for (int j = 0; j < k; j++)
			{
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

			workerClusterSum[minCluster * 3] += (int)(pRed);
			workerClusterSum[minCluster * 3 + 1] += (int)(pGrn);
			workerClusterSum[minCluster * 3 + 2] += (int)(pBlu);
			workerClusterCount[minCluster]++;
		}

		std::cout << "BEFORE MPI REDUCE" << std::endl;
		MPI_Reduce(workerClusterCount, allClusterCount, k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); //k clusters
		MPI_Reduce(workerClusterSum, allClusterSum, k * 3, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); //k clusters with 3 values each
		std::cout << "AFTER MPI REDUCE" << std::endl;



		//UPDATE GENERATORS
		if (id == 0)
		{
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
		}

		delete[] workerClusterCount;
		workerClusterCount = NULL;
		delete[] workerClusterSum;
		workerClusterSum = NULL;
		delete[] allClusterCount;
		allClusterCount = NULL;
		delete[] allClusterSum;
		allClusterSum = NULL;
	}

	std::cout << "Before FINISHING step" << std::endl;
	//STEP 5: FINISH in Master
	if (id == 0)
	{
		//Fill in the pixels with the color of its group's generator
		for (int i = 0; i < totalPixels; i++)
		{
			BYTE *pRed = &dataBuf[i * 3];
			BYTE *pGrn = &dataBuf[i * 3 + 1];
			BYTE *pBlu = &dataBuf[i * 3 + 2];

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
		std::cout << "Write the image..." << std::endl;
		JpegFile::RGBToJpegFile("bigimage_kmeans_MPI.jpg", dataBuf, width, height, true, 75);
		std::cout << "Finished!" << std::endl;
		delete dataBuf;
		std::cout << "Elapsed time: " << MPI_Wtime() - timeElapsed << std::endl;
	}

	MPI_Finalize();

	return 0;
}
