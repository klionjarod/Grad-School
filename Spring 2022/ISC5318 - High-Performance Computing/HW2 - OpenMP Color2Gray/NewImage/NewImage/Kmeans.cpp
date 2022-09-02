#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include "Jpegfile.h"
#include <omp.h>
using namespace std;
#include <iostream>
#include <vector>
#include <chrono>


int KMeans()
{
	
	for (int thread: {1, 2, 4, 8})
	{
		omp_set_num_threads(thread);
		

		UINT height;
		UINT width;
		BYTE* dataBuf;
		//read the file to dataBuf with RGB format
		dataBuf = JpegFile::JpegFileToRGB("bigimage.jpg", &width, &height);

		double time_start = omp_get_wtime(); //do not count reading file
		//UINT row, col;

		/*******************************
		*                              *
		*    STEP 1: INITIALIZATION    *
		*                              *
		********************************/
		const int k = 40; //choose k pixels
		int N = width * height; //total pixels in the image
		int chooser = N / k; //chose one pixel every one of these
		int iterations = 20; //number of times to repeat step 2 and 3

		//create the k-1 generators in a 2d array of k rows with 3 values (red, green, blue)
		BYTE* pRed, * pGrn, * pBlu;
		//use native arrays since amount of data is known
		int generators[k][3]{};

		for (int i = 0; i < k; i++)
		{
			pRed = dataBuf + i * chooser * 3;
			pGrn = dataBuf + i * chooser * 3 + 1;
			pBlu = dataBuf + i * chooser * 3 + 2;
			generators[i][0] = (int)(*pRed);
			generators[i][1] = (int)(*pGrn);
			generators[i][2] = (int)(*pBlu);
		}

		// Create a 2D vector outside the main loop to eventually store local cluster data of final iteration
		vector<vector<int>> mainClusters(k);


		/*******************************
		*                              *
		* STEP 2: GROUPING THE PIXELS  *
		*                              *
		********************************/

		//Loop through the amount of iterations we want of k-means clustering
		for (int iters = 0; iters < iterations; iters++)
		{
			//Store the amount of pixels in each cluster in its own array outside the parallel region
			int clusterSizes[k]{ 0 };
			double clusterSums[k][3]{ 0.0, 0.0, 0.0 };

			#pragma omp parallel shared(iters, generators, k, clusterSizes, clusterSums, mainClusters) private(pRed, pBlu, pGrn) 
			{
				//Each thread has its own cluster
				vector<vector<int>> localClusters(k); //dont know the cluster size beforehand so need dynamic vectors

				int row;
				int col;
				#pragma omp for private(col) schedule(dynamic)
				//copying code from original file to get pixel color
				for (row = 0; row < height; row++)
				{
					for (col = 0; col < width; col++)
					{
						pRed = dataBuf + row * width * 3 + col * 3;
						pGrn = dataBuf + row * width * 3 + col * 3 + 1;
						pBlu = dataBuf + row * width * 3 + col * 3 + 2;
						int pixel[] = { (*pRed), (*pGrn), (*pBlu) };

						//now find closest (w.r.t L2 norm) generator color to current pixel color for each pixel
						double min = 10000000.0; //different colors to collect basically (higher = more fidelity)
						int minCluster = 0; //cluster to put pixel to
						for (int i = 0; i < k; i++)
						{
							double L2norm = 0;
							//compute the L2 norm
							for (int j = 0; j < 3; j++)
							{
								L2norm += (generators[i][j] - pixel[j]) * (generators[i][j] - pixel[j]);
							}
							//find and set the cluster
							if (L2norm < min)
							{
								min = L2norm;
								minCluster = i;
							}
						}
						//set the index of the pixel in the localcluster
						localClusters[minCluster].push_back(row * width + col);

					}
				}


				/*******************************
				*                              *
				*  STEP 3: GET NEW GENERATORS  *
				*                              *
				********************************/

				// make localSums of each localCluster color values
				double localSums[k][3]{ 0.0, 0.0, 0.0 };
				#pragma omp for
				for (int i = 0; i < k; i++)
				{
					for (int j = 0; j < localClusters[i].size(); j++)
					{
						pRed = dataBuf + localClusters[i][j] * 3;
						pGrn = dataBuf + localClusters[i][j] * 3 + 1;
						pBlu = dataBuf + localClusters[i][j] * 3 + 2;
						localSums[i][0] += (*pRed);
						localSums[i][1] += (*pGrn);
						localSums[i][2] += (*pBlu);
					}
				}

				//loop fission
				#pragma omp for
				for (int i = 0; i < k; i++) 
				{
					clusterSizes[i] += localClusters[i].size();
				}

				// sum the size of localClusters for averaging purposes and put into outside clusterSizes for eventual averaging
				// need critical method so no data gets lost to parallelization
				#pragma omp critical
				{
					for (int i = 0; i < k; i++)
					{
						for (int j = 0; j < 3; j++) {
							clusterSums[i][j] += localSums[i][j]; //put localSums into outside clusterSums
						}
						//fill in the outside mainClusters if it's the last iteration of k-means clustering
						if (iters == iterations - 1)
						{
							for (int j = 0; j < localClusters[i].size(); j++)
							{
								mainClusters[i].push_back(localClusters[i][j]);

							}
						}
					}
				}

				//set new generators as average of old cluster
				#pragma omp for
				for (int i = 0; i < k; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						generators[i][j] = clusterSums[i][j] / clusterSizes[i];
					}
				}


			}


		}

		/*******************************
		*                              *
		*      STEP 5: FINISHING       *
		*                              *
		********************************/

		//Fill in the pixels with the color of its group's generator
		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < mainClusters[i].size(); j++)
			{
				pRed = dataBuf + mainClusters[i][j] * 3;
				pGrn = dataBuf + mainClusters[i][j] * 3 + 1;
				pBlu = dataBuf + mainClusters[i][j] * 3 + 2;
				*pRed = (BYTE)generators[i][0];
				*pGrn = (BYTE)generators[i][1];
				*pBlu = (BYTE)generators[i][2];
			}
		}
		double time_end = omp_get_wtime(); //do not count writing file

		//write the new kmeans to another jpg file
		JpegFile::RGBToJpegFile("bigimage_kmeans.jpg", dataBuf, width, height, true, 75);

		delete dataBuf;

		cout << "Time elapsed (s): " << time_end - time_start << " for " << thread << " threads" << endl;
	}

	return 1;
}