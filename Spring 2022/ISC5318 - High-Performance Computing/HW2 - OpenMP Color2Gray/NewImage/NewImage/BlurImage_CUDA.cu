#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include "Jpegfile.h"

#include <corecrt_math_defines.h>
#include <math.h>




__global__ void averageCalc(BYTE* dataBufD, BYTE* dataAvD, double* distancesD, int* distancesRowStartIndexD, int* rowStartD, int* rowEndD, UINT height, UINT width, int rInt) {

	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;
	//int stride = blockDim.x * gridDim.x;
	int stride = blockDim.x * gridDim.x + blockDim.y * gridDim.y;
	int const rInt21 = rInt * 2 + 1;
	int ND = (int)(height * width);

	for (int pixel = index_x + (index_y*height); pixel < ND; pixel += stride) {
		//borders + row and col of the pixel this thread is going to handle
		int borderTop = 0, borderBot = 0, row = (int)(pixel / (int)width);
		int col = pixel - row * (int)width;
		int rowStartBord[11], rowEndBord[11];
		for (int i = 0; i < rInt21; i++) {
			rowStartBord[i] = rowStartD[i];
			rowEndBord[i] = rowEndD[i];
		}


		//averages of the three colors
		double rAvg = 0.0, gAvg = 0.0, bAvg = 0.0;

		BYTE* pRed, * pGrn, * pBlu;
		double sum = 0.0;
		for (int i = borderTop; i <= rInt * 2 - borderBot; i++) {
			for (int j = 0; j <= rowEndBord[i] - rowStartBord[i]; j++) {
				if ((row + i - rInt) * (int)width + (col + rowStartBord[i] + j) >= 0 && (row + i - rInt) * (int)width + (col + rowStartBord[i] + j) < ND) {
					//must shift the row and columns according to the pixel in the ball we want
					//row is at center of ball then (i-rInt) is the +-indicating if we are above the center(-) or below(+)
					//col is at center of ball and rowStartBord[i] is 0 at most so j runs throught the row of the ball
					pRed = dataBufD + (row + i - rInt) * (int)width * 3 + (col + rowStartBord[i] + j) * 3;
					pGrn = dataBufD + (row + i - rInt) * (int)width * 3 + (col + rowStartBord[i] + j) * 3 + 1;
					pBlu = dataBufD + (row + i - rInt) * (int)width * 3 + (col + rowStartBord[i] + j) * 3 + 2;

					//add new information to the average multiplied by the corresponding weight 
					//distanceRowStartIndex starts you at the first spot of the this row on the stencil, 
					//but if we are on the left border we can't start the leftmost but however more to the right which is rowStartBord - rowStart
					//j is the increment for moving through the row 
					double distance = distancesD[distancesRowStartIndexD[i] + rowStartBord[i] - rowStartD[i] + j];
					sum += distance;
					rAvg += ((double)(*pRed)) * distance;
					gAvg += ((double)(*pGrn)) * distance;
					bAvg += ((double)(*pBlu)) * distance;
				}
			}
		}

		pRed = dataAvD + pixel * 3, pGrn = dataAvD + pixel * 3 + 1, pBlu = dataAvD + pixel * 3 + 2;

		//divide by the sum to get the weighted average, typcast int and add it to the vector
		*pRed = (BYTE)(rAvg / sum), * pGrn = (BYTE)(gAvg / sum), * pBlu = (BYTE)(bAvg / sum);

		free(rowStartBord);
		free(rowEndBord);
	}

}


int Blur_CUDA() {

	int experiment = 9;

	for (int blockMult = 4; blockMult < experiment; blockMult++) {
		printf("%d Blocks \n", (int)(pow(2, blockMult)));
		for (int threadMult = 4; threadMult < experiment; threadMult++) {


			int blocks = (int)(pow(2, blockMult)), threadsPerBlock = (int)(pow(2, threadMult));

			double pi = M_PI;

			double radius = 5.0;
			double std = 5.0;
			double stdSq = 2 * std * std;
			double bottom = pi * stdSq;		//standardization for gaussian

			//pre-processing to get distances/weights of gaussian distribution into a vector of vectors
			//find the border of our ball and then calculate the distances of each pixel in our ball from our zero pixel
			//treat the center of the ball as the the origin so roww=0 and coll=0. 
			//RowStart and RowEnd are inclusive so rowStart[2] = -2 and rowEnd[2] = 2; the row has 5 elements with index (-2,-1,0,1,2)
			int rInt = (int)radius;
			int rInt21 = rInt * 2 + 1;
			int* rowStart = (int*)malloc(rInt21 * sizeof(int)), * rowStartD;
			int* rowEnd = (int*)malloc(rInt21 * sizeof(int)), * rowEndD;
			int roww = -1 * rInt;	//row of the top most pixel of ball
			int coll = 0;			//column of the top most pixel of ball
			double* distances, *distancesD;


			//create objects for timer report
			float elapsed = 0;
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);



			//get the 2nd quadrant of the border and use symmetry to get the rest of the border
			// 
			//start at top and find the border by moving left, measuring the distance from center and
			//if the distance is larger than the radius, we move back right, put that index into the our rowStart indices vector, and move down to start this over
			//if it is smaller, then we start the while loop over again to find the start of this row
			while (roww != 0) {
				coll--;
				double length = sqrt(coll * coll + roww * roww);
				if (length > radius) {
					coll++;
					rowStart[rInt + roww] = coll;
					roww++;
				}
			}
			rowStart[rInt] = -1 * rInt; //middle row start

			//fill in rowStart and rowEnd using symmetry
			// the +-i from center will have same starting point
			for (int i = 1; i <= rInt; i++) { rowStart[rInt + i] = rowStart[rInt - i]; }
			// the ends are same distance from center but on other side
			for (int i = 0; i < rInt21; i++) { rowEnd[i] = -1 * rowStart[i]; }

			//allocate memory in the GPU and copt over rowStart and rowEnd into that memory
			cudaMalloc((void**)&rowStartD, rInt21 * sizeof(int));
			cudaMalloc((void**)&rowEndD, rInt21 * sizeof(int));
			cudaMemcpy(rowStartD, rowStart, rInt21 * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(rowEndD, rowEnd, rInt21 * sizeof(int), cudaMemcpyHostToDevice);


			//allocate memory for distances and *distancesRowStartIndex
			int distancesSize = 0;
			int* distancesRowStartIndex = (int*)malloc(rInt21 * sizeof(int)), * distancesRowStartIndexD;
			for (int i = 0; i < (rInt * 2 + 1); i++) {
				distancesRowStartIndex[i] = distancesSize;
				distancesSize += rowEnd[i] * 2 + 1;
			}
			distances = (double*)malloc((distancesSize) * sizeof(double));


			//fill in the distances data structure by going through the rows and calculating the distance and storing it
			double sum = 0.0;
			for (int i = 0; i <= rInt * 2; i++) {
				for (int j = 0; j < rowEnd[i] - rowStart[i]; j++) {	//actual distance from center
					distances[distancesRowStartIndex[i] + j] = exp(-1 * sqrt((rInt - i) * (rInt - i) + (rowStart[i] + j) * (rowStart[i] + j)) / stdSq) / bottom;
					sum = sum + distances[distancesRowStartIndex[i] + j];
				}
			}


			//free up memory in the GPU and copy over distances and distancesRowStartIndex
			cudaMalloc((void**)&distancesRowStartIndexD, rInt21 * sizeof(int));
			cudaMalloc((void**)&distancesD, distancesSize * sizeof(double));
			cudaMemcpy(distancesRowStartIndexD, distancesRowStartIndex, rInt21 * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(distancesD, distances, distancesSize * sizeof(double), cudaMemcpyHostToDevice);



			//image varaibles and reading
			UINT height, width;
			BYTE* dataBuf, * dataBufD, * dataAvD;
			dataBuf = JpegFile::JpegFileToRGB("bigimage.jpg", &width, &height);

			//actually start timer after reading image
			cudaEventRecord(start, 0);

			//get databuffer size for image with rgb channels for memory allocation
			int sizeDataBuf = width * height * 3 * (int)sizeof(BYTE);
			int N = height * width;

			//allocate space for device copies and copy them to device
			BYTE* dataAv = (BYTE*)malloc(sizeDataBuf);
			for (int i = 0; i < 3 * N; i++) { dataAv[i] = 0; }
			cudaMalloc((void**)&dataBufD, sizeDataBuf);
			cudaMalloc((void**)&dataAvD, sizeDataBuf);
			cudaMemcpy(dataBufD, dataBuf, sizeDataBuf, cudaMemcpyHostToDevice);
			cudaMemcpy(dataAvD, dataAv, sizeDataBuf, cudaMemcpyHostToDevice);


			averageCalc<<<blocks, threadsPerBlock>>>(dataBufD, dataAvD, distancesD, distancesRowStartIndexD, rowStartD, rowEndD, height, width, rInt);
			//averageCalc<<<1, 512>>>(dataBufD, dataAvD, distancesD, distancesRowStartIndexD, rowStartD, rowEndD, height, width, rInt);


			//wait for GPU to finish before accessing on host
			cudaDeviceSynchronize();

			//copy new data from the device back to host
			cudaMemcpy(dataBuf, dataAvD, sizeDataBuf, cudaMemcpyDeviceToHost);
			cudaMemcpy(rowStart, rowStartD, rInt21 * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(rowEnd, rowEndD, rInt21 * sizeof(int), cudaMemcpyDeviceToHost);

			//stop timer before writing image
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsed, start, stop);

			//write the blur to another jpg file
			JpegFile::RGBToJpegFile("bigimageblurred_CUDA.jpg", dataBuf, width, height, true, 100);

			//print elapsed time out 
			printf("%d threads: the elapsed time in gpu was %.4f ms \n", (int)(pow(2, threadMult)), elapsed);

			//cleanup 
			free(dataBuf);
			cudaFree(dataBufD);
			cudaFree(dataAvD);

			free(distances);
			cudaFree(distancesD);
			free(distancesRowStartIndex);
			cudaFree(distancesRowStartIndexD);

			free(rowStart);
			cudaFree(rowStartD);
			free(rowEnd);
			cudaFree(rowEndD);

			cudaEventDestroy(start);
			cudaEventDestroy(stop);


		}
		printf("\\\\ \n");
	}

	return 0;
}
