#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include "Jpegfile.h"
#include "mpi.h"

int ColorToGray_MPI(int argc, char *argv[])
{
	//initialize needed variables beforehand
	int id, p, totalPixels, numPixels;
	double wtime;

	// Initialize the MPI region
	MPI_Init(&argc, &argv);
	//Get the id of the worker
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	//Determine the number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	UINT height;
	UINT width;
	BYTE* dataBuf = NULL;

	//The master reads the file and number of pixels
	if (id == 0) {
		//STEP 1: read the file to dataBuf with RGB format
		dataBuf = JpegFile::JpegFileToRGB("bigimage.jpg", &width, &height);

		//start time count after reading file
		wtime = MPI_Wtime();
		//STEP 2:
		//Calculate the total number of pixels
		totalPixels = (int)width * (int)height;
		//Divide the image into several parts based on number of processes
		numPixels = totalPixels / p;
	}

	//STEP 3: Give the number of pixels to each worker
	MPI_Bcast(&numPixels, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//STEP 4: allocate memory to each worker to store color of pixels
	BYTE* workerBuf = NULL;
	workerBuf = new BYTE[numPixels * 3]; //3 color values for each pixel

	//STEP 5: Master scatters the pixel colors to the workerBuf
	MPI_Scatter(dataBuf, numPixels * 3, MPI_BYTE, workerBuf, numPixels * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

	//STEP 6 WORKER: the following code converts RGB to gray luminance for each worker.
	for (int i = 0; i < numPixels; i++) {
		BYTE pRed = workerBuf[3 * i];
		BYTE pGrn = workerBuf[3 * i + 1];
		BYTE pBlu = workerBuf[3 * i + 2];

		// luminance
		int lum = (int)(.299 * (double)(pRed) + .587 * (double)(pGrn) + .114 * (double)(pBlu));

		//set each workers buffer values to luminance
		workerBuf[3 * i] = lum;
		workerBuf[3 * i + 1] = lum;
		workerBuf[3 * i + 2] = lum;

	}
	
	//STEP 6 MASTER: Master finishes the residual part
	if (id == 0) {
		for (int i = numPixels * p; i < totalPixels; i++) {
			BYTE* pRed, * pGrn, * pBlu;

			pRed = dataBuf + i * 3;
			pGrn = dataBuf + i * 3 + 1;
			pBlu = dataBuf + i * 3 + 2;

			// luminance
			int lum = (int)(.299 * (double)(*pRed) + .587 * (double)(*pGrn) + .114 * (double)(*pBlu));

			*pRed = (BYTE)lum;
			*pGrn = (BYTE)lum;
			*pBlu = (BYTE)lum;

		}
	}
	
	//STEP 7: Gather the converted pixel colors back to Master
	MPI_Gather(workerBuf, numPixels * 3, MPI_BYTE, dataBuf, numPixels * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

	//delete workerBuf memory to prevent invalid memory references
	delete[] workerBuf;
	workerBuf = NULL;


	//STEP 8: Master writes the grayscale image to a jpg file
	if (id == 0) {
		printf("Elapsed time: %f\n", MPI_Wtime() - wtime);
		//write the gray luminance to another jpg file
		JpegFile::RGBToJpegFile("bigimagemono_MPI.jpg", dataBuf, width, height, true, 75);
	}
	delete dataBuf;

	MPI_Finalize();
	return 1;
}