#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include "Jpegfile.h"
#include <omp.h>

int ColorToGray()
{
	UINT height;
	UINT width;
	BYTE *dataBuf;
	//read the file to dataBuf with RGB format
	dataBuf = JpegFile::JpegFileToRGB("bigimage.jpg", &width, &height);

	//the following code convert RGB to gray luminance.
	UINT row,col;

	omp_set_num_threads(8);
	double t1 = omp_get_wtime();
	#pragma omp parallel for private(col)
		//tid = omp_get_thread_num();
		//printf("i am thread %d\n", tid);
		//double t1 = omp_get_wtime();
		//#pragma omp for private(col)
		for (int row = 0; row < height; row++) {
			for (int col = 0; col < width; col++) {
				BYTE* pRed, * pGrn, * pBlu;
				pRed = dataBuf + row * width * 3 + col * 3;
				pGrn = dataBuf + row * width * 3 + col * 3 + 1;
				pBlu = dataBuf + row * width * 3 + col * 3 + 2;

				// luminance
				int lum = (int)(.299 * (double)(*pRed) + .587 * (double)(*pGrn) + .114 * (double)(*pBlu));

				*pRed = (BYTE)lum;
				*pGrn = (BYTE)lum;
				*pBlu = (BYTE)lum;

			}
		}
		//double t2 = omp_get_wtime();
		//double tot_sec = t2 - t1;
		//printf("took %f seconds\n", tot_sec);
	double t2 = omp_get_wtime();
	double tot_sec = t2 - t1;
	printf("took %f seconds\n", tot_sec);

	//write the gray luminance to another jpg file
	JpegFile::RGBToJpegFile("bigimagemono.jpg", dataBuf, width, height, true, 75);
	
	delete dataBuf;
	return 1;
}