#include <stdio.h>
#include <stdlib.h>
#include "Jpegfile.h"
#include <thread>         // std::thread, std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
//#include <time.h>


int main()
{	
	clock_t t;
	t = clock();

	UINT height;
	UINT width;
	BYTE *dataBuf;
	//read the file to dataBuf with RGB format
	dataBuf = JpegFile::JpegFileToRGB("testcolor.jpg", &width, &height);

	//the following code convert RGB to gray luminance.

	//create the threads for the rows
	#pragma acc kernels copy(dataBuf[0:height*width*3-1])
	{
	for (UINT row=0;row<height;row++) {
		for (UINT col=0;col<width;col++) {
			BYTE pRed, pGrn, pBlu;
			pRed = dataBuf[row * width * 3 + col * 3];
			pGrn = dataBuf[row * width * 3 + col * 3+1];
			pBlu = dataBuf[row * width * 3 + col * 3+2];
			// luminance
			int lum = (int)(.299 * (int)(pRed) + .587 * (int)(pGrn) + .114 * (int)(pBlu));

			dataBuf[row * width * 3 + col * 3] = (BYTE)lum;
			dataBuf[row * width * 3 + col * 3+1] = (BYTE)lum;
			dataBuf[row * width * 3 + col * 3+2] = (BYTE)lum;
			}
		}

	}
	
	t = clock() - t;
	printf("It took me %f seconds.\n", ((float)t)/CLOCKS_PER_SEC);

	//write the gray luminance to another jpg file
	JpegFile::RGBToJpegFile("testmono_ACC.jpg", dataBuf, width, height, true, 75);
	
	delete dataBuf;

	return 0;
}
