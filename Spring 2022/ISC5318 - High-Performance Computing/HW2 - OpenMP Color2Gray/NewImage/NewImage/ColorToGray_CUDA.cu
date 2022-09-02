#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <math.h>
#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include "Jpegfile.h"


__global__ void luminanceCalc(BYTE* d_dataBuf, int ND) 
{
    //using parallel threads and parallel blocks
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < ND; i += stride) 
    {
        BYTE* pRed, * pGrn, * pBlu;
        pRed = d_dataBuf + i * 3;
        pGrn = d_dataBuf + i * 3 + 1;
        pBlu = d_dataBuf + i * 3 + 2;

        int lum = (int)(.299 * (double)(*pRed) + .587 * (double)(*pGrn) + .114 * (double)(*pBlu));

        *pRed = (BYTE)lum;
        *pGrn = (BYTE)lum;
        *pBlu = (BYTE)lum;
    }
}

//#define BLOCKS (256*256)
//#define THREADS_PER_BLOCK 512

int C2A_CUDA() 
{

    //set even numbers of blocks and threads (2^experiment)
    int experiment = 10;
   

    for (int blockMult = 4; blockMult < experiment; blockMult++) 
    {
        printf("blocks: %d \n", (int)(pow(2, blockMult)));
        for (int threadMult = 4; threadMult < experiment; threadMult++) 
        {

            //create objects for timer report
            float elapsed = 0;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            int blocks = (int)(pow(2, blockMult)), threadsPerBlock = (int)(pow(2, threadMult));

            UINT height, width;
            BYTE* dataBuf, * d_dataBuf; //host and device copy of databuf, respectively

            //read in jpg image
            dataBuf = JpegFile::JpegFileToRGB("bigimage.jpg", &width, &height);
            int dataBufSize = width * height * 3 * (int)sizeof(BYTE); //3 for rgb

            int N = height * width;

            //actually start timer after reading image
            cudaEventRecord(start, 0);

            //allocate space for device copies
            cudaMalloc((void**)&d_dataBuf, dataBufSize);

            //copy inputs to the device
            cudaMemcpy(d_dataBuf, dataBuf, dataBufSize, cudaMemcpyHostToDevice);

            //launch the luminanceCalc function
            luminanceCalc<<<blockMult, threadsPerBlock>>>(d_dataBuf, N);

            //wait for GPU to finish before accessing on host
            cudaDeviceSynchronize();

            //copy new data from the device back to host
            cudaMemcpy(dataBuf, d_dataBuf, dataBufSize, cudaMemcpyDeviceToHost);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);

            //write the gray luminance to new jpg file
            JpegFile::RGBToJpegFile("bigimage_mono_cuda.jpg", dataBuf, width, height, true, 75);
            

            //cleanup 
            free(dataBuf);
            cudaFree(d_dataBuf);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);


            //print elapsed time out 
            printf("Thread: %d & the elapsed time in gpu was %.4f ms \n", (int)(pow(2, threadMult)), elapsed);

        }
        printf("\\\\ \n");
    }

    return 0;
}