#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "Jpegfile.h"
#include <cmath>
#include <cuda_runtime.h>

#define BSZ 32      //Block width and height

__global__ void convolute(BYTE *databuf, BYTE *outbuf, int width, int height)
{
//	int weights[5][5] = { {1, 4, 6, 4, 1}, {4, 16, 24, 16, 4}, {6, 24, 36, 24, 6}, {4, 16, 24, 16, 4}, {1, 4, 6, 4, 1 }};				//gaussian kernel weights
	int weights[5][5] = { {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1 }};				//gaussian kernel weights
	__shared__ BYTE buffer[BSZ+4][BSZ+4][3];
	int i = threadIdx.x;
	int j = threadIdx.y;
	int I = blockIdx.x*BSZ+i;
	int J = blockIdx.y*BSZ+j;
	
	//initilize the shared buffer
	if (I < width && J <height) {   
		buffer[j+2][i+2][0] = databuf[(J*width+I)*3];
		buffer[j+2][i+2][1] = databuf[(J*width+I)*3+1];
		buffer[j+2][i+2][2] = databuf[(J*width+I)*3+2];
		int left = blockIdx.x * BSZ - 2;
		int right = blockIdx.x*BSZ + BSZ-1 + 2;
		int bottom = blockIdx.y*BSZ - 2;
		int top = blockIdx.y*BSZ + BSZ-1 + 2;
		if (i < 2) {   
			//update left ghost  
			if (left > 0) {
				//your code here
				
				
				//update left bottom ghosts
				if (j < 2) {
					if (bottom > 0) {
						//your code here
				
				
					}
				}
				//update left top ghosts
				if (j >= BSZ-2) {
					int jj = j - (BSZ-2);
					if (top < height) {
						//your code here
				
				
					}
				}
			}
		}
		if (i>=BSZ-2) {
			//update right ghost
			int ii = i - (BSZ-2);
			if (right < width) {
				//your code here
				
				
				
				//update right bottom ghosts
				if (j < 2) {
					if (bottom > 0) {
						//your code here
				
				
					}
				}
				//update right top ghosts
				if (j >= BSZ-2) {
					int jj = j - (BSZ-2);
					if (top < height) {
						//your code here
				
				
					}
				}
			}
		}
		if (j < 2) {   
			//update bottom ghost
			if (bottom > 0) {
				//your code here
				
				
			}
		}
		if (j >= BSZ-2) {
			//update top ghost
			int jj = j - (BSZ-2);
			if (top < height) {
				//your code here
				
				
			}
		}
	}
	
	__syncthreads();	
	
	if (I < width && J <height) {   
		int row, col;
		int totalr = 0, totalg = 0, totalb = 0;
		int totalw = 0;
		for (row = -2; row <= 2; row ++) {
			for (col = -2; col <= 2; col++) {
				int w = weights[row+2][col+2];
				totalr += buffer[j+row+2][i+col+2][0]*w;
				totalg += buffer[j+row+2][i+col+2][1]*w;
				totalb += buffer[j+row+2][i+col+2][2]*w;
				totalw += w;
			}
		}
		outbuf[(J*width+I)*3] = (BYTE)(totalr / totalw);
		outbuf[(J*width+I)*3+1] = (BYTE)(totalg / totalw);
		outbuf[(J*width+I)*3+2] = (BYTE)(totalb / totalw);
	}
}

int main()
{
	UINT height;
	UINT width;
	BYTE *dataBuf;
	//read the file to dataBuf with RGB format
	dataBuf = JpegFile::JpegFileToRGB("cappuccino.jpg", &width, &height);
	int d_size = 3*width*height*sizeof(BYTE);
	BYTE *outBuf = new BYTE[d_size];
	
	BYTE *d_databuf, *d_outbuf;
	cudaMalloc((void **)&d_databuf, d_size);
	cudaMalloc((void **)&d_outbuf, d_size);

	cudaMemcpy(d_databuf, dataBuf, d_size, cudaMemcpyHostToDevice);

	dim3 dimGrid((width+BSZ-1)/BSZ, (height+BSZ-1)/BSZ, 1);
	dim3 dimBlock(BSZ, BSZ, 1);
	convolute<<<dimGrid,dimBlock>>>(d_databuf, d_outbuf, width, height);


	cudaMemcpy(outBuf, d_outbuf, d_size, cudaMemcpyDeviceToHost);
	//write the blurred image to another jpg file
	JpegFile::RGBToJpegFile("cappuccino_blurred.jpg", outBuf, width, height, true, 75);

	cudaFree(d_databuf);
	cudaFree(d_outbuf);
	delete dataBuf;
	delete outBuf;
	return 0;
}
