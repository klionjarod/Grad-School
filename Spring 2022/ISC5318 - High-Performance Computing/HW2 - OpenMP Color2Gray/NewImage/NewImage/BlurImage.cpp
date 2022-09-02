#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include "Jpegfile.h"
using namespace std;
#include <corecrt_math_defines.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <omp.h>


int BlurImage()
{
		
	double pi = M_PI; //define pi
	double stdev = 3.0; 
	double stdevsq = stdev * stdev;
	double normalization = pi * stdevsq; //normalization factor for the gaussian
	double radius = 3.0; //disk of fixed radius 2

	//Store average color values in a dynamic 2D vector
	vector<vector<int>> redAvgs;
	vector<vector<int>> grnAvgs;
	vector<vector<int>> bluAvgs;

	
	//needed to read the file for the Jpeg code
	UINT height;
	UINT width;
	BYTE* dataBuf;
	//read the file to dataBuf with RGB format
	dataBuf = JpegFile::JpegFileToRGB("bigimage.jpg", &width, &height);


	/******************************************************************************************************************************************************
	* This section is preprocessing dedicated to finding the pixels in the circle and getting the weights 
	* Treat the center of the circle as the origin aka x=0, y=0
	* Calculate the distance of each pixel from this zero pixel and store in distances vector
	*******************************************************************************************************************************************************/
	int radInt = (int)radius; //typecast radius to int for easier calculations later
	int x = -radInt;	 // start the row at the top most pixel
	int y = 0;			 // start the column in the center pixel
	vector<int> xStart;
	vector<int> xEnd;
	vector<vector<double>> distances;

	// Because this is a circle, we only need to find borders of one quadrant and use symmetry to fill in the rest
	// This gets the 1st quadrant by starting at the top and moving right, checking the distance from 0 each iteration
	// if distance > radius, move back left and add that index into the xEnd vector, then move down a row and start cycle over
	// if distance < radius, start the cycle over immediately to find the end of the row
	while (x != 0)
	{
		y++;
		double length = sqrt(x * x + y * y);
		if (length > radius)
		{
			y--;
			xEnd.push_back(y);
			x++;
		}
	}

	xEnd.push_back(radInt); //End  of the row


	//fill in the remainder of xStart and xEnd using symmetry
	for (int i = 1; i <= radInt; i++)
	{
		xEnd.push_back(xEnd[radInt - i]); // -i from center to get same starting point as other quadrant
	}
	for (int i = 0; i <= radInt * 2; i++)
	{
		xStart.push_back(-xEnd[i]); // same distance from center but on the opposite side so negate the other array
	}

	//after filling in the pixel borders, fill in the distances
	double sum = 0.0;
	for (int row = 0; row <= radInt * 2; row++) //starting at top left row since we loop through diameter
	{
		distances.push_back({});
		for (int col = 0; col <= xEnd[row] - xStart[row]; col++) //again start at top left but columns in the border
		{
			distances[row].push_back(sqrt((radInt - row) * (radInt - row) + (xStart[row] + col) * (xStart[row] + col)));
			distances[row][col] = exp(-distances[row][col]/stdevsq) / normalization;
			sum += distances[row][col];
		}
	}

	// setup finished 
	/*********************************************************************************************************************************************/
	
	UINT row, col;

	//bias the averages with 0 vectors
	for (row = 0; row < height; row++) 
	{
		redAvgs.push_back({});
		grnAvgs.push_back({});
		bluAvgs.push_back({});
		for (col = 0; col < width; col++) 
		{
			redAvgs[(int)row].push_back(0);
			grnAvgs[(int)row].push_back(0);
			bluAvgs[(int)row].push_back(0);
		}

	}

	//define loop values outside of loop for omp practice
	int i, j;
	int threads[] = {1, 2, 4, 8};
	
	//loop through the different amount of threads instead of individually running the program 4 different times
	for (int k = 0; k < size(threads); k++)
	{
		double start_time = omp_get_wtime(); // starting time of program
		omp_set_num_threads(threads[k]);
		#pragma omp parallel for private(col, i, j)
		for (int row = 0; row < height; row++) {
			for (col = 0; col < width; col++) {
				// hold the averages of the three colors
				double rAvg = 0.0;
				double gAvg = 0.0;
				double bAvg = 0.0;

				// declare the border variables 
				int borderTop = 0;
				int borderBot = 0;
				vector<int> xStartBorder;
				vector<int> xEndBorder;
				// set values of new xStartBorder and xEndBorder to the originals from above
				for (i = 0; i < xStart.size(); i++) {
					xStartBorder.push_back(xStart[i]);
					xEndBorder.push_back(xEnd[i]);
				}

				// If needed, change the border variables
				// Tell the next loop to start at a higher row
				if ((int)row < radInt)
				{
					borderTop = radInt - row;
				}

				// Tell the next loop to stop at a lower row
				if ((int)row > height - radInt - 1)
				{
					borderBot = row - (height - radInt - 1);
				}

				// left boundary and change the indices of xStartBorder if they are outside boundary
				if ((int)col < radInt) 
				{
					for (i = 0; i < xStart.size(); i++) 
					{
						if (xStart[i] < -(int)col) 
						{
							xStartBorder[i] = -1 * col;
						}
					}
				}
				// right boundary and change the indices of xEndBorder if they end outside boundary
				if ((int)col > (int)width - radInt - 1) 
				{
					for (i = 0; i < xEnd.size(); i++) 
					{
						if (xEnd[i] > (int)width - (int)col - 1) 
						{
							xEndBorder[i] = (int)width - (int)col - 1;
						}
					}
				}

				BYTE* pRed, * pGrn, * pBlu;
				for (i = borderTop; i <= radInt * 2 - borderBot; i++) 
				{
					for (j = 0; j <= xEndBorder[i] - xStartBorder[i]; j++) 
					{
						// shift the row and columns according to the pixel in the desired circle
						// row is center of circle, so (i-radInt) is the +- indicating if we are above(+) or below(-) center
						// col is center of circle and xStartBorder[i] is 0 at most so j runs through the rows of the ball
						pRed = dataBuf + (row + i - radInt) * width * 3 + (col + xStartBorder[i] + j) * 3;
						pGrn = dataBuf + (row + i - radInt) * width * 3 + (col + xStartBorder[i] + j) * 3 + 1;
						pBlu = dataBuf + (row + i - radInt) * width * 3 + (col + xStartBorder[i] + j) * 3 + 2;

						// This adds new information to the average times the corresponding weight 
						rAvg += (double)(*pRed) * (distances[i][j]);
						gAvg += (double)(*pGrn) * (distances[i][j]);
						bAvg += (double)(*pBlu) * (distances[i][j]);
					}
				}
				// divide by the sum to get the weighted average, turn it to an int, then add it to the vector
				redAvgs[(int)row][(int)col] = (int)(rAvg / sum);
				grnAvgs[(int)row][(int)col] = (int)(gAvg / sum);
				bluAvgs[(int)row][(int)col] = (int)(bAvg / sum);
			}
		}

		#pragma omp parallel for 
		for (int row = 0; row < height; row++) 
		{
			for (int col = 0; col < width; col++) 
			{
				BYTE* pRed, * pGrn, * pBlu;
				pRed = dataBuf + row * width * 3 + col * 3;
				pGrn = dataBuf + row * width * 3 + col * 3 + 1;
				pBlu = dataBuf + row * width * 3 + col * 3 + 2;


				*pRed = (BYTE)redAvgs[(int)row][(int)col];
				*pGrn = (BYTE)grnAvgs[(int)row][(int)col];
				*pBlu = (BYTE)bluAvgs[(int)row][(int)col];

			}
		}

		//write the blur to another jpg file
		JpegFile::RGBToJpegFile("E:\\Users\\Jarod\\Stuff\\School Stuff\\Grad School\\Spring 2022\\ISC5318\\HW3\\bigimageblurred.jpg", dataBuf, width, height, true, 75);

		double end_time = omp_get_wtime();
		std::cout << threads[k] << " thread(s): " << (end_time - start_time) << " seconds elapsed" << endl;
	}
	delete dataBuf;
	return 1;
}