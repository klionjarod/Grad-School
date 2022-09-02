#include <mpi.h>
#include <iostream>
#include <fstream>
using namespace std;

int main(int argc, char* argv[]) {


	MPI_Init(&argc, &argv);
	int numProcesses, id; //number of processes and rank
	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	//discretize domain
	int M = 130, N = 130; //grid size
	double dx = 4.0 / (M - 1); //x variable step size
	double dy = 3.0 / (N - 1); //y variable step size
	double dxx = dx * dx, dyy = dy * dy; //denominators for finite difference
	double T = 20;//stop time
	double dt = 0.0001; //time step

	//actual function
	double* H = NULL;
	//bottom and center = 0
	H = new double[M * N]{ 0 };

	//initialize values for t=0 in master process
	if (id == 0)
	{
		//top edge = 1
		for (int j = 0; j < N; j++)
		{
			H[j] = 1;
		}

		//left and right edges = 1
		for (int i = 1; i < M - 1; i++)
		{
			H[i * N] = 1;
			H[i * N + N - 1] = 1;
		}
	}

	//check grid looks okay
	//if (id == 0) {
	//	cout << "initial grid" << endl;
	//	for (int i = 0; i < M * N; i++) {
	//		cout << H[i] << " ";
	//	}
	//	cout << endl;
	//}

	//start timing
	double timeElapsed = MPI_Wtime();

	//calculate size of workload
	int workerCount = M / numProcesses;
	//leftover workload for master
	int masterCount = workerCount + M % numProcesses;
	int count;

	//check if only one process so Master gets full workload
	if (numProcesses == 1)
	{
		count = masterCount;
	}
	//otherwise master (top) process gets extra row of overlap to communicate
	else if (id == 0) {
		count = masterCount + 1;
	}
	//last process also gets one row of overlap
	else if (id == numProcesses - 1) 
	{
		count = workerCount + 1;
	}
	//every process that isn't first or last gets 2 rows of overlap
	else
	{
		count = workerCount + 2;
	}
	
	//need buffer size for H at current and previous time
	double* Hbuf = new double[count * N];
	double* Hbuf_prev = new double[count * N];

	//set up send/receive sizes and displacements for scatterv and gatherv operations
	int* displacements = new int[numProcesses] {0};	//initialize with no displacement (master value)
	int* srCounts = new int[numProcesses] {(masterCount + 1) * N}; //initialize with Master values
	for (int i = 1; i < numProcesses; i++) 
	{
		//all processes not master or last need 2 overlap
		if (i != numProcesses - 1)
		{
			srCounts[i] = (workerCount + 2) * N;
		}
		//bottom process
		else 
		{
			srCounts[i] = (workerCount+ 1) * N;
		}

		if (id == 1) //first process data starts with last master row
		{
			displacements[i] = (masterCount - 1) * N;
		}
		else //each other worker process starts from previous displacement + its workload size
		{
			displacements[i] = displacements[i - 1] + workerCount * N;
		}
	}

	//scatter H values to each worker
	MPI_Scatterv(H, srCounts, displacements, MPI_DOUBLE, Hbuf, srCounts[id], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	for (double t = 0; t <= T;)
	{
		//start by updating Hbuf_prev
		for (int i = 0; i < srCounts[id]; i++)
		{
			Hbuf_prev[i] = Hbuf[i];
		}

		//update step for equation using explicit method
		for (int i = 1; i < count - 1; i++) //rows per worker
		{
			for (int j = 1; j < N - 1; j++) //columns 
			{
				//indexing 2D x-y grid converted to 1d for array
				Hbuf[i * N + j] = Hbuf_prev[i * N + j] 
								+ dt / dxx * (Hbuf_prev[(i - 1) * N + j] - 2 * Hbuf_prev[i * N + j] + Hbuf_prev[(i + 1) * N + j])
								+ dt / dyy * (Hbuf_prev[i * N + (j - 1)] - 2 * Hbuf_prev[i * N + j] + Hbuf_prev[i * N + (j + 1)]);

			}
		}
	
		//exchange border information between lower and upper neighbors
		if (numProcesses > 1) //only need to communicate if more than one process
		{
			//determine each worker's neighbors
			int bottom, top;
			if (id < numProcesses - 1) {
				bottom = id + 1;
			} else { bottom = MPI_PROC_NULL; }
			if (id > 0) {
				top = id - 1;
			}
			else { top = MPI_PROC_NULL; }
			//talk to neighbor below
			MPI_Sendrecv(&Hbuf[(count - 2) * N], N, MPI_DOUBLE, bottom, 0, &Hbuf[(count - 1) * N], N, MPI_DOUBLE, bottom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			//talk to neighbor above
			MPI_Sendrecv(&Hbuf[N], N, MPI_DOUBLE, top, 0, Hbuf, N, MPI_DOUBLE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		t += dt; //increase time by time step
	}



	//gather all the data afterwards
	MPI_Gatherv(Hbuf, srCounts[id], MPI_DOUBLE, H, srCounts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//free up dynamic memory after using
	delete[] displacements;
	delete[] srCounts;
	delete[] Hbuf;
	delete[] Hbuf_prev;

	//stop timer
	timeElapsed = MPI_Wtime() - timeElapsed;

	if (id == 0)
	{

		cout << endl << "Time Elapsed: " << timeElapsed << "s" << endl;

		cout << "start writing to file" << endl;
		ofstream file;
		file.open("H_values.txt");
		for (int i = 0; i < M * N; i++)
		{

			file << H[i] << "\t";
			if ((i + 1) % N == 0)
			{
				file << "\n";
			}
		}
		file.close();
		cout << "writing finished" << endl;
	}

	MPI_Finalize();
	return 0;
}