// Gaurav Sheni
// CSC 391 
// December 2, 2015
// Project 4

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

//function declartion
void CUDAErrorCheck();
__global__ void normalized_freq(int* ,  int* , int* );
//a function I found online to check why large values 15 million plus did not run
//this told me it was due to allocating too much memory from curand states
//https://code.google.com/p/stanford-cs193g-sp2010/wiki TutorialWhenSomethingGoesWrong
void CUDAErrorCheck()
{
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
                printf("CUDA error : %s (%d)\n", cudaGetErrorString(error), error);
                //don't exit, lets just keep going
                //exit(0);
        }
}
__global__ void normalized_freq(int* digits,  int* freq_count, int* number_of_points){

	//get the global id because need to know which thread in
	int global_id = blockDim.x * blockIdx.x + threadIdx.x;

	//just to make sure that no extra calculation than what is required is done
	if (global_id >= (*number_of_points)){
		return;
	}

	int rounded_x = digits[global_id];
	printf ("Digits: %i\n", rounded_x);
	atomicAdd(&freq_count[rounded_x], 1);
}

int main ( int argc, char *argv[] ) {

	///check for correct # of arguments
	if (argc != 2){
		printf ("Incorrect number of command line arugments.\r\n");
		exit(1);
	}
		
	int number_of_points = 10;
	int half_number_of_points = number_of_points/2;

	int *first_half_digits = (int *)malloc(half_number_of_points * sizeof(int));
	int *second_half_digits = (int *)malloc(half_number_of_points * sizeof(int));
	int c = 0;
	FILE *file = fopen(argv[1], "r");

	int i = 0;
	int j = 0;
	while(c != EOF){
		c = fgetc(file);
		//below make sure its a digit
		if(('0' <= c) && (c <= '9')){
			//printf("%d\n", c - '0');
			if (i < half_number_of_points ){
				*(first_half_digits+i) = c - '0';
				i++;
			}
			else{
				*(second_half_digits+j)= c - '0';
				j++;
			}
		}
	}

	for(i = 0; i < half_number_of_points ; i++){
		printf("F: %i\n",*(first_half_digits+i));
	}
	for(i = 0; i < half_number_of_points ; i++){
		printf("S: %i\n",*(second_half_digits+i));
	}

	//to keep track of frequencies 
	//initliaze to 0 
	int first_freq_count[10] = { 0 };
	int second_freq_count[10] = { 0 };

	int device_sync_count = 0;

	int *dev_first_freq_count;
	int *dev_first_half_digits;
	int *dev_first_number_of_points;

	cudaSetDevice(0);
	CUDAErrorCheck();

	cudaMalloc((void**) &dev_first_freq_count, 10*sizeof(int));
	CUDAErrorCheck();
	cudaMalloc((void**) &dev_first_half_digits, half_number_of_points*sizeof(int));
	CUDAErrorCheck();
	cudaMalloc((void**) &dev_first_number_of_points, sizeof(int));
	CUDAErrorCheck();

	cudaMemcpy(dev_first_freq_count, &first_freq_count, 10*sizeof(int), cudaMemcpyHostToDevice);
	CUDAErrorCheck();
	cudaMemcpy(dev_first_half_digits, &first_half_digits, half_number_of_points*sizeof(int), cudaMemcpyHostToDevice);
	CUDAErrorCheck();
	cudaMemcpy(dev_first_number_of_points, &half_number_of_points, sizeof(int), cudaMemcpyHostToDevice);
	CUDAErrorCheck();

	normalized_freq<<<(int)ceil(half_number_of_points/1024)+1, 1024>>>(dev_first_half_digits,  dev_first_freq_count, dev_first_number_of_points);
	CUDAErrorCheck();

	//playing catch up.
	// cudaThreadSynchronize();
	cudaDeviceSynchronize();
	CUDAErrorCheck();
	printf ("GPUs Synchronized (%i).\r\n", device_sync_count);
	device_sync_count++;

	cudaMemcpy(first_freq_count, dev_first_freq_count, 10*sizeof(int), cudaMemcpyDeviceToHost);
	CUDAErrorCheck();

	cudaFree(dev_first_freq_count);
	cudaFree(dev_first_half_digits);
	cudaFree(dev_first_number_of_points);

	cudaSetDevice(1);
	CUDAErrorCheck();

	int *dev_second_freq_count;
	int *dev_second_half_digits;
	int *dev_second_number_of_points;

	cudaMalloc((void**) &dev_second_freq_count, 10*sizeof(int));
	cudaMalloc((void**) &dev_second_half_digits, half_number_of_points*sizeof(int));
	cudaMalloc((void**) &dev_second_number_of_points, sizeof(int));
	CUDAErrorCheck();

	cudaMemcpy(dev_second_freq_count, &second_freq_count, 10*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_second_half_digits, &second_half_digits, half_number_of_points*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_second_number_of_points, &half_number_of_points, sizeof(int), cudaMemcpyHostToDevice);
	CUDAErrorCheck();

	normalized_freq<<<(int)ceil(half_number_of_points/1024)+1, 1024>>>(dev_second_half_digits,  dev_second_freq_count, dev_second_number_of_points);

	//playing catch up
	//cudaThreadSynchronize();
	cudaDeviceSynchronize();
	printf ("GPUs Synchronized (%i).\r\n", device_sync_count);
	device_sync_count++;
	CUDAErrorCheck();

	cudaMemcpy(second_freq_count, dev_second_freq_count, 10*sizeof(int), cudaMemcpyDeviceToHost);
	CUDAErrorCheck();

	//free memory
	cudaFree(dev_first_freq_count);
	cudaFree(dev_first_half_digits);
	cudaFree(dev_second_number_of_points);

	CUDAErrorCheck();

	for(i = 0; i < (10) ; i++){
		printf("FFreq: %i\n", *(first_freq_count + i));
	}

	for(i = 0; i < (10) ; i++){
		printf("SFreq: %i\n",*(second_freq_count + i));
	}
	
	//open file to be read into
	FILE *file_output = fopen("freq.dat", "w");
	if (file_output == NULL) {
	    printf("File could not be created. ");
	    exit(1);
	}

	int total_freq = 0;
	for(i=0;i<10;i++){
		//index of 1 becomes 0.1, 2 becomes 0.2
		double output = (double) i / 10;

		float output_freq = (float) (first_freq_count[i]+second_freq_count[i]) / ((float)number_of_points * 2.00);

		fprintf(file_output, "%0.0f\t", output*10);
		fprintf(file_output, "%f\n", output_freq);

		// printf("Frequence at %i, is %0.0f\n", i, output_freq);

		// below is for debuggin purpooses and just checking 
		total_freq =+ output_freq;
	}
	printf("Total Freq = %i\n", total_freq);

	//close file output
	fclose(file_output);

	//exit the program, done
	exit(0);
}
