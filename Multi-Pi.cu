
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

__global__ void first_call();
__global__ void normalized_freq(char* digits, int* global_mem_freq_count, cudaStream_t A);
void CUDAErrorCheck();

//number of digits to read each time
#define number_of_digits 5000000

int main ( int argc, char *argv[] ) {

	cudaSetDevice(0);
	first_call<<<1,1>>>();
	cudaSetDevice(1);
	first_call<<<1,1>>>();

	clock_t start; // for starting
	clock_t stop; //for stoping
	double execution_time;	//total time 
	start = clock();	//ready set go

	///check for correct # of arguments
	if (argc != 2){
		printf ("Incorrect number of command line arugments.\r\n");
		exit(1);
	}

	FILE *file_read = fopen(argv[1], "r+");
	if (file_read == NULL) {
	    printf("File could not be read. ");
	    exit(1);
	}

	int A_freq_count[10] = {0};
	int B_freq_count[10] = {0};
	char* A_digits = (char*)malloc(sizeof(char) * number_of_digits);

	int device_sync_count = 1;

	int *dev_A_freq_count;
	char *dev_A_digits;
	int *dev_B_freq_count;
	char *dev_B_digits;

	int number_of_freq_count = 10;

	cudaSetDevice(0);
	cudaStream_t stream1;
	CUDAErrorCheck();
	cudaStreamCreate(&stream1);
	CUDAErrorCheck();
	cudaHostAlloc((void**) &dev_A_freq_count, number_of_freq_count*sizeof(int), cudaHostAllocDefault);
	CUDAErrorCheck();
	cudaHostAlloc((void**) &dev_A_digits, number_of_digits*sizeof(char), cudaHostAllocDefault);
	CUDAErrorCheck();
	cudaMemcpyAsync(dev_A_freq_count, &A_freq_count, number_of_freq_count * sizeof(int), cudaMemcpyHostToDevice, stream1);
	CUDAErrorCheck();

	cudaSetDevice(1);
	cudaStream_t stream2;
	CUDAErrorCheck();
	cudaStreamCreate(&stream2);
	cudaHostAlloc((void**) &dev_B_freq_count, number_of_freq_count*sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**) &dev_B_digits, number_of_digits*sizeof(char), cudaHostAllocDefault);
	cudaMemcpyAsync(dev_B_freq_count, &B_freq_count, number_of_freq_count * sizeof(int), cudaMemcpyHostToDevice, stream2);
	CUDAErrorCheck();

	fgetc(file_read);

	while(fgets(A_digits, number_of_digits + 1, file_read) != NULL ) {
		// printf("Input: %s\n\n", A_digits);
		int len = strlen(A_digits);
		// printf("Size %i\n", len);
		strcat(A_digits, "---------------------------");
		// printf("Input %s\n", A_digits);
		cudaSetDevice(0);
		cudaMemcpyAsync(dev_A_digits, A_digits, number_of_digits * sizeof(char), cudaMemcpyHostToDevice, stream1);
		normalized_freq<<<(int)ceil(number_of_digits/1024) + 1, 1024>>>(dev_A_digits,  dev_A_freq_count, stream1);
		CUDAErrorCheck();
		cudaStreamSynchronize(stream1);

		if (fgets(A_digits, number_of_digits + 1, file_read) != NULL){
			// printf("Input: %s\n\n", A_digits);
			len = strlen(A_digits);
			// printf("Size %i\n", len);
			strcat(A_digits, "---------------------------");
			cudaSetDevice(1);
			cudaMemcpyAsync(dev_B_digits, A_digits, number_of_digits * sizeof(char), cudaMemcpyHostToDevice, stream2);
			normalized_freq<<<(int)ceil(number_of_digits/1024) + 1, 1024>>>(dev_B_digits,  dev_B_freq_count, stream2);
			CUDAErrorCheck();
			cudaStreamSynchronize(stream2);
		}
		cudaDeviceSynchronize();
		printf("GPUs Synchronized (%i)\n\n", device_sync_count);
		device_sync_count++;
	}

	cudaSetDevice(0);
	cudaMemcpyAsync(A_freq_count, dev_A_freq_count, number_of_freq_count * sizeof(int), cudaMemcpyDeviceToHost, stream1);
	cudaSetDevice(1);
	cudaMemcpyAsync(B_freq_count, dev_B_freq_count, number_of_freq_count * sizeof(int), cudaMemcpyDeviceToHost, stream2);

	cudaFree(dev_A_freq_count);
	cudaFree(dev_A_digits);
	cudaFree(dev_B_freq_count);
	cudaFree(dev_B_digits);

	A_freq_count[3]++;
	int total = 0;
	for (int i = 0; i < 10; i++) {
		printf("Frequence at %i, is %i\n", i, A_freq_count[i] + B_freq_count[i]);
		total = total + A_freq_count[i] + B_freq_count[i];
	}
	printf("Total Frequency: %i\n", total);
	//STOP
	stop = clock();

	//get the execution time
	execution_time = ((double) (stop - start)) / CLOCKS_PER_SEC;
	//Print the execution time
	printf("Execution Time in Seconds: %.8lf\n", execution_time );

	//exit the program, done
	exit(0);

}

//dummy function
__global__ void first_call(){
	int z = 1;
	if ( z != 1 ){
	}
}
__global__ void normalized_freq(char* digits, int* global_mem_freq_count, cudaStream_t A) {

	int global_ID = blockDim.x * blockIdx.x + threadIdx.x;

	if (global_ID >= number_of_digits || digits[global_ID] == '\0' || digits[global_ID] == '-'){
		return;
	}

	__shared__ int block_freq_count[10];

	if (threadIdx.x == 0){
		memset(block_freq_count, 0, 10 * sizeof(int));
	}
	syncthreads();

	// printf("Current input: %c, adding \n", digits[global_ID]);
	atomicAdd(&block_freq_count[digits[global_ID] - '0'], 1);

	syncthreads();
	if (threadIdx.x == 0){
			// printf("Global Adding Freq\n");
		for (int i = 0; i< 10 ; i++){
			atomicAdd(&global_mem_freq_count[i], block_freq_count[i]);
		}
	}
}
void CUDAErrorCheck()
{
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
                printf("CUDA -error : %s (%d)\n", cudaGetErrorString(error), error);
                //exit(0);
        }
}