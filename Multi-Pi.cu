
// Gaurav Sheni
// CSC 391 
// December 2, 2015
// Project 4

#include <stdio.h>

__global__ void first_call();
__global__ void normalized_freq(char* digits, int* global_mem_freq_count, cudaStream_t A);

//number of digits to read each time
#define number_of_digits 1000000

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
	FILE *file_output = fopen("freq.dat",  "w+");
	if (file_output == NULL) {
	    printf("File, freq.dat, could not be created.");
	    exit(1);
	}

	int* A_freq_count;
	int* B_freq_count;
	char* A_digits;
	char* B_digits;
	int* A2_freq_count;
	int* B2_freq_count;
	char* A2_digits;
	char* B2_digits;
	char* input_string = (char*)malloc(sizeof(char) * number_of_digits);
	cudaHostAlloc((void**) &A_digits, sizeof(char) * number_of_digits, cudaHostAllocDefault);
	cudaHostAlloc((void**) &B_digits, sizeof(char) * number_of_digits, cudaHostAllocDefault);
	cudaHostAlloc((void**) &A_freq_count, sizeof(int) * 10, cudaHostAllocDefault);
	cudaHostAlloc((void**) &B_freq_count, sizeof(int) * 10, cudaHostAllocDefault);
	cudaHostAlloc((void**) &A2_digits, sizeof(char) * number_of_digits, cudaHostAllocDefault);
	cudaHostAlloc((void**) &B2_digits, sizeof(char) * number_of_digits, cudaHostAllocDefault);
	cudaHostAlloc((void**) &A2_freq_count, sizeof(int) * 10, cudaHostAllocDefault);
	cudaHostAlloc((void**) &B2_freq_count, sizeof(int) * 10, cudaHostAllocDefault);
	for (int j = 0; j < 10 ; j++){
		A_freq_count[j] = 0;
		B_freq_count[j] = 0;
		A2_freq_count[j] = 0;
		B2_freq_count[j] = 0;
	}

	int device_sync_count = 1;

	int *dev_A_freq_count;
	char *dev_A_digits;
	int *dev_B_freq_count;
	char *dev_B_digits;
	int *dev_A2_freq_count;
	char *dev_A2_digits;
	int *dev_B2_freq_count;
	char *dev_B2_digits;

	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	cudaStream_t stream3;
	cudaStreamCreate(&stream3);
	cudaStream_t stream2;
	cudaStreamCreate(&stream2);
	cudaStream_t stream4;
	cudaStreamCreate(&stream4);


	cudaSetDevice(0);
	cudaMalloc((void**) &dev_A_freq_count, 10*sizeof(int));
	cudaMalloc((void**) &dev_A_digits, number_of_digits*sizeof(char));
	cudaMemcpyAsync(dev_A_freq_count, A_freq_count, 10 * sizeof(int), cudaMemcpyHostToDevice, stream1);
	cudaMalloc((void**) &dev_A2_freq_count, 10*sizeof(int));
	cudaMalloc((void**) &dev_A2_digits, number_of_digits*sizeof(char));
	cudaMemcpyAsync(dev_A2_freq_count, A2_freq_count, 10 * sizeof(int), cudaMemcpyHostToDevice, stream3);

	cudaSetDevice(1);
	cudaMalloc((void**) &dev_B_freq_count, 10*sizeof(int));
	cudaMalloc((void**) &dev_B_digits, number_of_digits*sizeof(char));
	cudaMemcpyAsync(dev_B_freq_count, B_freq_count, 10 * sizeof(int), cudaMemcpyHostToDevice, stream2);
	cudaMalloc((void**) &dev_B2_freq_count, 10*sizeof(int));
	cudaMalloc((void**) &dev_B2_digits, number_of_digits*sizeof(char));
	cudaMemcpyAsync(dev_B2_freq_count, B2_freq_count, 10 * sizeof(int), cudaMemcpyHostToDevice, stream4);

	fgetc(file_read);

	while(fgets(A_digits, number_of_digits + 1, file_read) != NULL ) {

		cudaSetDevice(0);
		cudaMemcpyAsync(dev_A_digits, A_digits, number_of_digits * sizeof(char), cudaMemcpyHostToDevice, stream1);
		fgets(A2_digits, number_of_digits + 1, file_read);
		cudaMemcpyAsync(dev_A2_digits, A2_digits, number_of_digits * sizeof(char), cudaMemcpyHostToDevice, stream3);
		cudaSetDevice(1);
		fgets(B_digits, number_of_digits + 1, file_read);
		cudaMemcpyAsync(dev_B_digits, B_digits, number_of_digits * sizeof(char), cudaMemcpyHostToDevice, stream2);
		fgets(B2_digits, number_of_digits + 1, file_read);
		cudaMemcpyAsync(dev_B2_digits, B2_digits, number_of_digits * sizeof(char), cudaMemcpyHostToDevice, stream4);
		cudaSetDevice(0);

		normalized_freq<<<(int)ceil(number_of_digits/235) + 1, 235>>>(dev_A_digits,  dev_A_freq_count, stream1);
		normalized_freq<<<(int)ceil(number_of_digits/235) + 1, 235>>>(dev_A2_digits,  dev_A2_freq_count, stream3);
		cudaSetDevice(1);
		normalized_freq<<<(int)ceil(number_of_digits/235) + 1, 235>>>(dev_B_digits,  dev_B_freq_count, stream2);
		normalized_freq<<<(int)ceil(number_of_digits/235) + 1, 235>>>(dev_B2_digits,  dev_B2_freq_count, stream4);
		cudaDeviceSynchronize();
		printf("GPUs Synchronized (%i)\n", device_sync_count);
		device_sync_count++;
	}

	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);
	cudaStreamSynchronize(stream3);
	cudaStreamSynchronize(stream4);
	cudaSetDevice(0);
	cudaMemcpyAsync(A_freq_count, dev_A_freq_count, 10 * sizeof(int), cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(A2_freq_count, dev_A2_freq_count, 10 * sizeof(int), cudaMemcpyDeviceToHost, stream3);
	cudaSetDevice(1);
	cudaMemcpyAsync(B_freq_count, dev_B_freq_count, 10 * sizeof(int), cudaMemcpyDeviceToHost, stream2);
	cudaMemcpyAsync(B2_freq_count, dev_B2_freq_count, 10 * sizeof(int), cudaMemcpyDeviceToHost, stream4);
	cudaFree(dev_A_freq_count);
	// cudaFree(dev_A_digits);
	// cudaFree(dev_B_freq_count);
	// cudaFree(dev_B_digits);
	// cudaFree(dev_A2_freq_count);
	// cudaFree(dev_A2_digits);
	// cudaFree(dev_B2_freq_count);
	// cudaFree(dev_B2_digits);
	// cudaStreamDestroy(stream1);
	// cudaStreamDestroy(stream2);
	// cudaStreamDestroy(stream3);
	// cudaStreamDestroy(stream4);

	A_freq_count[3]++;
	int total = 0;
	for (int i = 0; i < 10; i++) {
		printf("Frequence at %i, is %i\n", i, A_freq_count[i] + B_freq_count[i]+ A2_freq_count[i] + B2_freq_count[i]);
		float output = (float) ( A_freq_count[i] + B_freq_count[i]+ A2_freq_count[i] + B2_freq_count[i] ) / (float) 100000001;

		fprintf(file_output,"%i\t%f\n", i, output);
		total = total + A_freq_count[i] + B_freq_count[i] + A2_freq_count[i] + B2_freq_count[i];
	}
	printf("Total Frequency: %i\n", total);
	//STOP
	stop = clock();

	//get the execution time
	execution_time = ((double) (stop - start)) / CLOCKS_PER_SEC;
	//Print the execution time

	printf("Execution Time in Seconds: %.8lf\n", execution_time );

	// fclose(file_output);
	// fclose(file_read);

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

	if (global_ID >= number_of_digits ){
		return;
	}
	

	__shared__ int block_freq_count[10];

	if (threadIdx.x == 0){
		memset(block_freq_count, 0, 10 * sizeof(int));
	}
	syncthreads();

	atomicAdd(&block_freq_count[digits[global_ID] - '0'], 1);

	syncthreads();
	if (threadIdx.x == 0){
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