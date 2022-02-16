#include "common.h"
#include "timer.h"

__global__ void mm_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    // TODO

	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(col >= N || row >= M) return;
	
	float sum = 0.0f;
	for(unsigned int i = 0; i < K; i++){
		sum += A[row*K + i] * B[N*i + col];
	}
	
	C[row*N + col] = sum;
}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);

    // TODO

	float *A_d, *B_d, *C_d;
	
	cudaMalloc((void**)&A_d, sizeof(float) * M * K);
	cudaMalloc((void**)&B_d, sizeof(float) * K * N);
	cudaMalloc((void**)&C_d, sizeof(float) * M * N);




    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);

    // TODO

	cudaMemcpy(A_d, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, sizeof(float) * K *N, cudaMemcpyHostToDevice);




    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);

    // TODO

	dim3 numberOfThreadsPerBlock(32, 32);
	dim3 numberOfBlocks((N + numberOfThreadsPerBlock.x - 1) / numberOfThreadsPerBlock.x, (M + numberOfThreadsPerBlock.y - 1) / numberOfThreadsPerBlock.y);

	mm_kernel <<< numberOfBlocks, numberOfThreadsPerBlock >>> (A_d, B_d, C_d, M, N, K);


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);

    // TODO


	cudaMemcpy(C, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);



    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    // TODO

	cudaFree((void*)A_d);
	cudaFree((void*)B_d);
	cudaFree((void*)C_d);




    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}