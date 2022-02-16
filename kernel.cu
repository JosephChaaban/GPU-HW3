
#include "common.h"
#include "timer.h"

#define TILE_DIM 32

__global__ void mm_tiled_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    // TODO

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    unsigned int row = blockIdx.y*TILE_DIM + threadIdx.y;
    unsigned int col = blockIdx.x*TILE_DIM + threadIdx.x;
    float sum = 0.0f;
 
for (unsigned int tile = 0; tile < ceilf(N/(float)TILE_DIM; ++tile)){
    
    if(row < M && (tile*TILE_DIM + threadIdx.x)<K){
    A_s[threadIdx.y][threadIdx.x] = A[row*K + tile*TILE_DIM + threadIdx.x];}
    else {
        A_s[threadIdx.y][threadIdx.x] = 0;
    }
    if(col <N && (i*TILE_DIM + threadIdx.y) < K){
    B_s[threadIdx.y][threadIdx.x] = B[(tile*TILE_DIM + threadIdx.y)*N + col];}
    else{
        B_s[threadIdx.x][threadIdx.y] = 0;
    }

    __syncthreads();

    for(unsigned int j = 0; j < TILE_DIM; ++j){
        sum += A_s[threadIdx.y][j]*B_s[j][threadIdx];
    }
    
    __syncthreads();


}
    if(row < M && col < N){
        C[row*N + col] = sum;
    }


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

	mm_tiled_kernel <<< numberOfBlocks, numberOfThreadsPerBlock >>> (A_d, B_d, C_d, M, N, K);


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

