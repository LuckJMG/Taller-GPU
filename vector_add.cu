#include <cuda_runtime.h>

__global__ void kernel(int N, int* A, int* B, int* C){
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if (tId < N) {
		C[tId] = A[tId] + B[tId];
	}
}

void printVector(int* vec, int size) {
	for (int i=0; i<size; i++) {
		printf("%d ", vec[i]);
	}
	printf("\n");
}

int main(int argc, char **argv){
	// Configuración GPU
	cudaError_t err = cudaSuccess;
	int N = 256;
	int Nblocks = 1;
	int Nthreads = 256;

	// Inicializar Vectores
	int* d_A = NULL;
	int* d_B = NULL;
	int* d_C = NULL;
	int* A = new int[N];
	int* B = new int[N];
	int* C = new int[N];
	size_t size = N * sizeof(int);

	for(int i = 0; i < N ;i++){
		A[i] = 1;
		B[i] = 2;
	}

	printVector(A, N);
	printVector(B, N);

	// Asignar memoria en la GPU
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	// Copiar inputs a GPU
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	// Correr kernel en la GPU
	kernel<<<Nblocks, Nthreads>>>(N, d_A, d_B, d_C);
	cudaDeviceSynchronize();  // Esperar a que todos los threads terminen
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copiar output de la GPU
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	printVector(C, N);

	// Limpiar memoria
	delete[] A;
	delete[] B;
	delete[] C;
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
