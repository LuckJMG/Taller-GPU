#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel(int N, int* A, int* B, int* C){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
		int idx = y*N + x;
		C[idx] = 0;
		for (int i = 0; i < N; i++) {
			int idA = (i*N) + x;
			int idB = (y*N) + i;
			C[idx] += A[idA]*B[idB];
		}
    }
}

void printMatrix(int* matrix, int size) {
	for (int row=0; row < size; row++) {
		for (int col=0; col < size; col++) {
			int id = row*size + col;
			printf("%d ", matrix[id]);
		}
		printf("\n");
	}
}

int main(int argc, char **argv){
	// Configuración GPU
	cudaError_t err = cudaSuccess;
	int N = 4;
	dim3 block(16, 16);
	dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
	printf("Grid: %d x %d, Block: %d x %d\n", grid.x, grid.y, block.x, block.y);

	// Inicializar Vectores
	int* d_A = NULL;
	int* d_B = NULL;
	int* d_C = NULL;
	int* A = new int[N*N];
	int* B = new int[N*N];
	int* C = new int[N*N];
	size_t size = N * N * sizeof(int);

	for(int i = 0; i < N*N ;i++){
		A[i] = 1;
		B[i] = 2;
	}

	printMatrix(A, N);
	printMatrix(B, N);

	// Asignar memoria en la GPU
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	// Copiar inputs a GPU
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	// Correr kernel en la GPU
	kernel<<<grid, block>>>(N, d_A, d_B, d_C);
	cudaDeviceSynchronize();  // Esperar a que todos los threads terminen
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copiar output de la GPU
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	printMatrix(C, N);

	// Limpiar memoria
	delete[] A;
	delete[] B;
	delete[] C;
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
