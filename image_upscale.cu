#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "attachments/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "attachments/stb_image_write.h"

#include <cuda_runtime.h>

__global__ void kernel(unsigned char* d_img, int width, int height, int channels, unsigned char* out_img){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
		int idx_tl = (2*y * width + 2*x) * channels;
		int idx_tr = (2*y * width + 2*x + 1) * channels;
		int idx_bl = ((2*y + 1) * width + 2*x) * channels;
		int idx_br = ((2*y + 1) * width + 2*x + 1) * channels;

		for (int c = 0; c < channels; c++) {
			out_img[idx_tl + c] = d_img[idx + c];
			out_img[idx_tr + c] = d_img[idx + c];
			out_img[idx_bl + c] = d_img[idx + c];
			out_img[idx_br + c] = d_img[idx + c];
		}
    }
}

int main(int argc, char **argv){
	// Cargar imagen
	int width, height, channels;
	unsigned char* img = stbi_load("attachments/image.jpg", &width, &height, &channels, 0);
	if (!img) {
		printf("Error at loading image.jpg\n");
		return 1;
	}

	// Configuración GPU
	cudaError_t err = cudaSuccess;
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	printf("Grid: %d x %d, Block: %d x %d\n", grid.x, grid.y, block.x, block.y);

	// Inicializar valores
	size_t size = width * height * channels * sizeof(unsigned char);
	unsigned char* d_img = NULL;
	unsigned char* d_out_img = NULL;

	// Asignar memoria en GPU
	cudaMalloc((void**)&d_img, size);
	cudaMalloc((void**)&d_out_img, 4*size);
	cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);  // Copiar imagen a GPU

	// Procesar imagen
	kernel<<<grid, block>>>(d_img, width, height, channels, d_out_img);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	unsigned char* out_img = (unsigned char*)malloc(4*size);
	cudaMemcpy(out_img, d_out_img, 4*size, cudaMemcpyDeviceToHost);  // Copiar imagen procesada de la GPU

	// Guardar imagen procesada
	if (!stbi_write_png("processed.jpg", 2*width, 2*height, channels, out_img, 2*width * channels)) {
		printf("Error at saving processed.jpg\n");
	} else {
		printf("Inverted image saved at processed.jpg\n");
	}

	// Limpiar memoria
	cudaFree(d_img);
	stbi_image_free(img);
	delete d_out_img;

	return 0;
}
