#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "attachments/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "attachments/stb_image_write.h"

#include <cuda_runtime.h>

__global__ void kernel(unsigned char* d_img, int width, int height, int channels){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;

        d_img[idx + 0] = 255 - d_img[idx + 0]; // R
        d_img[idx + 1] = 255 - d_img[idx + 1]; // G
        d_img[idx + 2] = 255 - d_img[idx + 2]; // B
    }
}

int main(int argc, char **argv){
	// Cargar imagen
	int width, height, channels;
	unsigned char *img = stbi_load("attachments/image.jpg", &width, &height, &channels, 0);
	if (!img) {
		printf("Error at loading image.jpg\n");
		return 1;
	}

	// Configurar GPU
	cudaError_t err = cudaSuccess;
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	printf("Grid: %d x %d, Block: %d x %d\n", grid.x, grid.y, block.x, block.y);

	// Inicializar valores
	size_t size = width * height * channels * sizeof(unsigned char);
	unsigned char* d_img = NULL;

	cudaMalloc((void**)&d_img, size);  // Asignar memoria en GPU
	cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);  // Copiar imagen a GPU

	// Procesar imagen
	kernel<<<grid, block>>>(d_img, width, height, channels);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);  // Copiar imagen procesada de la GPU

	// Guardar imagen procesada
	if (!stbi_write_png("inverted.jpg", width, height, channels, img, width * channels)) {
		printf("Error at saving inverted.jpg\n");
	} else {
		printf("Inverted image saved at inverted.jpg\n");
	}

	// Limpiar memoria
	cudaFree(d_img);
	stbi_image_free(img);

	return 0;
}
