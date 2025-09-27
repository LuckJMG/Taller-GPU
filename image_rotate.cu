#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>

#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>

__global__ void kernel(unsigned char* d_img, int width, int height, int channels, unsigned char* out_img){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
		int size = height * width * channels;
        int idx = (y * width + x) * channels;

        out_img[size - idx + 0] = d_img[idx + 0]; // R
        out_img[size - idx + 1] = d_img[idx + 1]; // G
        out_img[size - idx + 2] = d_img[idx + 2]; // B
    }
}

void printVector(int* vec, int size) {
	for (int i=0; i<size; i++) {
		printf("%d ", vec[i]);
	}
	printf("\n");
}

int main(int argc, char **argv){
	cudaError_t err = cudaSuccess;

	int width, height, channels;
	unsigned char* img = stbi_load("image.jpg", &width, &height, &channels, 0);
	unsigned char* out_img = new unsigned char[width * height * channels];
	if (!img) {
		printf("Error at loading image.jpg\n");
		return 1;
	}


	size_t size = width * height * channels * sizeof(unsigned char);
	unsigned char* d_img;

	cudaMalloc((void**)&d_img, size);
	cudaMalloc((void**)&out_img, size);
	cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	printf("Grid: %d x %d, Block: %d x %d\n", grid.x, grid.y, block.x, block.y);

	kernel<<<grid, block>>>(d_img, width, height, channels, out_img);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaMemcpy(img, out_img, size, cudaMemcpyDeviceToHost);

	if (!stbi_write_png("inverted.jpg", width, height, channels, img, width * channels)) {
		printf("Error at saving inverted.jpg\n");
	} else {
		printf("Inverted image saved at inverted.jpg\n");
	}

	cudaFree(d_img);
	stbi_image_free(img);
	delete[] out_img;

	return 0;
}
