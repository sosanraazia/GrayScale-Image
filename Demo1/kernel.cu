
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <omp.h>

void grayScaleFilterCPU(cv::Mat inputImage,  cv::Mat outputImage, int width, int height) {
	#pragma omp parallel for
	for (int y = 0; y < height; ++y) {
		// Parallelize the inner loop for grayscale conversion
		#pragma omp parallel for
		for (int x = 0; x < width; ++x) {
			int pixelIndex = y * width + x;
			unsigned char r = inputImage.at<cv::Vec3b>(y, x)[0];
			unsigned char g = inputImage.at<cv::Vec3b>(y, x)[1];
			unsigned char b = inputImage.at<cv::Vec3b>(y, x)[2];
			unsigned char gray = 0.299f * r + 0.587f * g + 0.114f * b;
			outputImage.at<unsigned char>(y, x) = gray;
		}
	}


}
__global__ void grayscaleFilter(unsigned char* inputImage, unsigned char* outputImage, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int pixelIndex = y * width + x;
		unsigned char r = inputImage[3 * pixelIndex];
		unsigned char g = inputImage[3 * pixelIndex + 1];
		unsigned char b = inputImage[3 * pixelIndex + 2];
		unsigned char gray = 0.299f * r + 0.587f * g + 0.114f * b;
		outputImage[pixelIndex] = gray;
	}
}

int main()
{
	// Declare a cv::Mat to load image and allocate memory for it
	cv::Mat inputImage = cv::imread("demo1.jpg", cv::IMREAD_COLOR);
	if (inputImage.empty()) {
		std::cerr << "Could not open or find the image!" << std::endl;
		return -1;
	}

	//extract width and height of the image
	int width = inputImage.cols;
	int height = inputImage.rows;

	// Declare a cv::Mat for the CPU output image and allocate memory for it
	cv::Mat cpuoutputImage(height, width, CV_8UC1);

	// cpu timer started
	double cpuStart = static_cast<double>(cv::getTickCount());

	//grayscale on CPU
	grayScaleFilterCPU(inputImage, cpuoutputImage, width, height);

	//cpu timer ended
	double cpuEnd = static_cast<double>(cv::getTickCount());

	// Save the CPU output image as a grayscale image file
	std::string outputFileName = "cpuOutput.jpg";

	//write file
	bool success = cv::imwrite(outputFileName, cpuoutputImage);
	if (success) {
		std::cout << "Grayscale image saved as " << outputFileName << std::endl;
	}
	else {
		std::cerr << "Failed to save the grayscale image!" << std::endl;
	}

	//calculate CPU time
	double cpuTime = (cpuEnd - cpuStart) / cv::getTickFrequency();

	//std::cout << "CPU execution time: " << cpuTime << " seconds" << std::endl;
	std::cout << "CPU execution time using multi-thread: " << cpuTime << " seconds" << std::endl;
		
	unsigned char* h_inputImage = inputImage.data;
	unsigned char* h_outputImage = new unsigned char[width * height];
	unsigned char* d_inputImage, * d_outputImage;

	//allocate memory on device for input and output images
	cudaMalloc((void**)&d_inputImage, 3 * width * height * sizeof(unsigned char));
	cudaMalloc((void**)&d_outputImage, width * height * sizeof(unsigned char));

	//copy input image data to device memory
	cudaMemcpy(d_inputImage, h_inputImage, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

	//calculate launch configuration for kernel launch
	dim3 blockDim(16, 16);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

	//initiate GPU timer
	double gpuStart = static_cast<double>(cv::getTickCount());

	//kernel call with synchronization
	grayscaleFilter << <gridDim, blockDim >> > (d_inputImage, d_outputImage, width, height);
	cudaDeviceSynchronize();


	//gpu timer ended
	double gpuEnd = static_cast<double>(cv::getTickCount());

	//calculate gpu time
	double gpuTime = (gpuEnd - gpuStart) / cv::getTickFrequency();
	std::cout << "GPU execution time: " << gpuTime << " seconds" << std::endl;

	//copy processed image data from device to host (gpu to cpu)	
	cudaMemcpy(h_outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	//write processed data from the device to the host
	cv::Mat outputImage(height, width, CV_8UC1, h_outputImage);
	cv::imwrite("gpuOutput.jpg", outputImage);

	// Calculate speedup
	double speedup = cpuTime / gpuTime;
	std::cout << "Speedup: " << speedup << "x" << std::endl;

	//free host and device memory
	cudaFree(d_inputImage);
	cudaFree(d_outputImage);
	delete[] h_outputImage;

	std::cout << "Grayscale filter applied using GPU and saved as output.jpg." << std::endl;

	return 0;
}


