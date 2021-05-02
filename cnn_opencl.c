#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "cnn.h"

/*
 * TODO
 * Define global variables here. For example,
 * cl_platform_id platform;
 */
cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_mem inputObj = NULL, outputObj = NULL, filterObj = NULL, biasObj=NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_platform_id platform_id = NULL;
cl_uint ret_num_devices;
cl_int err;

#define MAX_SOURCE_SIZE 4000
#define ReLU(x) (((x)>0)?(x):0)

static void pooling2x2(float *input, float *output, int N) {
	int i, j, k, l;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			float max = 0;
			for (k = 0; k < 2; k++) {
				for (l = 0; l < 2; l++) {
					float pixel = input[(i * 2 + k) * 2 * N + j * 2 + l];
					max = (max > pixel) ? max : pixel;
				}
			}
			output[i * N + j] = max;
		}
	}
}

/*
 * D = channel size
 * N = width and height of an output image
 * Thus, input is (D, N * 2, N * 2) and output is (D, N, N).
 */
static void pooling_layer(float *inputs, float *outputs, int D, int N) {
	int i;
	for (i = 0; i < D; i++) {
		float * input = inputs + i * N * N * 4;
		float * output = outputs + i * N * N;
		pooling2x2(input, output, N);
	}
}

static void convolution3x3(float *input, float *output, float *filter, int N) {
	int i, j, k, l;

	for (j = 0; j < N; j++) {
		float sum = 0;
		for (k = 1; k < 3; k++) {
			for (l = 0; l < 3; l++) {
				int x = k - 1;
				int y = j + l - 1;

				(0 <= y && y < N) && (sum += input[x * N + y] * filter[k * 3 + l]);
			}
		}
		output[j] += sum;
	}

	for (i = 1; i < N - 1; i++) {
		float sum = 0;
		for (k = 0; k < 3; k++) {
			for (l = 1; l < 3; l++) {
				int x = i + k - 1;
				int y = l - 1;

				(y < N) && (sum += input[x * N + y] * filter[k * 3 + l]);
			}
		}
		output[i * N] += sum;
		for (j = 1; j < N; j++) {
			sum = 0;
			for (k = 0; k < 3; k++) {
				for (l = 0; l < 3; l++) {
					int x = i + k - 1;
					int y = j + l - 1;

					(y < N) && (sum += input[x * N + y] * filter[k * 3 + l]);
				}
			}
			output[i * N + j] += sum;
		}
	}
	for (j = 0; j < N; j++) {
		float sum = 0;
		for (k = 0; k < 2; k++) {
			for (l = 0; l < 3; l++) {
				int x = i + k - 1;
				int y = j + l - 1;

				(0 <= y && y < N) && (sum += input[x * N + y] * filter[k * 3 + l]);
			}
		}
		output[(N - 1) * N + j] += sum;
	}

}

static void convolution_layer3x64(float *inputs, float *outputs, float *filters, float *biases, int D2, int D1, int N) {

	
	
	
	inputObj = clCreateBuffer(context, 0, sizeof(float) * N * N * D1 , NULL, &err);
	outputObj = clCreateBuffer(context, 0, sizeof(float) * N * N * D2, NULL, &err);
	filterObj = clCreateBuffer(context, 0, sizeof(float) * 3*3*D1*D2, NULL, &err);
	biasObj = clCreateBuffer(context, 0, sizeof(float) * N * N *D2, NULL, &err);

	/*OpenCL 커널 파라미터 설정*/
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&inputObj);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&outputObj);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&filterObj);
	err = clSetKernelArg(kernel, 3, sizeof(int) , &N);
	err = clSetKernelArg(kernel, 4, sizeof(int), &D1);
	err = clSetKernelArg(kernel, 5, sizeof(int), &D2);
	err = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&biasObj);
	
	
	err = clEnqueueWriteBuffer(command_queue, inputObj, CL_TRUE, 0, sizeof(float) * N * N * D1, inputs, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, outputObj, CL_TRUE, 0, sizeof(float) * N * N * D2, 0, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterObj, CL_TRUE, 0, sizeof(float) * 3 * 3 *D1*D2, filters, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, filterObj, CL_TRUE, 0, sizeof(float) * N * N *D2, biases, 0, NULL, NULL);
	/*OpenCL 커널 실행*/
	size_t global_size[2] = { N*N,D2 };//이거 몰로하지

	size_t local_size[2] = {16,16};

	clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

	/*실행 결과를 메모리 버퍼에서 얻음*/
	err = clEnqueueReadBuffer(command_queue, outputObj, CL_TRUE, 0, sizeof(float) * N * N * D2, outputs, 0, NULL, NULL);
	
	
}
static void convolution_layer(float *inputs, float *outputs, float *filters, float *biases, int D2, int D1, int N) {
	int i, j;

	memset(outputs, 0, sizeof(float) * N * N * D2);

	for (j = 0; j < D2; j++) {
		for (i = 0; i < D1; i++) {
			float * input = inputs + N * N * i;
			float * output = outputs + N * N * j;
			float * filter = filters + 3 * 3 * (j * D1 + i);
			convolution3x3(input, output, filter, N);
		}
	}

	for (i = 0; i < D2; i++) {
		float * output = outputs + N * N * i;
		float bias = biases[i];
		for (j = 0; j < N * N; j++) {
			output[j] = ReLU(output[j] + bias);
		}
	}
}

/*
 * M = output size
 * N = input size
 */
static void fc_layer(float *input_neuron, float *output_neuron, float *weights, float *biases, int M, int N) {
	int i, j;
	for (j = 0; j < M; j++) {
		float sum = 0;
		for (i = 0; i < N; i++) {
			sum += input_neuron[i] * weights[j * N + i];
		}
		sum += biases[j];
		output_neuron[j] = ReLU(sum);
	}


}

static void softmax(float *output, int N) {
	int i;
	float max = output[0];
	for (i = 1; i < N; i++) {
		max = (output[i] > max) ? output[i] : max;
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(output[i] - max);
	}
	for (i = 0; i < N; i++) {
		output[i] = exp(output[i] - max) / sum;
	}
}

static int find_max(float *fc, int N) {
	int i;
	int maxid = 0;
	float maxval = 0;
	for (i = 0; i < N; i++) {
		if (maxval < fc[i]) {
			maxval = fc[i];
			maxid = i;
		}
	}
	return maxid;
}
float* alloc_layer(size_t n) {
	return (float*)malloc(n * sizeof(float));
}

void cnn_init() {
	int i;
	FILE *fp;
	char fileName[] = "./kernel.cl";
	char *source_code;

	size_t source_size;
	/*커널을 포함한 소스 코드를 로드*/
	fp = fopen(fileName, "r");

	if (!fp) {
		exit(1);
	}
	source_code = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_code, 1, MAX_SOURCE_SIZE, fp);

	fclose(fp);

	/*플랫폼, 디바이스 정보를 얻음*/
	err = clGetPlatformIDs(1, &platform_id, NULL);
	//3번째 파라미터는 얻어야할 디바이스 수를 지정한다.
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);

	/*소스 코드 생성-> 커널 생성*/
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "convolution3x3", &err);

}

void cnn(float *images, float **network, int *labels, float *confidences, int num_images) {
	float *w1_1, *b1_1, *w1_2, *b1_2;
	float *w2_1, *b2_1, *w2_2, *b2_2;
	float *w3_1, *b3_1, *w3_2, *b3_2, *w3_3, *b3_3;
	float *w4_1, *b4_1, *w4_2, *b4_2, *w4_3, *b4_3;
	float *w5_1, *b5_1, *w5_2, *b5_2, *w5_3, *b5_3;
	float *w1, *b1, *w2, *b2, *w3, *b3;
	w1_1 = network[0]; b1_1 = network[1];
	w1_2 = network[2]; b1_2 = network[3];
	w2_1 = network[4]; b2_1 = network[5];
	w2_2 = network[6]; b2_2 = network[7];
	w3_1 = network[8]; b3_1 = network[9];
	w3_2 = network[10]; b3_2 = network[11];
	w3_3 = network[12]; b3_3 = network[13];
	w4_1 = network[14]; b4_1 = network[15];
	w4_2 = network[16]; b4_2 = network[17];
	w4_3 = network[18]; b4_3 = network[19];
	w5_1 = network[20]; b5_1 = network[21];
	w5_2 = network[22]; b5_2 = network[23];
	w5_3 = network[24]; b5_3 = network[25];
	w1 = network[26]; b1 = network[27];
	w2 = network[28]; b2 = network[29];
	w3 = network[30]; b3 = network[31];

	// allocate memory for output of each layer
	float *c1_1, *c1_2, *p1;
	float *c2_1, *c2_2, *p2;
	float *c3_1, *c3_2, *c3_3, *p3;
	float *c4_1, *c4_2, *c4_3, *p4;
	float *c5_1, *c5_2, *c5_3, *p5;
	float *fc1, *fc2, *fc3;
	c1_1 = alloc_layer(64 * 32 * 32);
	c1_2 = alloc_layer(64 * 32 * 32);
	p1 = alloc_layer(64 * 16 * 16);
	c2_1 = alloc_layer(128 * 16 * 16);
	c2_2 = alloc_layer(128 * 16 * 16);
	p2 = alloc_layer(128 * 8 * 8);
	c3_1 = alloc_layer(256 * 8 * 8);
	c3_2 = alloc_layer(256 * 8 * 8);
	c3_3 = alloc_layer(256 * 8 * 8);
	p3 = alloc_layer(256 * 4 * 4);
	c4_1 = alloc_layer(512 * 4 * 4);
	c4_2 = alloc_layer(512 * 4 * 4);
	c4_3 = alloc_layer(512 * 4 * 4);
	p4 = alloc_layer(512 * 2 * 2);
	c5_1 = alloc_layer(512 * 2 * 2);
	c5_2 = alloc_layer(512 * 2 * 2);
	c5_3 = alloc_layer(512 * 2 * 2);
	p5 = alloc_layer(512 * 1 * 1);
	fc1 = alloc_layer(512);
	fc2 = alloc_layer(512);
	fc3 = alloc_layer(10);
	

	// run network
	for (int i = 0; i < num_images; ++i)
	{
		float *image = images + i * 3 * 32 * 32;

		convolution_layer3x64(image, c1_1, w1_1, b1_1, 64, 3, 32);//32*32*64
		convolution_layer(c1_1, c1_2, w1_2, b1_2, 64, 64, 32);//32*32*64
		pooling_layer(c1_2, p1, 64, 16);

		convolution_layer(p1, c2_1, w2_1, b2_1, 128, 64, 16);//16*16*128
		convolution_layer(c2_1, c2_2, w2_2, b2_2, 128, 128, 16);//16*128*128
		pooling_layer(c2_2, p2, 128, 8);

		convolution_layer(p2, c3_1, w3_1, b3_1, 256, 128, 8);//8*8*256
		convolution_layer(c3_1, c3_2, w3_2, b3_2, 256, 256, 8);//8*8*256
		convolution_layer(c3_2, c3_3, w3_3, b3_3, 256, 256, 8);//8*8*256
		pooling_layer(c3_3, p3, 256, 4);

		convolution_layer(p3, c4_1, w4_1, b4_1, 512, 256, 4);//4*4*512
		convolution_layer(c4_1, c4_2, w4_2, b4_2, 512, 512, 4);//4*4*512
		convolution_layer(c4_2, c4_3, w4_3, b4_3, 512, 512, 4);//4*4*512
		pooling_layer(c4_3, p4, 512, 2);

		convolution_layer(p4, c5_1, w5_1, b5_1, 512, 512, 2);//2*2*512
		convolution_layer(c5_1, c5_2, w5_2, b5_2, 512, 512, 2);//2*2*512
		convolution_layer(c5_2, c5_3, w5_3, b5_3, 512, 512, 2);
		pooling_layer(c5_3, p5, 512, 1);

		fc_layer(p5, fc1, w1, b1, 512, 512);
		fc_layer(fc1, fc2, w2, b2, 512, 512);
		fc_layer(fc2, fc3, w3, b3, 10, 512);

		softmax(fc3, 10);

		labels[i] = find_max(fc3, 10);
		confidences[i] = fc3[labels[i]];
	}


	//-----------------------------------------------------------------
	free(c1_1); free(c1_2); free(p1);
	free(c2_1); free(c2_2); free(p2);
	free(c3_1); free(c3_2); free(c3_3); free(p3);
	free(c4_1); free(c4_2); free(c4_3); free(p4);
	free(c5_1); free(c5_2); free(c5_3); free(p5);
	free(fc1); free(fc2); free(fc3);
}
