#include "cnn_inference.h"
#include "h5_format.h"
#include "rgb_format.h"
#include <time.h>



// // // // This file servers the purpose of guiding you through creating your first model using this library. // // // //



int main(int argc, char *argv[]){


    // For running the executable wih different input images without recompiling
	if(argc!=2){
		printf("\nPlease run as: model.c <image_name>. Image name must be given without the 0,1,2 and without its extension.");
		exit(EXIT_FAILURE);
	}



    // Optional
    // If you wish, you may get the inference time
	clock_t start, end;
	double cpu_time_used;



	// Preparing your input
    // The name of the image is read as a command line argument.
    // INPUT_IMAGE_HEIGHT and INPUT_IMAGE_WIDTH represent the height and the width of the input image to your model
    // These values should either be defined in your header, or be replaced and hard-coded here
    // The load_RGB() function could be found in rgb_format.c
	float ***img;
	img = load_RGB(argv[1], INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH);



    // Configuring the tensors
    // You can make tensors using the make_tensor() function from cnn_inference.c
	Tensor *t;
	t = make_tensor(3, 63, 63, img);



	// Generating Layers
    // Before running your inference experiment, you will first need to replicate your model's Conv and Dense layers using the functions in cnn_inference.c
    // After generating your layers you will need to load their weights. This is done by the functions in h5_format.c.
    // For loading your weights you will first need to export them using generator.py, for details on this module see the README or the documentation.
    // After having generated the weights, all you need to do is to pass the layers and their names to the loading functions.
    // If you've used generator.extract_weights(), then the names of your layers will be the names of your layers from Keras.
	ConvLayer *_conv2d_1;
	_conv2d_1 = empty_Conv(32, 3, 3, 3, 2, 2, VALID);
	load_Conv(_conv2d_1, "conv2d_1");

	ConvLayer *_conv2d_2;
	_conv2d_2 = empty_Conv(32, 32, 3, 3, 1, 1, SAME);
	load_Conv(_conv2d_2, "conv2d_2");

	DenseLayer *_dense_1;
	_dense_1 = empty_Dense(512, 1568, 1, 1);
	load_Dense(_dense_1, "dense_1");

	DenseLayer *_dense_2;
	_dense_2 = empty_Dense(1, 512, 1, 1);
	load_Dense(_dense_2, "dense_2");

	
    
    
    // Inference
    // Once you've created your Dense and Conv layers and loaded their weights, you may start inference.
    // During inference you have access to a variety of tensor operation implementations.
    // Take a look at cnn_inference.h and the documentation for the list of available functions and their details.
	start = clock();

	t = Conv(t, _conv2d_1, ReLU_activation, 1);
	t = MaxPool(t, 3, 3, 2, 2, VALID, 1);
	t = Conv(t, _conv2d_2, ReLU_activation, 1);
	t = MaxPool(t, 3, 3, 2, 2, VALID, 1);
	t = FlattenD(t, 1);
	t = Dense(t, _dense_1, ReLU_activation, 1);
	t = Dense(t, _dense_2, linear_activation, 1);

	end = clock();




    // Output
    // The print_tensor() function will print the output tensor to std_out
	print_tensor(t);




    // Optional
    // If you wish, you may get the inference time
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("\nInference completed in: %f", cpu_time_used);




	// Freeing memory
    // You may use these functions to free the weights of your model once inference is complete.
	free_ConvLayer(_conv2d_1);
	free_ConvLayer(_conv2d_2);
	free_DenseLayer(_dense_1);
	free_DenseLayer(_dense_2);
}