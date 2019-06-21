#ifndef CNN_INFERENCE
#define CNN_INFERENCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct tensor *Tensor;
typedef struct Kernel_Box KernelBox;
typedef struct Bias_Array BiasArray;
typedef struct Convolutional_Layer ConvLayer;
typedef struct Dense_Layer DenseLayer; //aka Fully Connected Layer

struct tensor{
    int dims[3]; //Dimensions
    float ***T; //Tensor itself
};

struct Kernel_Box{ //Each kernel box holds a depth of kernels, i.e. in Conv1 we have 32 'kernel boxes', each 3x3x3
    int dims[3];
    float ***KB;
};


struct Bias_Array{
    int size;
    float *B;
};

struct Convolutional_Layer{
    int n_kb; //Number of kernel boxes in this layer
    int kernel_box_dims[3];
    KernelBox *kernel_box_group;
    BiasArray bias_array;
    int stride;
    int padding;
};


struct Dense_Layer{ //aka Fully Connected Layer
    int n_kb; //Number of kernel boxes in this layer
    int kernel_box_dims[3];
    KernelBox *kernel_box_group;
    BiasArray bias_array;
};

//Layer generators
ConvLayer *empty_Conv(int n_kb, int d_kb, int h_kb, int w_kb, int stride, int padding);
ConvLayer *new_Conv(int n_kb, int d_kb, int h_kb, int w_kb, float **** weights_array, float * biases_array, int stride, int padding, int copy);
DenseLayer *empty_Dense(int n_kb, int d_kb, int h_kb, int w_kb);
DenseLayer *new_Dense(int n_kb, int d_kb, int h_kb, int w_kb, float **** weights_array, float * biases_array, int copy);

//Tensor operations
Tensor Conv(Tensor input, ConvLayer *layer, int dispose_of_input, Tensor (*activation)(Tensor,int));
Tensor Dense(Tensor input, DenseLayer *layer, int dispose_of_input, Tensor (*activation)(Tensor,int));
Tensor sigmoid_activation(Tensor input, int dispose_of_input);
Tensor ReLU_activation(Tensor input, int dispose_of_input);
Tensor linear_activation(Tensor input, int dispose_of_input);
Tensor apply_padding(Tensor input, int padding, int dispose_of_input);
Tensor MaxPool(Tensor input, int height, int width, int stride, int dispose_of_input);
Tensor FlattenW(Tensor input, int dispose_of_input);
Tensor FlattenH(Tensor input, int dispose_of_input);
Tensor FlattenD(Tensor input, int dispose_of_input);

//utility functions
void print_tensor(Tensor t);
float ****alloc_4D(int b, int d, int h, int w);
float ***alloc_3D(int d, int h, int w);
void print_conv_details(ConvLayer layer);
void free_tensor(Tensor t);
Tensor make_tensor(int d, int h, int w, float ***array);

#endif