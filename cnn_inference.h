#ifndef CNN_INFERENCE
#define CNN_INFERENCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef enum pm {VALID, SAME} padding_mode;

typedef struct {
    int dims[3]; //Dimensions
    float ***T; //Tensor itself
} Tensor;

typedef struct {
    int n_kb; //Number of kernel boxes in this layer
    int kernel_box_dims[3];
    float ****kernel_box_group;
    float *bias_array;
    int stride_x;
    int stride_y;
    padding_mode padding;
} ConvLayer;

typedef struct {
    int n_kb; //Number of kernel boxes in this layer
    int kernel_box_dims[3];
    float ****kernel_box_group;
    float *bias_array;
} DenseLayer;  //aka Fully Connected Layer

//Layer generators
ConvLayer *empty_Conv(int n_kb, int d_kb, int h_kb, int w_kb, int stride_x, int stride_y, padding_mode padding);
ConvLayer *new_Conv(int n_kb, int d_kb, int h_kb, int w_kb, float **** weights_array, float * biases_array, int stride_x, int stride_y, padding_mode padding);
DenseLayer *empty_Dense(int n_kb, int d_kb, int h_kb, int w_kb);
DenseLayer *new_Dense(int n_kb, int d_kb, int h_kb, int w_kb, float **** weights_array, float * biases_array);

//Tensor operations
Tensor *Conv(Tensor *input, ConvLayer *layer, Tensor *(*activation)(Tensor *,int), int free_input);
Tensor *Dense(Tensor *input, DenseLayer *layer, Tensor *(*activation)(Tensor *,int), int free_input);
Tensor *sigmoid_activation(Tensor *input, int free_input);
Tensor *ReLU_activation(Tensor *input, int free_input);
Tensor *linear_activation(Tensor *input, int free_input);
Tensor *apply_same_padding(Tensor *input, ConvLayer *layer, int free_input);
Tensor *MaxPool(Tensor *input, int height, int width, int stride_x, int stride_y, int free_input);
Tensor *FlattenW(Tensor *input, int free_input);
Tensor *FlattenH(Tensor *input, int free_input);
Tensor *FlattenD(Tensor *input, int free_input);
Tensor *Add(Tensor **input_tensors, int n_tensors, int free_inputs);
Tensor *Average(Tensor **input_tensors, int n_tensors, int free_inputs);

//utility functions
void print_tensor(Tensor *t);
float ****alloc_4D(int b, int d, int h, int w);
float ***alloc_3D(int d, int h, int w);
void print_conv_details(ConvLayer layer);
void free_tensor(Tensor *t);
Tensor *make_tensor(int d, int h, int w, float ***array);
void free_ConvLayer(ConvLayer *layer);
void free_DenseLayer(DenseLayer *layer);

#endif