#ifndef CNN_INFERENCE
#define CNN_INFERENCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * Padding mode enumeration.
 * VALID means no padding will be done on the input tensor before the following convolution operation.
 * SAME means the input tensor will be padded in such a way that the output of the following convolution operation will have the same height and width as the input tensor.
*/
typedef enum pm {VALID, SAME} padding_mode;


/**
 * The structure representing a tensor. Basically a convenient way to have a 3D float array and its dimensions together.
*/
typedef struct {
    int dims[3]; /**< Dimensions of the 3D tensor [depth, height, width] */
    float ***T; /**< The 3D float array representing the tensor */
} Tensor;


/**
 * The structure representing a convolution layer. Each layer has a set of *kernel boxes* which is a tool for intuitively defining the weights.
 * Each *kernel box* holds **d** kernels, where **d** is the depth of the input tensor, meaning there is 1 kernel/filter for each input layer.
*/
typedef struct {
    int n_kb; /**< The number of kernel boxes(filters) in this layer. Also equal to the length of the *bias_array* and to the depth of the output tensor. */
    int kernel_box_dims[3]; /**< Dimensions of the kernel box [depth, height, width] */
    float ****kernel_box_group; /**< The array of kernel boxes which represents the weights of this convolution layer as described above. */
    float *bias_array; /**< The bias array of this layer. The length of this array is n_kb. */
    int stride_x; /**< The stride of the kernel window horizontally */
    int stride_y; /**< The stride of the kernel window vertically */
    padding_mode padding; /**< Padding option for this convolution layer as described in padding_mode */
} ConvLayer;


/**
 * The structure representing a dense layer. Each layer has a set of *kernel boxes* which is a tool for intuitively defining the weights.
 * Each *kernel box* holds **d** kernels, where **d** is the depth of the input tensor, meaning there is 1 kernel/filter for each input layer.
 * Another name for dense layers are fully connected layers, therefore, naturally, the height and the width of the kernels in the dense layers must match those of the input layer.
 * Altough *kernel box groups* are used in this library to hold the weights of the dense layers for completeness, one could think of conventional dense layers as
 * layers whose kernel boxes have height and width 1. The number of kernel boxes in these layers may be thought of as the number of their units.
*/
typedef struct {
    int n_kb; /**< The number of kernel boxes(filters) in this layer. Also equal to the length of the *bias_array* and to the depth of the output tensor. This can also be thought of as the number of units this dense layer has.*/
    int kernel_box_dims[3]; /**< Dimensions of the kernel box [depth, height, width] */
    float ****kernel_box_group; /**< The array of kernel boxes which represents the weights of this convolution layer as described above. */
    float *bias_array; /**< The bias array of this layer. The length of this array is n_kb. */
} DenseLayer;


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
Tensor *ELU_activation(Tensor *input, int free_input);
Tensor *linear_activation(Tensor *input, int free_input);
Tensor *apply_padding(Tensor *input, int padding_x, int padding_y, int free_input);
Tensor *MaxPool(Tensor *input, int height, int width, int stride_x, int stride_y, padding_mode padding, int free_input);
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