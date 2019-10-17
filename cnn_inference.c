#include "cnn_inference.h"

/** \defgroup layergenerators Layer Generators */

/** \defgroup tensoroperations Tensor Operations */

/** \defgroup utilityfunctions Utility Functions */

/** @ingroup layergenerators
 * Creates a convolution layer without weights and returns a pointer to it.
 * 
 * @param n_kb Number of kernel boxes, which is also the number of biases
 * @param d_kb Depth of kernel boxes
 * @param h_kb Height of kernel boxes
 * @param w_kb Width of kernel boxes
 * @param stride_x Stride of kernel window on the x axis(horizontal)
 * @param stride_y Stride of kernel window on the y axis(vertical)
 * @param padding Option from padding_mode (VALID or SAME). VALID means no padding will be done, 
 * SAME means the input tensor will be padded in such a way that the output of the following convolution operation will
 * have the same height and width as the input tensor.
 * 
 * @sa new_Conv(), padding_mode
 * 
 * @return A pointer to the newly created convolution layer
*/
ConvLayer *empty_Conv(int n_kb, int d_kb, int h_kb, int w_kb, int stride_x, int stride_y, padding_mode padding){
    ConvLayer *convolution_layer_pointer;
    convolution_layer_pointer = malloc(sizeof(ConvLayer));
    if(convolution_layer_pointer==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to convolution_layer_pointer in new_Conv.");
		exit(EXIT_FAILURE);
    }

    convolution_layer_pointer->n_kb = n_kb;
    convolution_layer_pointer->kernel_box_dims[0]=d_kb;
    convolution_layer_pointer->kernel_box_dims[1]=h_kb;
    convolution_layer_pointer->kernel_box_dims[2]=w_kb;
    
    convolution_layer_pointer->kernel_box_group = alloc_4D(n_kb, d_kb, h_kb, w_kb);
    convolution_layer_pointer->bias_array = malloc(n_kb*sizeof(float));

    convolution_layer_pointer->stride_x = stride_x;
    convolution_layer_pointer->stride_y = stride_y;
    convolution_layer_pointer->padding = padding;

    return convolution_layer_pointer;
}

/** @ingroup layergenerators
 * Creates a convolution layer with the given weights and returns a pointer to it.
 * 
 * @param n_kb Number of kernel boxes, which is also the number of biases
 * @param d_kb Depth of kernel boxes
 * @param h_kb Height of kernel boxes
 * @param w_kb Width of kernel boxes
 * @param weights_array A 4D float array of dimensions (n_kb * d_kb * h_kb * w_kb) containing the kernel weights
 * @param biases_array A float array of length n_kb conraining the biases
 * @param stride_x Stride of kernel window on the x axis(horizontal)
 * @param stride_y Stride of kernel window on the y axis(vertical)
 * @param padding Option from padding_mode (VALID or SAME). VALID means no padding will be done, 
 * SAME means the input tensor will be padded in such a way that the output of the following convolution operation will
 * have the same height and width as the input tensor.
 * 
 * @sa empty_Conv()
 * 
 * @return A pointer to the newly created convolution layer
*/
ConvLayer *new_Conv(int n_kb, int d_kb, int h_kb, int w_kb, float **** weights_array, float * biases_array, int stride_x, int stride_y, padding_mode padding){
    ConvLayer *convolution_layer_pointer;
    convolution_layer_pointer = malloc(sizeof(ConvLayer)); //convolution_layer_pointer: Convolutional Layer Pointer
    if(convolution_layer_pointer==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to convolution_layer_pointer in new_Conv.");
		exit(EXIT_FAILURE);
    }

    convolution_layer_pointer->n_kb = n_kb;
    convolution_layer_pointer->kernel_box_dims[0]=d_kb;
    convolution_layer_pointer->kernel_box_dims[1]=h_kb;
    convolution_layer_pointer->kernel_box_dims[2]=w_kb;
    
    convolution_layer_pointer->kernel_box_group = weights_array;
    convolution_layer_pointer->bias_array = biases_array;

    convolution_layer_pointer->stride_x = stride_x;
    convolution_layer_pointer->stride_y = stride_y;
    convolution_layer_pointer->padding = padding;

    return convolution_layer_pointer;
}


/** @ingroup layergenerators
 * Creates a dense layer without weights and returns a pointer to it.
 * 
 * @param n_kb Number of kernel boxes aka number of output neurons for dense layers
 * @param d_kb Depth of kernel boxes, this should match the depth of the expected input tensor
 * @param h_kb Height of kernel boxes, this should match the height of the expected input tensor
 * @param w_kb Width of the kernel boxes, this should match the width of the expected input tensor
 * 
 * @return A pointer to the newly created dense layer
*/
DenseLayer *empty_Dense(int n_kb, int d_kb, int h_kb, int w_kb){
    DenseLayer *dense_layer_pointer;
    dense_layer_pointer = malloc(sizeof(DenseLayer)); //dense_layer_pointer: Dense Layer Pointer
    if(dense_layer_pointer==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to dense_layer_pointer in new_Dense.");
		exit(EXIT_FAILURE);
    }

    dense_layer_pointer->n_kb = n_kb;
    dense_layer_pointer->kernel_box_dims[0]=d_kb;
    dense_layer_pointer->kernel_box_dims[1]=h_kb;
    dense_layer_pointer->kernel_box_dims[2]=w_kb;
    
    dense_layer_pointer->kernel_box_group = alloc_4D(n_kb, d_kb, h_kb, w_kb);
    dense_layer_pointer->bias_array = malloc(n_kb*sizeof(float));

   return dense_layer_pointer;
}


/** @ingroup layergenerators
 * Creates a dense layer with the given weights and returns a pointer to it.
 * 
 * @param n_kb Number of kernel boxes aka number of output neurons for dense layers
 * @param d_kb Depth of kernel boxes, this should match the depth of the expected input tensor
 * @param h_kb Height of kernel boxes, this should match the height of the expected input tensor
 * @param w_kb Width of the kernel boxes, this should match the width of the expected input tensor
 * @param weights_array A 4D float array of dimensions (n_kb * d_kb * h_kb * w_kb) containing the kernel weights
 * @param biases_array A float array of length n_kb conraining the biases
 * 
 * @sa empty_Dense()
 * 
 * @return A pointer to the newly created dense layer
*/
DenseLayer *new_Dense(int n_kb, int d_kb, int h_kb, int w_kb, float **** weights_array, float * biases_array){
    DenseLayer *dense_layer_pointer;
    dense_layer_pointer = malloc(sizeof(DenseLayer)); //dense_layer_pointer: Dense Layer Pointer
    if(dense_layer_pointer==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to dense_layer_pointer in new_Dense.");
		exit(EXIT_FAILURE);
    }

    dense_layer_pointer->n_kb = n_kb;
    dense_layer_pointer->kernel_box_dims[0]=d_kb;
    dense_layer_pointer->kernel_box_dims[1]=h_kb;
    dense_layer_pointer->kernel_box_dims[2]=w_kb;
    
    dense_layer_pointer->kernel_box_group = weights_array;
    dense_layer_pointer->bias_array = biases_array;

    return dense_layer_pointer;
}


/** @ingroup tensoroperations
 * @brief Does a convolution operation
 * @details Takes the given tensor through the given convolution layer before applying the given activation function
 * 
 * @param input Input tensor
 * @param layer The convolution layer
 * @param activation A function pointer to the activation function
 * @param free_input Whether to free or overwrite the input tensor, if free_input==1 then the input tensor is lost
 * 
 * @sa Dense()
 * 
 * @return The output tensor  
*/
Tensor *Conv(Tensor *input, ConvLayer *layer, Tensor *(*activation)(Tensor *,int), int free_input){
    if(input->dims[0]!=layer->kernel_box_dims[0]){
        fprintf(stderr, "Error: The depth of the kernel boxes in this layer(%d) and that of its input tensor(%d) must match", layer->kernel_box_dims[0], input->dims[0]);
        exit(EXIT_FAILURE);
    }

    if(layer->padding==SAME){
        int padding_x, padding_y;
        padding_x = layer->stride_x * (input->dims[2]-1) - input->dims[2] + layer->kernel_box_dims[2]; // left + right
        padding_y = layer->stride_y * (input->dims[1]-1) - input->dims[1] + layer->kernel_box_dims[1]; // top + bottom
        input = apply_padding(input, padding_x, padding_y, free_input);
        free_input = 1; // if the padding operation makes 'input' point to a copy of the original input then freeing 'input' is safe
    }

    int output_d = layer->n_kb;
    int output_w, output_h;
    
    // The padding terms in the formulas are omitted because the input tensor at this point has already been padded
    // And its dimensions have been updated accordingly
    output_h = ((input->dims[1] /*+ 2*layer->padding */ - layer->kernel_box_dims[1])/layer->stride_y)+1;
    output_w = ((input->dims[2] /*+ 2*layer->padding */ - layer->kernel_box_dims[2])/layer->stride_x)+1;
    
    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int d,h,w,id,by,bx,i,j;

    // This thing goes over the output array and calculates each cell's value one by one
    for(d=0; d<output_d; d++){ //output depth
        for(h=0; h<output_h; h++){ //output height
            for(w=0; w<output_w; w++){ //output width
                output_array[d][h][w] = 0; //this will hold the sum of the convolutions over each "channel" of the input tensor(the sum over its depth)
                for(id=0; id<input->dims[0]; id++){ //input depth
                    by = h*layer->stride_y; //"begin y" defines where the top edge of the kernel window is on the input layer
                    bx = w*layer->stride_x; //"begin x" defines where the left edge of the kernel window is on the input layer
                    for(i=0; i<(layer->kernel_box_dims[1]); i++){ //traverses the height of kernel window
                        for(j=0; j<(layer->kernel_box_dims[2]); j++){ //traverses the width of kernel window
                            output_array[d][h][w] += input->T[id][by+i][bx+j] * layer->kernel_box_group[d][id][i][j];
                        }
                    }
                }
                
                //Add the bias
                output_array[d][h][w] += layer->bias_array[d];
            }
        }
    }
    
    if(free_input) free_tensor(input);

    Tensor *output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    return activation(output, 1);
}


/** @ingroup tensoroperations
 * @brief Does a dense operation
 * @details Takes the given tensor through the given dense layer before applying the given activation function
 * 
 * @param input Input tensor
 * @param layer The dense layer
 * @param activation A function pointer to the activation function
 * @param free_input Whether to free or overwrite the input tensor, if free_input==1 then the input tensor is lost
 * 
 * @sa Conv()
 * 
 * @return The output tensor  
*/
Tensor *Dense(Tensor *input, DenseLayer *layer, Tensor *(*activation)(Tensor *,int), int free_input){
    if(input->dims[0]!=layer->kernel_box_dims[0] || input->dims[1]!=layer->kernel_box_dims[1] || input->dims[2]!=layer->kernel_box_dims[2]){
        fprintf(stderr,"Error: The dimensions of the kernel boxes of the Dense layer must exactly match those of the input tensor.\n");
        fprintf(stderr,"input has d:%d h:%d w:%d | kernel boxes have d:%d h:%d w:%d", input->dims[0], input->dims[1], input->dims[2], layer->kernel_box_dims[0], layer->kernel_box_dims[1], layer->kernel_box_dims[2]);
        exit(EXIT_FAILURE);
    }

    int output_d = layer->n_kb;
    int output_w =1, output_h = 1;
    
    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int d,h,w,id,i,j;
    float result;
    
    // This thing goes over the output array and calculates each cell's value one by one
    for(d=0; d<output_d; d++){ //output depth
        for(h=0; h<output_h; h++){ //output height
            for(w=0; w<output_w; w++){ //output width
                output_array[d][h][w] = 0;
                for(id=0; id<input->dims[0]; id++){ //input depth, usually 1 for Dense layers as they are usually preceded by a Flattening operation
                    for(i=0; i<layer->kernel_box_dims[1]; i++){ //traverses the height of kernel window
                        for(j=0; j<layer->kernel_box_dims[2]; j++){ //traverses the width of kernel window
                            output_array[d][h][w] += input->T[id][i][j] * layer->kernel_box_group[d][id][i][j];
                        } //here by and bx are both 0 and they never change as the kernel dimensions are equal to the input tensor layer dimensions
                    }
                }

                //Add the bias
                output_array[d][h][w] += layer->bias_array[d];
            }
        }
    }

    if(free_input) free_tensor(input);

    Tensor *output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    return activation(output, 1);
}


/** @ingroup tensoroperations
 * Carries out the sigmoid activation
 * 
 * @param input Input tensor
 * @param free_input Whether to free or overwrite the input tensor, if free_input==1 then the input tensor is lost
 * 
 * @sa linear_activation(), ReLU_activation()
 * 
 * @return The output tensor
*/
Tensor *sigmoid_activation(Tensor *input, int free_input){
    Tensor *output;
    int d,h,w;

    if(free_input){
        output = input;
    } else {
        float ***output_array = alloc_3D(input->dims[0], input->dims[1], input->dims[2]);
        output = make_tensor(input->dims[0], input->dims[1], input->dims[2], output_array);
    }

    for(d=0; d<output->dims[0]; d++){
        for(h=0; h<output->dims[1]; h++){
            for(w=0; w<output->dims[2]; w++){
                output->T[d][h][w] = ((float) (1/(1+exp((double) -1*(input->T[d][h][w])))));
            }
        }
    }

    return output;
}


/** @ingroup tensoroperations
 * Carries out the ReLU activation
 * 
 * @param input Input tensor
 * @param free_input Whether to free or overwrite the input tensor, if free_input==1 then the input tensor is lost
 * 
 * @sa sigmoid_activation(), linear_activation()
 * 
 * @return The output tensor
*/
Tensor *ReLU_activation(Tensor *input, int free_input){
    Tensor *output;
    int d,h,w;

    if(free_input){
        output = input;
    } else {
        float ***output_array = alloc_3D(input->dims[0], input->dims[1], input->dims[2]);
        output = make_tensor(input->dims[0], input->dims[1], input->dims[2], output_array);
    }

    for(d=0; d<output->dims[0]; d++){
        for(h=0; h<output->dims[1]; h++){
            for(w=0; w<output->dims[2]; w++){
                output->T[d][h][w] = (input->T[d][h][w] < 0) ? 0 : input->T[d][h][w];
            }
        }
    }

    return output;
}


/** @ingroup tensoroperations
 * Carries out the linear activation
 * 
 * @param input Input tensor
 * @param free_input Whether to free or overwrite the input tensor, if free_input==1 then the input tensor is lost
 * 
 * @sa sigmoid_activation(), ReLU_activation()
 * 
 * @return The output tensor
*/
Tensor *linear_activation(Tensor *input, int free_input){
    if(free_input)
        return input;
    
    Tensor *output;
    int d,h,w;

    float ***output_array = alloc_3D(input->dims[0], input->dims[1], input->dims[2]);
    output = make_tensor(input->dims[0], input->dims[1], input->dims[2], output_array);
    

    for(d=0; d<output->dims[0]; d++){
        for(h=0; h<output->dims[1]; h++){
            for(w=0; w<output->dims[2]; w++){
                output->T[d][h][w] = input->T[d][h][w];
            }
        }
    }

    return output;
}


/** @ingroup tensoroperations
 * Applies padding to the input tensor. 
 * 
 * If SAME padding is desired but due to the operation parameters symmetric padding is not possible,
 * then this function will follow the tensorflow backend implementation and the bottom and the
 * right side of the tensor will get the additional padding.
 * 
 * @param input Input tensor
 * @param padding_x Padding to the left + Padding to the right in pixels
 * @param padding_x Padding to the top + Padding to the bottom in pixels
 * @param free_input Whether to free or overwrite the input tensor, if free_input==1 then the input tensor is lost
 * 
 * @sa padding_mode
 * 
 * @return The output tensor
*/
Tensor *apply_padding(Tensor *input, int padding_x, int padding_y, int free_input){
    int output_d = input->dims[0];
    int output_h = input->dims[1] + padding_y;
    int output_w = input->dims[2] + padding_x;

    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int d,x,y, squeeze_along_x, squeeze_along_y;
    
    for(d=0; d<output_d; d++){
        //pad symmetric top and bottom
        for(squeeze_along_y=0; squeeze_along_y<(padding_y/2); squeeze_along_y++){
            for(x=0; x<output_w; x++){
                output_array[d][squeeze_along_y][x] = output_array[d][(output_h-1)-squeeze_along_y][x] = 0;
            }
        }

        //handle asymmetry along vertical axis
        if(padding_y%2){
            //pad extra bottom
            for(x=0; x<output_w; x++){
                output_array[d][(output_h-1)-(padding_y/2)][x] = 0;
            }
        }
        
        //pad symmetric left and right
        for(squeeze_along_x=0; squeeze_along_x<(padding_x/2); squeeze_along_x++){
            for(y=0; y<output_h; y++){
                output_array[d][y][squeeze_along_x] = output_array[d][y][(output_w-1)-squeeze_along_x] = 0;
            }
        }

        //handle asymmetry along horizontal axis
        if(padding_x%2){
            //pad extra right
            for(y=0; y<output_h; y++){
                output_array[d][y][(output_w-1)-(padding_x/2)] = 0;
            }
        }

        //load the middle
        for(x=(padding_x/2); x<(output_w-(padding_x/2)-(padding_x%2)); x++){
            for(y=(padding_y/2); y<(output_h-(padding_y/2)-(padding_y%2)); y++){
                output_array[d][y][x] = input->T[d][y-(padding_y/2)][x-(padding_x/2)];
            }    
        }
    }

    Tensor *output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if(free_input) free_tensor(input);

    return output;
}



/** @ingroup tensoroperations
 * Executes max pooling on the given input tensor
 * 
 * @param input Input tensor
 * @param height Height of the pooling window
 * @param width Width of the pooling window
 * @param stride_x Stride of kernel window on the x axis(horizontal)
 * @param stride_y Stride of kernel window on the y axis(vertical)
 * @param padding Option from padding_mode (VALID or SAME). VALID means no padding will be done, 
 * SAME means the input tensor will be padded in such a way that the output of the following pooling operation will
 * have the same height and width as the input tensor.
 * @param free_input Whether to free or overwrite the input tensor, if free_input==1 then the input tensor is lost
 * 
 * @return The output tensor
*/
Tensor *MaxPool(Tensor *input, int height, int width, int stride_x, int stride_y, padding_mode padding, int free_input){
    if(padding==SAME){
        int padding_x, padding_y;
        padding_x = stride_x * (input->dims[2]-1) - input->dims[2] + width; // left + right
        padding_y = stride_y * (input->dims[1]-1) - input->dims[1] + height; // top + bottom
        input = apply_padding(input, padding_x, padding_y, free_input);
        free_input = 1; // if the padding operation makes 'input' point to a copy of the original input then freeing 'input' is safe
    }

    int output_d = input->dims[0];
    int output_w, output_h;
    output_w = ((input->dims[1] - height)/stride_x)+1; // The same formula from the Conv layer
    output_h = ((input->dims[1] - height)/stride_y)+1; // The same formula from the Conv layer

    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int d,h,w,i,j,by,bx;
    float max;

    // This thing goes over the output array and calculates each cell's value one by one
    for(d=0; d<output_d; d++){ //output depth
        for(h=0; h<output_h; h++){ //output height
            for(w=0; w<output_w; w++){ //output width
                by = h*stride_y;
                bx = w*stride_x;
                max = input->T[d][by][bx];
                for(i=0; i<height; i++){ //traverses the height of window
                    for(j=0; j<width; j++){ //traverses the width of window
                        if((input->T[d][by+i][bx+j])>max){
                            max = input->T[d][by+i][bx+j];
                        }
                    }
                }
                output_array[d][h][w] = max;
            }
        }
    }

    Tensor *output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if(free_input) free_tensor(input);

    return output;
}


/** @ingroup tensoroperations
 * Flattens the input tensor into its width such that the output depth and height are 1.
 * 
 * @param input Input tensor
 * @param free_input Whether to free or overwrite the input tensor, if free_input==1 then the input tensor is lost
 * 
 * @sa FlattenH(), FlattenD()
 * 
 * @return The output tensor
*/
Tensor *FlattenW(Tensor *input, int free_input){
    int input_d = input->dims[0], input_h = input->dims[1], input_w = input->dims[2];

    int output_d = 1, output_h = 1;
    int output_w = input_d*input_h*input_w;

    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int w;

    for(w=0; w<output_w; w++){
        output_array[0][0][w] = input->T[w/(input_h*input_w)][(w/input_w)%input_h][w%input_w];
    }

    Tensor *output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if(free_input) free_tensor(input);

    return output;
}


/** @ingroup tensoroperations
 * Flattens the input tensor into its height such that the output depth and width are 1.
 * 
 * @param input Input tensor
 * @param free_input Whether to free or overwrite the input tensor, if free_input==1 then the input tensor is lost
 * 
 * @sa FlattenW(), FlattenD()
 * 
 * @return The output tensor
*/
Tensor *FlattenH(Tensor *input, int free_input){
    int input_d = input->dims[0], input_h = input->dims[1], input_w = input->dims[2];

    int output_d = 1, output_w = 1;
    int output_h = input_d*input_h*input_w;

    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int h;

    for(h=0; h<output_h; h++){
        output_array[0][h][0] = input->T[h/(input_h*input_w)][(h/input_w)%input_h][h%input_w];
    }

    Tensor *output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if(free_input) free_tensor(input);

    return output;
}


/** @ingroup tensoroperations
 * Flattens the input tensor into its depth such that the output height and width are 1.
 * 
 * @param input Input tensor
 * @param free_input Whether to free or overwrite the input tensor, if free_input==1 then the input tensor is lost
 * 
 * @sa FlattenW(), FlattenH()
 * 
 * @return The output tensor
*/
Tensor *FlattenD(Tensor *input, int free_input){
    int input_d = input->dims[0], input_h = input->dims[1], input_w = input->dims[2];

    int output_w = 1, output_h = 1;
    int output_d = input_d*input_h*input_w;

    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int d;

    for(d=0; d<output_d; d++){
        output_array[d][0][0] = input->T[d/(input_h*input_w)][(d/input_w)%input_h][d%input_w];
    }

    Tensor *output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if(free_input) free_tensor(input);

    return output;
}


/** @ingroup tensoroperations
 * Does an element-wise summation of the tensors in the array.
 * 
 * @param input_tensors Array of tensors
 * @param n_tensors Number of tensors in the array
 * @param free_inputs Whether to free or overwrite the input tensors, if free_inputs==0 then the input tensors are lost
 * 
 * @sa Average()
 * 
 * @return The output tensor
*/
Tensor *Add(Tensor **input_tensors, int n_tensors, int free_inputs){
    int i,j;
    for(i=1; i<n_tensors; i++){
        for(j=0; j<3; j++){
            if(input_tensors[i-1]->dims[j] != input_tensors[i]->dims[j]){
                fprintf(stderr, "Error: The input layers to Add() must have the same dimensions.\n");
                fprintf(stderr, "input_tensor_%d has dim_%d=%d | input_tensor_%d has dim_%d=%d", i-1, j, input_tensors[i-1]->dims[j], i, j, input_tensors[i]->dims[j]);
                exit(EXIT_FAILURE);
            }
        }
    }
    
    Tensor *output;
    int d,h,w;
    float ***output_array = alloc_3D(input_tensors[0]->dims[0], input_tensors[0]->dims[1], input_tensors[0]->dims[2]);
    output = make_tensor(input_tensors[0]->dims[0], input_tensors[0]->dims[1], input_tensors[0]->dims[2], output_array);

    for(d=0; d<output->dims[0]; d++){
        for(h=0; h<output->dims[1]; h++){
            for(w=0; w<output->dims[2]; w++){
                output->T[d][h][w] = 0;
                for(i=0; i<n_tensors; i++){
                    output->T[d][h][w] += input_tensors[i]->T[d][h][w];
                }
            }
        }
    }

    if(free_inputs){
        for(i=0; i<n_tensors; i++){
            free_tensor(input_tensors[i]);
        }
    }

    return output;
}


/** @ingroup tensoroperations
 * After doing an element-wise summation of the tensors in the array,
 * the function divides each element in the resulting array by n_tensors
 * to get the element-wise means
 * 
 * @param input_tensors Array of tensors
 * @param n_tensors Number of tensors in the array
 * @param free_inputs Whether to free or overwrite the input tensors, if free_inputs==0 then the input tensors are lost
 * 
 * @sa Add()
 * 
 * @return The output tensor
*/
Tensor *Average(Tensor **input_tensors, int n_tensors, int free_inputs){
    int i,j;
    for(i=1; i<n_tensors; i++){
        for(j=0; j<3; j++){
            if(input_tensors[i-1]->dims[j] != input_tensors[i]->dims[j]){
                fprintf(stderr, "Error: The input layers to Average() must have the same dimensions.\n");
                fprintf(stderr, "input_tensor_%d has dim_%d=%d | input_tensor_%d has dim_%d=%d", i-1, j, input_tensors[i-1]->dims[j], i, j, input_tensors[i]->dims[j]);
                exit(EXIT_FAILURE);
            }
        }
    }
    
    Tensor *output;
    int d,h,w;
    float ***output_array = alloc_3D(input_tensors[0]->dims[0], input_tensors[0]->dims[1], input_tensors[0]->dims[2]);
    output = make_tensor(input_tensors[0]->dims[0], input_tensors[0]->dims[1], input_tensors[0]->dims[2], output_array);

    for(d=0; d<output->dims[0]; d++){
        for(h=0; h<output->dims[1]; h++){
            for(w=0; w<output->dims[2]; w++){
                output->T[d][h][w] = 0;
                for(i=0; i<n_tensors; i++){
                    output->T[d][h][w] += input_tensors[i]->T[d][h][w];
                }
                output->T[d][h][w] /= n_tensors;
            }
        }
    }

    if(free_inputs){
        for(i=0; i<n_tensors; i++){
            free_tensor(input_tensors[i]);
        }
    }

    return output;
}


/** @ingroup utilityfunctions
 * Prints the tensor to stdout.
 * 
 * @param t Tensor *to be printed
*/
void print_tensor(Tensor *t){
    int i,j,k;
    for(i=0; i<t->dims[0]; i++){
        printf("\nLayer %d:\n\n", i);
        for(j=0; j<t->dims[1]; j++){
            for(k=0; k<t->dims[2]; k++){
                printf("%f ", t->T[i][j][k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

/** @ingroup utilityfunctions
 * Allocates memory for a 4D float array with dimensions (b * d * h * w) and returns the pointer.
 * 
 * @param b Dimension 0, size of float *** array
 * @param d Dimension 1, size of float ** array
 * @param h Dimension 2, size of float * array
 * @param w Dimension 3, size of float array
 * 
 * @sa alloc_3D
 * 
 * @return The pointer to the allocated memory
*/
float ****alloc_4D(int b, int d, int h, int w){
    float **** new;
    new = malloc(b*sizeof(float***));
    if(new==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to new in alloc_4D.");
		exit(EXIT_FAILURE);
    }

    int i,j,k;
    for(i=0; i<b; i++){
        new[i] = malloc(d*sizeof(float**));
        if(new[i]==NULL){
            fprintf(stderr, "Error: Unable to allocate memory to new[%d] in alloc_4D.",i);
            exit(EXIT_FAILURE);
        }
        for(j=0; j<d; j++){
            new[i][j] = malloc(h*sizeof(float*));
            if(new[i][j]==NULL){
                fprintf(stderr, "Error: Unable to allocate memory to new[%d][%d] in alloc_4D.",i,j);
                exit(EXIT_FAILURE);
            }
            for(k=0; k<h; k++){
                new[i][j][k] = malloc(w*sizeof(float));
                if(new[i][j][k]==NULL){
                    fprintf(stderr, "Error: Unable to allocate memory to new[%d][%d][%d] in alloc_4D.",i,j,k);
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
    return new;
}


/** @ingroup utilityfunctions
 * Allocates memory for a 3D float array with dimensions (d * h * w) and returns the pointer.
 * 
 * @param d Dimension 0, size of float ** array
 * @param h Dimension 1, size of float * array
 * @param w Dimension 2, size of float array
 * 
 * @sa alloc_4D
 * 
 * @return The pointer to the allocated memory
*/
float ***alloc_3D(int d, int h, int w){
    float ***new;
    new = malloc(d*sizeof(float**));
    if(new==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to new in alloc_3D.");
		exit(EXIT_FAILURE);
    }

    int i,j;
    for(i=0; i<d; i++){
        new[i] = malloc(h*sizeof(float*));
        if(new[i]==NULL){
            fprintf(stderr, "Error: Unable to allocate memory to new[%d] in alloc_3D.",i);
            exit(EXIT_FAILURE);
        }
        for(j=0; j<h; j++){
            new[i][j] = malloc(w*sizeof(float));
            if(new[i][j]==NULL){
                fprintf(stderr, "Error: Unable to allocate memory to new[%d][%d] in alloc_3D.",i,j);
                exit(EXIT_FAILURE);
            }
        }
    }
    return new;
}


/** @ingroup utilityfunctions
 * Prints information about the given convolution layer
 * 
 * @param layer Convolution layer whose information is to be printed 
*/
void print_conv_details(ConvLayer layer){
    printf("Convolutional layer at %x\n\n", &layer);
    printf("\tn_kb = %d\n", layer.n_kb);
    printf("\tkernel_box_dims = %d,%d,%d\n", layer.kernel_box_dims[0], layer.kernel_box_dims[1], layer.kernel_box_dims[2]);
    printf("\tstride_x = %d\n", layer.stride_x);
    printf("\tstride_y = %d\n", layer.stride_y);
    printf("\tpadding = %d\n\n", layer.padding);

    int n,d,h,w;
    for(n=0; n<layer.n_kb; n++){
        printf("\tBox %d:\n", n);
        for(d=0; d<layer.kernel_box_dims[0]; d++){
            printf("\t\tLayer %d:\n", d);
            for(h=0; h<layer.kernel_box_dims[1]; h++){
                for(w=0; w<layer.kernel_box_dims[2]; w++){
                    printf("\t\t\t%f ", layer.kernel_box_group[n][d][h][w]);
                }
                printf("\n");
            }
        }
    }
}


/** @ingroup utilityfunctions
 * Frees the memory occupied by the tensor t
 * 
 * @param t The tensor to be freed
*/
void free_tensor(Tensor *t){
    int d,h;
    for(d=0; d<t->dims[0]; d++){
        for(h=0; h<t->dims[1]; h++){
            free(t->T[d][h]);
        }
        free(t->T[d]);
    }
    free(t->dims);
    free(t);
}


/** @ingroup utilityfunctions
 * Creates and configures a new tensor of dimensions (d * h * w)
 * 
 * @param d Depth of tensor
 * @param h Height of tensor
 * @param w Width of tensor
 * @param array The 3D float array with dimensions (d * h * w) from which the tensor is going to be built
 * 
 * @return The newly created tensor
*/
Tensor *make_tensor(int d, int h, int w, float ***array){
    Tensor *new_tensor;
    new_tensor = malloc(sizeof(Tensor));
    new_tensor->T = array;
    new_tensor->dims[0] = d;
    new_tensor->dims[1] = h;
    new_tensor->dims[2] = w;

    return new_tensor;
}

/** @ingroup utilityfunctions
 * Frees memory space allocated to the given Convolution Layer
 * 
 * @param layer Convolution Layer to be freed
*/
void free_ConvLayer(ConvLayer *layer){
    int n,d,h;
    for(n=0; n<layer->n_kb; n++){
        for(d=0; d<layer->kernel_box_dims[0]; d++){
            for(h=0; h<layer->kernel_box_dims[1]; h++){
                free(layer->kernel_box_group[n][d][h]);
            }
            free(layer->kernel_box_group[n][d]);
        }
        free(layer->kernel_box_group[n]);
    }
    free(layer->kernel_box_group);
    free(layer->bias_array);
    free(layer->kernel_box_dims);
    free(layer);
}

/** @ingroup utilityfunctions
 * Frees memory space allocated to the given Dense Layer
 * 
 * @param layer Dense Layer to be freed
*/
void free_DenseLayer(DenseLayer *layer){
    int n,d,h;
    for(n=0; n<layer->n_kb; n++){
        for(d=0; d<layer->kernel_box_dims[0]; d++){
            for(h=0; h<layer->kernel_box_dims[1]; h++){
                free(layer->kernel_box_group[n][d][h]);
            }
            free(layer->kernel_box_group[n][d]);
        }
        free(layer->kernel_box_group[n]);
    }
    free(layer->kernel_box_group);
    free(layer->bias_array);
    free(layer->kernel_box_dims);
    free(layer);
}