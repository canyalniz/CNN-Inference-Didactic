#include "cnn_inference.h"
ConvLayer *empty_Conv(int n_kb, int d_kb, int h_kb, int w_kb, int stride, int padding){
    ConvLayer *clp;
    clp = malloc(sizeof(ConvLayer)); //clp: Convolutional Layer Pointer
    if(clp==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to clp in new_Conv.");
		exit(EXIT_FAILURE);
    }

    KernelBox *boxes = malloc(n_kb*sizeof(KernelBox));
    if(boxes==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to boxes in new_Conv.");
		exit(EXIT_FAILURE);
    }

    int n,d,h,w;
    for(n=0; n<n_kb; n++){
        boxes[n].dims[0] = d_kb; //Kernel Box Depth
        boxes[n].dims[1] = h_kb; //Kernel Box Height
        boxes[n].dims[2] = w_kb; //Kernel Box Width
        boxes[n].KB = alloc_3D(d_kb, h_kb, w_kb);
    }

    clp->kernel_box_group = boxes;
    clp->n_kb = n_kb;
    clp->bias_array.size = n_kb;
    clp->kernel_box_dims[0]=d_kb;
    clp->kernel_box_dims[1]=h_kb;
    clp->kernel_box_dims[2]=w_kb;
    //The number of biases in our layers is equal to the depth of the output tensor, which is equal to the number of kernel boxes in the layer.
    //i.e. in Conv1 we have 32 biases, one for each layer of the output tensor which is 32x31x31
    clp->stride = stride;
    clp->padding = padding;
    clp->bias_array.B = malloc(n_kb*sizeof(float));

    return clp;
}


ConvLayer *new_Conv(int n_kb, int d_kb, int h_kb, int w_kb, float **** weights_array, float * biases_array, int stride, int padding, int copy){
    ConvLayer *clp;
    clp = malloc(sizeof(ConvLayer)); //clp: Convolutional Layer Pointer
    if(clp==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to clp in new_Conv.");
		exit(EXIT_FAILURE);
    }

    KernelBox *boxes = malloc(n_kb*sizeof(KernelBox));
    if(boxes==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to boxes in new_Conv.");
		exit(EXIT_FAILURE);
    }

    int n,d,h,w;
    for(n=0; n<n_kb; n++){
        boxes[n].dims[0] = d_kb; //Kernel Box Depth
        boxes[n].dims[1] = h_kb; //Kernel Box Height
        boxes[n].dims[2] = w_kb; //Kernel Box Width
        if(copy){
            boxes[n].KB = alloc_3D(d_kb, h_kb, w_kb);
            for(d=0; d<d_kb; d++){
                for(h=0; h<h_kb; h++){
                    for(w=0; w<w_kb; w++){
                        boxes[n].KB[d][h][w] = weights_array[n][d][h][w];
                    }
                }
            }
        } else {
            boxes[n].KB = weights_array[n];
        }
    }

    clp->kernel_box_group = boxes;
    clp->n_kb = n_kb;
    clp->bias_array.size = n_kb;
    clp->kernel_box_dims[0]=d_kb;
    clp->kernel_box_dims[1]=h_kb;
    clp->kernel_box_dims[2]=w_kb;
    //The number of biases in our layers is equal to the depth of the output tensor, which is equal to the number of kernel boxes in the layer.
    //i.e. in Conv1 we have 32 biases, one for each layer of the output tensor which is 32x31x31
    clp->stride = stride;
    clp->padding = padding;

    if(copy){
        clp->bias_array.B = malloc(n_kb*sizeof(float));
        for(n=0; n<n_kb; n++){
            clp->bias_array.B[n] = biases_array[n];
        }
    } else {
        clp->bias_array.B = biases_array;
    }

    return clp;
}

DenseLayer *empty_Dense(int n_kb, int d_kb, int h_kb, int w_kb){
    DenseLayer *dlp;
    dlp = malloc(sizeof(DenseLayer)); //dlp: Dense Layer Pointer
    if(dlp==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to dlp in new_Dense.");
		exit(EXIT_FAILURE);
    }

    KernelBox *boxes = malloc(n_kb*sizeof(KernelBox));
    if(boxes==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to boxes in new_Dense.");
		exit(EXIT_FAILURE);
    }

    int n,d,h,w;
    for(n=0; n<n_kb; n++){
        boxes[n].dims[0] = d_kb; //Kernel Box Depth
        boxes[n].dims[1] = h_kb; //Kernel Box Height
        boxes[n].dims[2] = w_kb; //Kernel Box Width
        boxes[n].KB = alloc_3D(d_kb, h_kb, w_kb);
    }

    dlp->kernel_box_group = boxes;
    dlp->n_kb = n_kb;
    dlp->kernel_box_dims[0] = d_kb;
    dlp->kernel_box_dims[1] = h_kb;
    dlp->kernel_box_dims[2] = w_kb;
    dlp->bias_array.B = malloc(n_kb*sizeof(float));

    /*
    As you may have noticed Dense layers and Conv layers are basically identical.
    The only difference is that the kernel boxes in the Dense layers have the same dimensions as the input tensor(and there are no strides or paddings).
    Because we're Flattening before feeding the tensor to the Dense Layer this is much more like a traditional NN layer.
    Also, because we're Flattening, you can basically think of n_kb as the number of neurons in this Dense layer.
    */

   return dlp;
}

DenseLayer *new_Dense(int n_kb, int d_kb, int h_kb, int w_kb, float **** weights_array, float * biases_array, int copy){
    DenseLayer *dlp;
    dlp = malloc(sizeof(DenseLayer)); //dlp: Dense Layer Pointer
    if(dlp==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to dlp in new_Dense.");
		exit(EXIT_FAILURE);
    }

    KernelBox *boxes = malloc(n_kb*sizeof(KernelBox));
    if(boxes==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to boxes in new_Dense.");
		exit(EXIT_FAILURE);
    }

    int n,d,h,w;
    for(n=0; n<n_kb; n++){
        boxes[n].dims[0] = d_kb; //Kernel Box Depth
        boxes[n].dims[1] = h_kb; //Kernel Box Height
        boxes[n].dims[2] = w_kb; //Kernel Box Width
        if(copy){
            boxes[n].KB = alloc_3D(d_kb, h_kb, w_kb);
            for(d=0; d<d_kb; d++){
                for(h=0; h<h_kb; h++){
                    for(w=0; w<w_kb; w++){
                        boxes[n].KB[d][h][w] = weights_array[n][d][h][w];
                    }
                }
            }
        } else {
            boxes[n].KB = weights_array[n];
        }
    }

    dlp->kernel_box_group = boxes;
    dlp->n_kb = n_kb;
    dlp->kernel_box_dims[0] = d_kb;
    dlp->kernel_box_dims[1] = h_kb;
    dlp->kernel_box_dims[2] = w_kb;

    if(copy){
        dlp->bias_array.B = malloc(n_kb*sizeof(float));
        for(n=0; n<n_kb; n++){
            dlp->bias_array.B[n] = biases_array[n];
        }
    } else {
        dlp->bias_array.B = biases_array;
    }
    dlp->bias_array.size = n_kb;

    return dlp;

    /*
    As you may have noticed Dense layers and Conv layers are basically identical.
    The only difference is that the kernel boxes in the Dense layers have the same dimensions as the input tensor(and there are no strides or paddings).
    Because we're Flattening before feeding the tensor to the Dense Layer this is much more like a traditional NN layer.
    Also, because we're Flattening, you can basically think of n_kb as the number of neurons in this Dense layer.
    */
}

Tensor Conv(Tensor input, ConvLayer *layer, int dispose_of_input, Tensor (*activation)(Tensor,int)){
    if(input->dims[0]!=layer->kernel_box_dims[0]){
        fprintf(stderr, "Error: The depth of the kernel boxes in this layer(%d) and that of its input tensor(%d) must match", layer->kernel_box_dims[0], input->dims[0]);
        exit(EXIT_FAILURE);
    }

    if(layer->padding!=0){
        input = apply_padding(input,layer->padding,dispose_of_input);
    }

    int output_d = layer->n_kb;
    int output_w, output_h;
    output_h = ((input->dims[1] /*+ 2*layer->padding */ - layer->kernel_box_dims[1])/layer->stride)+1;
    output_w = ((input->dims[2] /*+ 2*layer->padding */ - layer->kernel_box_dims[2])/layer->stride)+1;
    //This is just the formula for getting the output height and width given the input dimensions, padding, kernel(filter) dimensions and stride
    //In our case output_h=output_w as we have square kernels(filters)
    
    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int d,h,w,id,by,bx,i,j;
    float result;

    // This thing goes over the output array and calculates each cell's value one by one
    for(d=0; d<output_d; d++){ //output depth
        for(h=0; h<output_h; h++){ //output height
            for(w=0; w<output_w; w++){ //output width
                result = 0; //this will hold the sum of the convolutions over each "channel" of the input tensor(the sum over its depth)
                for(id=0; id<input->dims[0]; id++){ //input depth
                    by = h*layer->stride; //"begin y" defines where the top edge of the kernel window is on the input layer
                    bx = w*layer->stride; //"begin x" defines where the left edge of the kernel window is on the input layer
                    for(i=0; i<(layer->kernel_box_dims[1]); i++){ //traverses the height of kernel window
                        for(j=0; j<(layer->kernel_box_dims[2]); j++){ //traverses the width of kernel window
                            result += input->T[id][by+i][bx+j] * layer->kernel_box_group[d].KB[id][i][j];
                        }
                    }
                }
                
                //Add the bias
                result += layer->bias_array.B[d];
                output_array[d][h][w] = result;
            }
        }
    }
    
    Tensor output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    output = activation(output, 1);
    
    if(dispose_of_input) free_tensor(input);
    
    return output;
}

Tensor Dense(Tensor input, DenseLayer *layer, int dispose_of_input, Tensor (*activation)(Tensor,int)){
    if(input->dims[0]!=layer->kernel_box_dims[0] || input->dims[1]!=layer->kernel_box_dims[1] || input->dims[2]!=layer->kernel_box_dims[2]){
        fprintf(stderr,"Error: The dimensions of the kernel boxes of the Dense layer must exactly match those of the input tensor.\n");
        fprintf(stderr,"input has d:%d h:%d w:%d | kernel boxes have d:%d h:%d w:%d", input->dims[0], input->dims[1], input->dims[2], layer->kernel_box_dims[0], layer->kernel_box_dims[1], layer->kernel_box_dims[2]);
        exit(EXIT_FAILURE);
    }

    int output_d = layer->n_kb;
    int output_w =1, output_h = 1; //You can get this from the formula above as well
    
    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int d,h,w,id,i,j;
    float result;
    
    // This thing goes over the output array and calculates each cell's value one by one
    for(d=0; d<output_d; d++){ //output depth
        for(h=0; h<output_h; h++){ //output height
            for(w=0; w<output_w; w++){ //output width
                result = 0;
                for(id=0; id<input->dims[0]; id++){ //input depth, usually 1 for Dense layers as they are usually preceded by a Flattening operation
                    for(i=0; i<layer->kernel_box_dims[1]; i++){ //traverses the height of kernel window
                        for(j=0; j<layer->kernel_box_dims[2]; j++){ //traverses the width of kernel window
                            result += input->T[id][i][j] * layer->kernel_box_group[d].KB[id][i][j];
                        } //here by and bx are both 0 and they never change as the kernel dimensions are equal to the input tensor layer dimensions
                    }
                }

                //Add the bias
                result += layer->bias_array.B[d];
                output_array[d][h][w] = result;
            }
        }
    }

    Tensor output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    output = activation(output, 1);

    if(dispose_of_input) free_tensor(input);

    return output;
}

Tensor sigmoid_activation(Tensor input, int dispose_of_input){
    Tensor output;
    int d,h,w;

    if(dispose_of_input){
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

Tensor ReLU_activation(Tensor input, int dispose_of_input){
    Tensor output;
    int d,h,w;

    if(dispose_of_input){
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

Tensor linear_activation(Tensor input, int dispose_of_input){
    Tensor output;
    int d,h,w;

    if(dispose_of_input){
        output = input;
    } else {
        float ***output_array = alloc_3D(input->dims[0], input->dims[1], input->dims[2]);
        output = make_tensor(input->dims[0], input->dims[1], input->dims[2], output_array);
    }

    for(d=0; d<output->dims[0]; d++){
        for(h=0; h<output->dims[1]; h++){
            for(w=0; w<output->dims[2]; w++){
                output->T[d][h][w] = input->T[d][h][w];
            }
        }
    }

    return output;
}

Tensor apply_padding(Tensor input, int padding, int dispose_of_input){
    int output_d = input->dims[0];
    int output_h = input->dims[1] + 2*padding;
    int output_w = input->dims[2] + 2*padding;

    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int d,h,w,x,y;
    
    for(d=0; d<output_d; d++){
        //pad top and bottom
        for(x=0; x<output_w; x++){
            output_array[d][0][x] = output_array[d][output_h-1][x] = 0;
        }
        //pad left and right
        for(y=0; y<output_h; y++){
            output_array[d][y][0] = output_array[d][y][output_w-1] = 0;
        }
        //load the middle
        for(x=padding; x<(output_w-padding); x++){
            for(y=padding; y<(output_h-padding); y++){
                output_array[d][y][x] = input->T[d][y-padding][x-padding];
            }    
        }
    }

    Tensor output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if(dispose_of_input) free_tensor(input);

    return output;
}

Tensor MaxPool(Tensor input, int height, int width, int stride, int dispose_of_input){
    int output_d = input->dims[0];
    int output_w, output_h;
    output_w = output_h = ((input->dims[1] - height)/stride)+1; // The same formula from the Conv layer

    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int d,h,w,i,j,by,bx;
    float max;

    // This thing goes over the output array and calculates each cell's value one by one
    for(d=0; d<output_d; d++){ //output depth
        for(h=0; h<output_h; h++){ //output height
            for(w=0; w<output_w; w++){ //output width
                by = h*stride;
                bx = w*stride;
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

    Tensor output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if(dispose_of_input) free_tensor(input);

    return output;
}

Tensor FlattenW(Tensor input, int dispose_of_input){
    int input_d = input->dims[0], input_h = input->dims[1], input_w = input->dims[2];

    int output_d = 1, output_h = 1;
    int output_w = input_d*input_h*input_w;

    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int w;

    for(w=0; w<output_w; w++){
        output_array[0][0][w] = input->T[w/(input_h*input_w)][(w/input_w)%input_h][w%input_w];
    }

    Tensor output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if(dispose_of_input) free_tensor(input);

    return output;
}

Tensor FlattenH(Tensor input, int dispose_of_input){
    int input_d = input->dims[0], input_h = input->dims[1], input_w = input->dims[2];

    int output_d = 1, output_w = 1;
    int output_h = input_d*input_h*input_w;

    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int h;

    for(h=0; h<output_h; h++){
        output_array[0][h][0] = input->T[h/(input_h*input_w)][(h/input_w)%input_h][h%input_w];
    }

    Tensor output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if(dispose_of_input) free_tensor(input);

    return output;
}

Tensor FlattenD(Tensor input, int dispose_of_input){
    int input_d = input->dims[0], input_h = input->dims[1], input_w = input->dims[2];

    int output_w = 1, output_h = 1;
    int output_d = input_d*input_h*input_w;

    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int d;

    for(d=0; d<output_d; d++){
        output_array[d][0][0] = input->T[d/(input_h*input_w)][(d/input_w)%input_h][d%input_w];
    }

    Tensor output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if(dispose_of_input) free_tensor(input);

    return output;
}

void print_tensor(Tensor t){
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

void print_conv_details(ConvLayer layer){
    printf("Convolutional layer at %x\n\n", &layer);
    printf("\tn_kb = %d\n", layer.n_kb);
    printf("\tkernel_box_dims = %d,%d,%d\n", layer.kernel_box_dims[0], layer.kernel_box_dims[1], layer.kernel_box_dims[2]);
    printf("\tstride = %d\n", layer.stride);
    printf("\tpadding = %d\n\n", layer.padding);

    int n,d,h,w;
    for(n=0; n<layer.n_kb; n++){
        printf("\tBox %d:\n", n);
        for(d=0; d<layer.kernel_box_group[n].dims[0]; d++){
            printf("\t\tLayer %d:\n", d);
            for(h=0; h<layer.kernel_box_group[n].dims[1]; h++){
                for(w=0; w<layer.kernel_box_group[n].dims[2]; w++){
                    printf("\t\t\t%f ", layer.kernel_box_group[n].KB[d][h][w]);
                }
                printf("\n");
            }
        }
    }
}

void free_tensor(Tensor t){
    int d,h;
    for(d=0; d<t->dims[0]; d++){
        for(h=0; h<t->dims[1]; h++){
            free(t->T[d][h]);
        }
        free(t->T[d]);
    }
    free(t->dims);
    //free(t);
}

Tensor make_tensor(int d, int h, int w, float ***array){
    Tensor new_tensor;
    new_tensor = malloc(sizeof(struct tensor));
    new_tensor->T = array;
    new_tensor->dims[0] = d;
    new_tensor->dims[1] = h;
    new_tensor->dims[2] = w;
}