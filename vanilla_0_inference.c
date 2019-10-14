#include "cnn_inference.h"
#include "h5_format.h"
#include "jpg_format.h"
#include <time.h>

int main(){
    clock_t start, end;
    double cpu_time_used;

    char fn[3][MAX_FN] = {"img_0.txt", "img_1.txt", "img_2.txt"};

    float ***img;
    img = load_RGB(fn, 63, 63); //loads the jpg
    
    Tensor *input;
    input = make_tensor(3, 63, 63, img);


    //Before you can load the weights of the model you need to run weights_to_txt.py
    //The txt files need to be in the directory
    ConvLayer *conv1;
    conv1 = empty_Conv(32, 3, 3, 3, 2, 2, 0);
    load_Conv(conv1, 1); 

    ConvLayer *conv2;
    conv2 = empty_Conv(32, 32, 3, 3, 1, 1, 1);
    load_Conv(conv2, 4);
    
    DenseLayer *fc;
    fc = empty_Dense(512, 1568, 1, 1);
    load_Dense(fc, 8);

    DenseLayer *out;
    out = empty_Dense(1, 512, 1, 1);
    load_Dense(out, 10);

    Tensor *x, *output;

    start = clock();
    x = Conv(input, conv1, ReLU_activation, 1);
    x = MaxPool(x, 3, 3, 2, 2, 1);
    x = Conv(x, conv2, ReLU_activation, 1);
    x = MaxPool(x, 3, 3, 2, 2, 1);
    x = FlattenD(x, 1);
    x = Dense(x, fc, ReLU_activation, 1);
    output = Dense(x, out, linear_activation, 1); //This used to be a sigmoid activation but for clarity I changed the model's activation to linear
    end = clock();
    print_tensor(output);
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nInference completed in: %f", cpu_time_used);

    free_ConvLayer(conv1);
    free_ConvLayer(conv2);
    free_DenseLayer(fc);
    free_DenseLayer(out);

    return 0;
}