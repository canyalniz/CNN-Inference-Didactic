#include "cnn_inference.h"
#include "h5_format.h"
#include "jpg_format.h"
#include <time.h>

int main(){
    clock_t start, end;
    double cpu_time_used;

    int f;
    char **fn;
    fn = malloc(3*sizeof(char*)); //This array is going to hold the three channels of the jpg, you need to run jpg_to_txt.py first
    for(f=0; f<3; f++){
        fn[f] = malloc(MAX_FN*sizeof(char));
    }
    
    fn[0] = "img_0.txt"; //Channel 1 file name
    fn[1] = "img_1.txt"; //Channel 2 file name
    fn[2] = "img_2.txt"; //Channel 3 file name

    float ***img;
    img = load_RGB(fn, 63, 63); //loads the jpg
    
    Tensor input;
    input = make_tensor(3, 63, 63, img);


    //Before you can load the weights of the model you need to run weights_to_txt.py
    //The txt files need to be in the directory
    ConvLayer *conv1;
    conv1 = empty_Conv(32, 3, 3, 3, 2, 0);
    load_Conv(conv1, 1); 

    ConvLayer *conv2;
    conv2 = empty_Conv(32, 32, 3, 3, 1, 1);
    load_Conv(conv2, 4);
    
    DenseLayer *fc;
    fc = empty_Dense(512, 1568, 1, 1);
    load_Dense(fc, 8);

    DenseLayer *out;
    out = empty_Dense(1, 512, 1, 1);
    load_Dense(out, 10);

    Tensor x, output;

    start = clock();
    x = Conv(input, conv1, 1, ReLU_activation);
    x = MaxPool(x, 3, 3, 2, 1);
    x = Conv(x, conv2, 0, ReLU_activation);
    x = MaxPool(x, 3, 3, 2, 1);
    x = FlattenD(x, 1);
    x = Dense(x, fc, 1, ReLU_activation);
    output = Dense(x, out, 1, linear_activation); //This used to be a sigmoid activation but for clarity I changed the model's activation to linear
    end = clock();
    print_tensor(output);
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nTime taken: %f", cpu_time_used);

    return 0;
}