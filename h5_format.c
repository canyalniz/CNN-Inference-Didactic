#include "h5_format.h"

void load_Conv(ConvLayer *layer, char *layer_name){
    FILE *fp;
    char filename[MAX_FN];
    int n,d,h,w;

    sprintf(filename, "model_weights_txt/%s_weights.txt", layer_name);
    if((fp = fopen(filename, "r"))==NULL){
        fprintf(stderr, "Error, unable to access %s", filename);
        exit(EXIT_FAILURE);
    }

    for(n=0; n<layer->n_kb; n++){
        for(d=0; d<layer->kernel_box_dims[0]; d++){
            for(h=0; h<layer->kernel_box_dims[2]; h++){
                for(w=0; w<layer->kernel_box_dims[1]; w++){
                    fscanf(fp, "%f", &(layer->kernel_box_group[n][d][h][w]));
                }
            }
        }
    }

    fclose(fp);

    sprintf(filename, "model_weights_txt/%s_biases.txt", layer_name);
    if((fp=fopen(filename, "r"))==NULL){
        fprintf(stderr, "Error, unable to access %s", filename);
        exit(EXIT_FAILURE);
    }

    for(n=0; n<layer->n_kb; n++){
        fscanf(fp, "%f", &(layer->bias_array[n]));
    }
    fclose(fp);
}

void load_Dense(DenseLayer *layer, char *layer_name){
    FILE *fp;
    char filename[MAX_FN];
    int n,d,h,w;

    sprintf(filename, "model_weights_txt/%s_weights.txt", layer_name);
    if((fp = fopen(filename, "r"))==NULL){
        fprintf(stderr, "Error, unable to access %s", filename);
        exit(EXIT_FAILURE);
    }

    for(n=0; n<layer->n_kb; n++){
        for(d=0; d<layer->kernel_box_dims[0]; d++){
            for(h=0; h<layer->kernel_box_dims[1]; h++){
                for(w=0; w<layer->kernel_box_dims[2]; w++){
                    fscanf(fp, "%f", &(layer->kernel_box_group[n][d][h][w]));
                }
            }
        }
    }

    fclose(fp);

    sprintf(filename, "model_weights_txt/%s_biases.txt", layer_name);
    if((fp=fopen(filename, "r"))==NULL){
        fprintf(stderr, "Error, unable to access %s", filename);
        exit(EXIT_FAILURE);
    }

    for(n=0; n<layer->n_kb; n++){
        fscanf(fp, "%f", &(layer->bias_array[n]));
    }
    fclose(fp);
}