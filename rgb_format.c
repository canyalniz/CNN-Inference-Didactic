#include "rgb_format.h"

float ***load_RGB(char *image_name, int h, int w){
    FILE *fp;
    char file_name[MAX_FN];
    int f,i,j;


    float ***img;
    img = malloc(3*sizeof(float**));
    if(img==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to img in load_RGB.");
		exit(EXIT_FAILURE);
    }

    for(i=0; i<3; i++){
        img[i] = malloc(h*sizeof(float*));
        if(img[i]==NULL){
            fprintf(stderr, "Error: Unable to allocate memory to img[%d] in load_RGB.",i);
            exit(EXIT_FAILURE);
        }
        for(j=0; j<h; j++){
            img[i][j] = malloc(w*sizeof(float));
            if(img[i][j]==NULL){
                fprintf(stderr, "Error: Unable to allocate memory to img[%d][%d] in load_RGB.",i,j);
                exit(EXIT_FAILURE);
            }
        }
    }

    for(f=0; f<3; f++){
        sprintf(file_name, "input_images_txt/%s_%d.txt", image_name, f);
        if((fp=fopen(file_name, "r"))==NULL){
            fprintf(stderr, "Error, unable to access %s", file_name);
            exit(EXIT_FAILURE);
        }

        for(i=0; i<h; i++){
            for(j=0; j<w; j++){
                fscanf(fp, "%f", &(img[f][i][j]));
            }
        }

        fclose(fp);
    }

    return img;
}
