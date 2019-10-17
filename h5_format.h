#ifndef H5_FORMAT
#define H5_FORMAT
#include "cnn_inference.h"

#define MAX_FN 50

void load_Conv(ConvLayer *layer, char *layer_name);
void load_Dense(DenseLayer *layer, char *layer_name);

#endif