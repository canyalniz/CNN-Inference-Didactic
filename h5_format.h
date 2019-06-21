#ifndef H5_FORMAT
#define H5_FORMAT
#include "cnn_inference.h"

#define MAX_FN 50

void load_Conv(ConvLayer *layer, int layer_id);
void load_Dense(DenseLayer *layer, int layer_id);

#endif