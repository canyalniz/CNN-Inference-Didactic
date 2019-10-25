import numpy as np
import keras
import sys
import os
from keras.preprocessing import image

def generate(model):
    """ Generates the driver for the inference library """

    weights = []
    config = []
    for layer in model.layers:
        weights.append(layer.get_weights())
        config.append(layer.get_config())
    
    f = open("model.c", "w+")
    f.write('#include "cnn_inference.h"\n#include "h5_format.h"\n#include "rgb_format.h"\n#include <time.h>\n')
    f.write('\nint main(int argc, char *argv[]){\n\n')
    f.write('\tif(argc!=2){\n\t\tprintf("\\nPlease run as: model.c <image_name>. Image name must be given without the 0,1,2 and without its extension.\\n");\n\t\texit(EXIT_FAILURE);\n\t}\n')
    f.write('\n\tclock_t start, end;\n\tdouble cpu_time_used;\n\n')
    
    f.write('\t//Prepare Input\n')
    f.write('\tfloat ***img;\n\timg = load_RGB(argv[1], {0}, {1});\n\n'.format(config[0]['batch_input_shape'][2], config[0]['batch_input_shape'][3]))
    f.write('\tTensor *t;\n\tt = make_tensor(3, {0}, {1}, img);\n\n'.format(config[0]['batch_input_shape'][2], config[0]['batch_input_shape'][3]))
    
    f.write('\t//Generate Layers\n')
    for layer in model.layers:
        if isinstance(layer, keras.layers.convolutional.Conv2D):
            f.write('\tConvLayer *_{0};\n\t_{0} = empty_Conv({1}, {2}, {3}, {4}, {5}, {6}, {7});\n'.format(layer.name, layer.filters, layer.input_shape[1], layer.kernel_size[0], layer.kernel_size[1], layer.strides[0], layer.strides[1], layer.padding.upper()))
            f.write('\tload_Conv(_{0}, "{0}");\n\n'.format(layer.name))
            
        elif isinstance(layer, keras.layers.core.Dense):
            f.write('\tDenseLayer *_{0};\n\t_{0} = empty_Dense({1}, {2}, 1, 1);\n'.format(layer.name, layer.units, layer.input_shape[1]))
            f.write('\tload_Dense(_{0}, "{0}");\n\n'.format(layer.name))
    
    
    f.write('\t//Inference\n')
    f.write('\tstart = clock();\n\n')
    for layer in model.layers:
        if isinstance(layer, keras.layers.convolutional.Conv2D):
            if layer.get_config()['activation'].upper() == "RELU":
                activation = "ReLU_activation"
            elif layer.get_config()['activation'].upper() == "LINEAR":
                activation = "linear_activation"
            elif layer.get_config()['activation'].upper() == "SIGMOID":
                activation = "sigmoid_activation"
            elif layer.get_config()['activation'].upper() == "ELU":
                activation = "ELU_activation"
    
            f.write('\tt = Conv(t, _{0}, {1}, 1);\n'.format(layer.name, activation))
        
        elif isinstance(layer, keras.layers.core.Dense):
            if layer.get_config()['activation'].upper() == "RELU":
                activation = "ReLU_activation"
            elif layer.get_config()['activation'].upper() == "LINEAR":
                activation = "linear_activation"
            elif layer.get_config()['activation'].upper() == "SIGMOID":
                activation = "sigmoid_activation"
            elif layer.get_config()['activation'].upper() == "ELU":
                activation = "ELU_activation"
            
            f.write('\tt = Dense(t, _{0}, {1}, 1);\n'.format(layer.name, activation))
        
        elif isinstance(layer, keras.layers.pooling.MaxPooling2D):
            f.write('\tt = MaxPool(t, {0}, {1}, {2}, {3}, {4}, 1);\n'.format(layer.pool_size[0], layer.pool_size[1], layer.strides[0], layer.strides[1], layer.padding.upper()))
        
        elif isinstance(layer, keras.layers.core.Flatten):
            f.write('\tt = FlattenD(t, 1);\n')
            
        elif isinstance(layer, keras.layers.core.Activation):
            if layer.get_config()['activation'].upper() == "RELU":
                f.write('\tt = ReLU_activation(t, 1);\n')
            elif layer.get_config()['activation'].upper() == "LINEAR":
                f.write('\tt = linear_activation(t, 1);\n')
            elif layer.get_config()['activation'].upper() == "SIGMOID":
                f.write('\tt = sigmoid_activation(t, 1);\n')
            elif layer.get_config()['activation'].upper() == "ELU":
                f.write('\tt = ELU_activation(t, 1);\n')
    
    f.write('\n\tend = clock();\n\n\tprint_tensor(t);\n\n\tcpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;\n\tprintf("\\nInference completed in: %f", cpu_time_used);\n\n')
    
    f.write("\t//Free Memory\n")
    for layer in model.layers:
        if isinstance(layer, keras.layers.convolutional.Conv2D):
            f.write("\tfree_ConvLayer(_{0});\n".format(layer.name))
            
        elif isinstance(layer, keras.layers.core.Dense):
            f.write("\tfree_DenseLayer(_{0});\n".format(layer.name))
    
    f.write('}')
    f.close()
    

def extract_weights(model):
    """ Extracts the weights of the given model to .txt files to work with the C library """

    path_weights = './model_weights_txt/'
    try:  
        os.mkdir(path_weights)
    except FileExistsError:
        pass


    for layer in model.layers:
        if isinstance(layer, keras.layers.convolutional.Conv2D):
            fw = open("{0}/{1}_weights.txt".format(path_weights, layer.name), "w+")
            fb = open("{0}/{1}_biases.txt".format(path_weights, layer.name), "w+")
            weights = np.transpose(layer.get_weights()[0])
            
            biases = layer.get_weights()[1]
            
            for n in range(layer.filters):
                fb.write(format(biases[n], ".10f") + " ")
                for d in range(layer.input_shape[1]):
                    for h in range(layer.kernel_size[0]):
                        for w in range(layer.kernel_size[1]):
                            fw.write(format(weights[n][d][w][h], ".10f") + " ")
            
            fw.close()
            fb.close()
            
        elif isinstance(layer, keras.layers.core.Dense):
            fw = open("{0}/{1}_weights.txt".format(path_weights, layer.name), "w+")
            fb = open("{0}/{1}_biases.txt".format(path_weights, layer.name), "w+")
            weights = np.transpose(layer.get_weights()[0])
            biases = layer.get_weights()[1]
            
            for n in range(layer.units):
                fb.write(format(biases[n], ".10f") + " ")
                for d in range(layer.input_shape[1]):
                    fw.write(format(weights[n][d], ".10f") + " ")
            
            fw.close()
            fb.close()

def main():
    if not len(sys.argv)==2:
        print("Please run as: generator.py <path to .h5 model file>")
        sys.exit()

    generate(keras.models.load_model(sys.argv[1]))
    extract_weights(keras.models.load_model(sys.argv[1]))

if __name__ == "__main__":
    main()