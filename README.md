# CNN-Inference-Didactic

In this repositary you will find a C library written with didactic intentions using straight-forward algorithms, simple structures and lots of comments.

## Channels First

This library uses a channels first convention. Anyone can read the code with this in mind, but in order to run experiments by replicating your model, its weights must be configured accordingly.

## Preparing the Input

The input to your Convolutional Neural Network must be an RGB image. In order to prepare your input images for inference, you must run **rgb_prep.py** with the arguments `<path/to/image> <CNN input height> <CNN input width>`. This will automatically generate the input files for the C model to read.

## Replicating Your Model

In order to experiment with this library, it is necessary to build a replica of your model in C. If your model is a *[Keras Sequential Model](https://keras.io/models/sequential/)*  or was built in Keras in a sequential way, then you may choose to automate this process by interfacing with Keras directly(see below). Otherwise you will have to manually build your model using the library functions. See **sample_model.c** for an example model.

## Extracting Trained Weights

Once you've built your model, you will need to load your trained weights into it. This is handled by the functions in **h5_format.c**. As the name suggests, the functions in this file were designed to work with .h5 style weights from Keras. Even if your model isn't sequential, all you need to do is to run `generator.extract_weights()` as shown below with your model's h5py file. This will generate the necessary files for the C model. All you need to do is to keep track of the names of your layers as shown in **sample_model.c**. **_This framework assumes all your layers have unique names._**

	import keras
	import generator

	generator.extract_weights(keras.models.load_model("path/to/your/model"))


## Interfacing With Keras

If you have a *[Keras Sequential Model](https://keras.io/models/sequential/)*, then you may choose to directly interface with Keras and jump right into running your experiments after preparing your input(see above). In order to generate your model in C automatically, you can either run **generator.py** with the argument `<path/to/your/model>` or you may call the generator functions from another scripts like this:

	import keras
	import generator

	model = keras.models.load_model("path/to/your/model")
	generator.generate(model)
    generator.extract_weights(model)


## Running Your Experiment

After successfully preparing your input, replicating your model and extracting your trained weights it is time for running your inference.

Run

	gcc model.c cnn_inference.c h5_format.c rgb_format.c -o infer.exe

to compile.