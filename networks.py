import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops as cfo

def log(x, base):
    return tf.log(x) / tf.log(base)

def log2(x): return 0 if (x == 1).all() else 1 + log2(x >> 1)

def lerp(a, b, t):
    return t * a + (1.0 - t) * b

min_resolution = np.asanyarray([2, 16])
max_resolution = np.asanyarray([128, 1024])
min_channels = 32
max_channels = 256
min_depth = 0 
max_depth = 6 

def upscale2d(inputs, factors=[2, 2]):
    factors = np.asanyarray(factors)
    if (factors == 1).all():
        return inputs
    shape = inputs.shape.as_list()
    inputs = tf.reshape(inputs, [-1, shape[1], shape[2], 1, shape[3], 1])
    inputs = tf.tile(inputs, [1, 1, 1, factors[0], 1, factors[1]])
    inputs = tf.reshape(inputs, [-1, shape[1], shape[2] * factors[0], shape[3] * factors[1]])
    return inputs


###############################################################################################################################
def generator(latents, labels, growing_level, name="generator", reuse=tf.AUTO_REUSE):
    
    
    growing_depth = log(1 + ((1 << (max_depth + 1)) - 1) * growing_level, 2.0)

    def get_resolution(depth):
        res_values = {
            0: [2,16],
            1: [4,32],
            2: [8,64],
            3: [16,128],
            4: [32,256],
            5: [64,512],
            6: [128,1024],
        }[depth]  
        
        return np.asanyarray(res_values)
    
    def channels(depth):
        channel_values = {
            0: 256,
            1: 256,
            2: 256,
            3: 256,
            4: 128,
            5: 64,
            6: 32,
        }[depth]
        
        return np.asanyarray(channel_values)
        
    def pixel_normalization_(inputs, epsilon=1.0e-12):
        pixel_norm = tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + epsilon)
        inputs = inputs / pixel_norm
        return inputs     
    
    # Reshape the size of the image in input increasing it 
    def upscale2d(inputs, factors=[2, 2]):
        factors = np.asanyarray(factors)
        if (factors == 1).all():
            return inputs
        shape = inputs.shape.as_list()
        inputs = tf.reshape(inputs, [-1, shape[1], shape[2], 1, shape[3], 1])
        inputs = tf.tile(inputs, [1, 1, 1, factors[0], 1, factors[1]])
        inputs = tf.reshape(inputs, [-1, shape[1], shape[2] * factors[0], shape[3] * factors[1]])
        return inputs
    
    def conv_layer(depth, inputs):
        
        #get the shape
        filters=channels(depth)                        
        kernel_size=[3, 3]
        # [k_width, k_height, in_channels, out_channels]
        shape=[kernel_size[0], kernel_size[1], inputs.shape[1].value, filters] 

        # Compute the initializer
        variance_scale=2.0
        stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
        initializer = tf.initializers.truncated_normal(0.0, 1.0)
        # get the weight from the model
        weight = tf.get_variable(name="weight", shape=shape, initializer=initializer) * stddev 
        # Compute the 2-D convolution of the input
        inputs = tf.nn.conv2d(input=inputs, filter=weight, strides=[1, 1, 1, 1], padding="SAME", data_format="NCHW")
        # get bias
        initializer=tf.initializers.zeros()
        bias = tf.get_variable(name="bias",shape=[inputs.shape[1].value],initializer=initializer)
        # add the bias
        inputs = tf.nn.bias_add(inputs, bias, data_format="NCHW")
        # apply a non linear function to the output    
        inputs = tf.nn.leaky_relu(inputs)
        # pixel normalization
        inputs = pixel_normalization_(inputs)
        
        return inputs

    def conv_block(inputs, depth, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("conv_block_{}x{}".format(*get_resolution(depth)), reuse=reuse):
            # when depth = 0
            if depth == min_depth:
                #Pixel normalization
                inputs = pixel_normalization_(inputs)
               
                # Dense layer
                with tf.variable_scope("dense"):
                    #get the number of units
                    units=channels(depth) * get_resolution(depth).prod()
                    shape=[inputs.shape[1].value, units]
                    
                    variance_scale=2.0
                    stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
                    initializer = tf.initializers.truncated_normal(0.0, 1.0)
                    # get the weights from the model
                    weight = tf.get_variable(name="weight",shape=shape,initializer=initializer)
                    weight = weight * stddev
                    
                    # multiply the input with the weight
                    inputs = tf.matmul(inputs, weight)
                    initializer=tf.initializers.zeros()
                    # get the bias
                    shape = [inputs.shape[1].value]
                    bias = tf.get_variable(name="bias",shape=[inputs.shape[1].value],initializer=initializer)
                    # add the bias to the previus operation(inputs x weights) 
                    inputs = tf.nn.bias_add(inputs, bias)
                    # reshape the output
                    inputs = tf.reshape( tensor=inputs, shape=[-1, channels(depth), *get_resolution(depth)])                      
                    # apply a non linear function to the output 
                    inputs = tf.nn.leaky_relu(inputs)
                    # pixel normalization
                    inputs = pixel_normalization_(inputs) 
                    
                # Convolutional layer
                with tf.variable_scope("conv"):
                    inputs = conv_layer(depth,inputs)
                    
                return inputs
            # for all depth diverse from 0    
            else:
                # upscale layer
                with tf.variable_scope("upscale_conv"):
                    # get the shape
                    strides=[2, 2]
                    filters=channels(depth)
                    kernel_size=[3, 3]
                    # [k_width, k_height, in_channels, out_channels]
                    shape=[kernel_size[0], kernel_size[1], inputs.shape[1].value, filters]
                    
                    # compute the initializer 
                    variance_scale=2.0
                    stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
                    initializer = tf.initializers.truncated_normal(0.0, 1.0)
                   
                    # get the weight from the model
                    weight = tf.get_variable(name="weight",shape=shape,initializer=initializer) * stddev
                    weight = tf.transpose(weight, [0, 1, 3, 2])
                    
                    # get the input shape
                    input_shape = np.array(inputs.shape.as_list())
                    
                    # compute the output shape 
                    output_shape = [input_shape[0], filters, *input_shape[2:] * strides]
                    
                    #Compute the transpose of a 2-D convolution of the input
                    inputs = tf.nn.conv2d_transpose( 
                            value=inputs,
                            filter=weight,
                            output_shape=output_shape,
                            strides=[1, 1, 2, 2],
                            padding="SAME",
                            data_format="NCHW" )
                    
                    # apply a non linear function to the output 
                    inputs = tf.nn.leaky_relu(inputs)
                    # pixel normalization
                    inputs = pixel_normalization_(inputs) 
                    
                # Convolutional layer    
                with tf.variable_scope("conv"):
                    inputs = conv_layer(depth, inputs)
                   
                return inputs

    def color_block(inputs, depth, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("color_block_{}x{}".format(*get_resolution(depth)), reuse=reuse):
            # Convolutional Layer
            with tf.variable_scope("conv"):
                # get the shape
                filters=2                        
                kernel_size=[1, 1]
                shape=[kernel_size[0], kernel_size[1], inputs.shape[1].value, filters] 
                
                # compute the initializer
                variance_scale=1.0
                stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
                initializer=tf.initializers.truncated_normal(0.0, 1.0)
                # get the weight from the model
                weight = tf.get_variable(name="weight",shape=shape,initializer=initializer) * stddev    
                # Compute the 2-D convolution of the input
                inputs = tf.nn.conv2d(input=inputs, filter=weight, strides=[1, 1, 1, 1], padding="SAME", data_format="NCHW")
                
                # get bias
                initializer=tf.initializers.zeros()
                bias = tf.get_variable(name="bias",shape=[inputs.shape[1].value],initializer=initializer)
                # add the bias
                inputs = tf.nn.bias_add(inputs, bias, data_format="NCHW")
                
                # apply the hyperbolic function
                inputs = tf.nn.tanh(inputs)
            
            return inputs

    def grow(feature_maps, depth):

        def high_resolution_images():
            return grow(conv_block(feature_maps, depth), depth + 1)

        def middle_resolution_images():
            return upscale2d(
                inputs=color_block(conv_block(feature_maps, depth), depth),
                factors=get_resolution(max_depth) // get_resolution(depth)
                    )

        def low_resolution_images():
            return upscale2d(
                inputs=color_block(feature_maps, depth - 1),
                factors=get_resolution(max_depth) // get_resolution(depth - 1)
            )

        if depth == min_depth:

            images = tf.cond(
                    pred=tf.greater(growing_depth, depth),
                    true_fn=high_resolution_images,
                    false_fn=middle_resolution_images
                )
        elif depth == max_depth:
            with tf.variable_scope("image_elif"):
                images = tf.cond(
                    pred=tf.greater(growing_depth, depth),
                    true_fn=middle_resolution_images,
                    false_fn=lambda: lerp(
                        a=low_resolution_images(),
                        b=middle_resolution_images(),
                        t=depth - growing_depth
                    )
                )
        else:
            with tf.variable_scope("image_else"):
                images = tf.cond(
                    pred=tf.greater(growing_depth, depth),
                    true_fn=high_resolution_images,
                    false_fn=lambda: lerp(
                        a=low_resolution_images(),
                        b=middle_resolution_images(),
                        t=depth - growing_depth
                    )
                )
        return images

    with tf.variable_scope(name, reuse=reuse):
        
        shape=[labels.shape[1].value, latents.shape[1]]
        
        variance_scale = 1.0
        stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
        initializer=tf.initializers.truncated_normal(0.0, 1.0)
        weight = tf.get_variable( name="weight",shape=shape,initializer=initializer) * stddev
        
        labels = tf.nn.embedding_lookup(weight, tf.argmax(labels, axis=1))
        
        return grow(tf.concat([latents, labels], axis=1), min_depth)


######################################################################################################################################################
def discriminator(images, labels, growing_level, name="discriminator", reuse=tf.AUTO_REUSE):

    growing_depth = log(1 + ((1 << (max_depth + 1)) - 1) * growing_level, 2.0)
    
    def get_resolution(depth):
        res_values = {
            0: [2,16],
            1: [4,32],
            2: [8,64],
            3: [16,128],
            4: [32,256],
            5: [64,512],
            6: [128,1024],
        }[depth]  
        
        return np.asanyarray(res_values)
    
    def channels(depth):
        channel_values = {
           -1: 256,
            0: 256,
            1: 256,
            2: 256,
            3: 256,
            4: 128,
            5: 64,
            6: 32,
        }[depth]
        
        return np.asanyarray(channel_values)
        
    def downscale2d(inputs, factors=[2, 2]):
        factors = np.asanyarray(factors)
        if (factors == 1).all():
            return inputs
        # perform the average pooling 
        inputs = tf.nn.avg_pool( 
            value=inputs,
            ksize=[1, 1, *factors],
            strides=[1, 1, *factors],
            padding="SAME",
            data_format="NCHW"
        )
        return inputs
    
    def batch_stddev(inputs, groups=4, epsilon=1.0e-12):
        shape = inputs.shape
        inputs = tf.reshape(inputs, [groups, -1, *shape[1:]])
        inputs -= tf.reduce_mean(inputs, axis=0, keepdims=True)
        inputs = tf.square(inputs)
        inputs = tf.reduce_mean(inputs, axis=0)
        inputs = tf.sqrt(inputs + epsilon)
        inputs = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
        inputs = tf.tile(inputs, [groups, 1, *shape[2:]])
        return inputs
    
    def conv_layer(depth, inputs):
        
        #get the shape
        filters=channels(depth)                        
        kernel_size=[3, 3]
        # [k_width, k_height, in_channels, out_channels]
        shape=[kernel_size[0], kernel_size[1], inputs.shape[1].value, filters] 

        # Compute the initializer
        variance_scale=2.0
        stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
        initializer = tf.initializers.truncated_normal(0.0, 1.0)
        
        # get the weight from the model
        weight = tf.get_variable(name="weight", shape=shape, initializer=initializer) * stddev 
        # Compute the 2-D convolution of the input
        inputs = tf.nn.conv2d(input=inputs, filter=weight, strides=[1, 1, 1, 1], padding="SAME", data_format="NCHW")
       
        # get bias
        initializer=tf.initializers.zeros()
        bias = tf.get_variable(name="bias",shape=[inputs.shape[1].value],initializer=initializer)
        # add the bias
        inputs = tf.nn.bias_add(inputs, bias, data_format="NCHW")
       
        # apply a non linear function to the output    
        inputs = tf.nn.leaky_relu(inputs)

        return inputs
            
    def conv_block(inputs, depth, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("conv_block_{}x{}".format(*get_resolution(depth)), reuse=reuse):
            if depth == min_depth:
                inputs = tf.concat([inputs, batch_stddev(inputs)], axis=1)
                
                # convolutional layer
                with tf.variable_scope("conv"):
                    inputs = conv_layer(depth, inputs)
                   
                # dense layer
                with tf.variable_scope("dense"):
                    # flattens the input preserving the batch axis 
                    inputs = tf.layers.flatten(inputs)
                    # get the shape
                    shape=[inputs.shape[1].value, channels(depth - 1)]
                    
                    # compute the initializer 
                    variance_scale=2.0
                    stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
                    initializer = tf.initializers.truncated_normal(0.0, 1.0)
                   
                    # get the weight from the model
                    weight = tf.get_variable(name="weight",shape=shape,initializer = initializer) * stddev
                    # multiply input x weigth
                    inputs = tf.matmul(inputs, weight)
                 
                    # get bias
                    initializer=tf.initializers.zeros()
                    bias = tf.get_variable(name="bias",shape=[inputs.shape[1].value],initializer=initializer)
                    # add bias to the input 
                    inputs = tf.nn.bias_add(inputs, bias)
                  
                    # apply a non linear function rectified linear function 
                    inputs = tf.nn.leaky_relu(inputs)
                    features = inputs
                    
                with tf.variable_scope("logits"):
                    
                    # get the shape
                    shape=[inputs.shape[1].value, labels.shape[1]]
                    # compute the initializer
                    variance_scale=1.0
                    stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
                    initializer=tf.initializers.truncated_normal(0.0, 1.0)
                    
                    # get the weight from the model 
                    weight = tf.get_variable(name="weight",shape=shape,initializer = initializer) * stddev
                    # multiply input x weigth
                    inputs = tf.matmul(inputs, weight)
                    initializer=tf.initializers.zeros()
                   
                    # get the bias 
                    bias = tf.get_variable(name="bias",shape=[inputs.shape[1].value],initializer=initializer)
                    # add bias to the input 
                    inputs = tf.nn.bias_add(inputs, bias)
                    logits = inputs
                    
                return features, logits
            else:
                
                # convolution layer
                with tf.variable_scope("conv"):
                    inputs = conv_layer(depth, inputs)
                
                # convolution downscale layer
                with tf.variable_scope("conv_downscale"):
                    
                    # get the shape
                    filters=channels(depth-1)                        
                    kernel_size=[3, 3]
                    shape=[kernel_size[0], kernel_size[1], inputs.shape[1].value, filters]
                    
                    # compute the initializer
                    variance_scale=2.0
                    stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
                    initializer = tf.initializers.truncated_normal(0.0, 1.0)
                    # get the weight from the model 
                    weight = tf.get_variable(name="weight",shape=shape, initializer=initializer) * stddev  
                    # Compute the 2-D convolution of the input
                    inputs = tf.nn.conv2d(
                        input=inputs, 
                        filter=weight, 
                        strides=[1, 1, 2, 2], 
                        padding="SAME", 
                        data_format="NCHW"
                             )
                    # get bias
                    initializer=tf.initializers.zeros()
                    bias = tf.get_variable(name="bias",shape=[inputs.shape[1].value],initializer=initializer)
                    # add the bias
                    inputs = tf.nn.bias_add(inputs, bias, data_format="NCHW")
                    # apply the non linear function to the output
                    inputs = tf.nn.leaky_relu(inputs)
                    
                return inputs

    def color_block(inputs, depth, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("color_block_{}x{}".format(*get_resolution(depth)), reuse=reuse):
            # Convolutional Layer
            with tf.variable_scope("conv"):
                # get the shape
                filters=channels(depth)                        
                kernel_size=[1, 1]
                shape=[kernel_size[0], kernel_size[1], inputs.shape[1].value, filters] 
                
                # compute the initializer
                variance_scale=2.0
                stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
                initializer=tf.initializers.truncated_normal(0.0, 1.0)

                # get the weight from the model
                weight = tf.get_variable(name="weight",shape=shape,initializer=initializer) * stddev    
                # Compute the 2-D convolution of the input
                inputs = tf.nn.conv2d(input=inputs, filter=weight, strides=[1, 1, 1, 1], padding="SAME", data_format="NCHW")
                
                # get bias
                initializer=tf.initializers.zeros()
                bias = tf.get_variable(name="bias",shape=[inputs.shape[1].value],initializer=initializer)
                # add the bias
                inputs = tf.nn.bias_add(inputs, bias, data_format="NCHW")
             
                # apply the rectified linear function to the output
                inputs = tf.nn.leaky_relu(inputs)
            
            return inputs

    def grow(images, depth):

        def high_resolution_feature_maps():
            return conv_block(grow(images, depth + 1), depth)

        def middle_resolution_feature_maps():
            return conv_block(color_block(downscale2d(
                inputs=images,
                factors=get_resolution(max_depth) // get_resolution(depth)
            ), depth), depth)

        def low_resolution_feature_maps():
            return color_block(downscale2d(
                inputs=images,
                factors=get_resolution(max_depth) // get_resolution(depth - 1)
                 ), depth - 1)

        if depth == min_depth:
            feature_maps = tf.cond(
                pred=tf.greater(growing_depth, depth),
                true_fn=high_resolution_feature_maps,
                false_fn=middle_resolution_feature_maps
            )
        elif depth == max_depth:
            feature_maps = tf.cond(
                pred=tf.greater(growing_depth, depth),
                true_fn=middle_resolution_feature_maps,
                false_fn=lambda: lerp(
                    a=low_resolution_feature_maps(),
                    b=middle_resolution_feature_maps(),
                    t=depth - growing_depth
                )
            )
        else:
            feature_maps = tf.cond(
                pred=tf.greater(growing_depth, depth),
                true_fn=high_resolution_feature_maps,
                false_fn=lambda: lerp(
                    a=low_resolution_feature_maps(),
                    b=middle_resolution_feature_maps(),
                    t=depth - growing_depth
                )
            )
        return feature_maps

    with tf.variable_scope(name, reuse=reuse):
        return grow(images, min_depth)

