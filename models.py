import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import glob
from spectral_ops import *
import functools
import os
from dataset import nsynth_input_fn
from utils import Dict


class GANSynth(object):

    def __init__(self, args, generator, discriminator, growing_level):
        
        '''
        HYPERPARAMETERS
        '''
        # generator and discriminator learning rate
        generator_lr = 8e-4
        discriminator_lr = 8e-4 
        mode_seeking_loss_weight=0.1
        real_gradient_penalty_weight=5.0
        fake_gradient_penalty_weight=0.0
        
        # parameters to extract spectrogram
        spectral_params = Dict(
        waveform_length=64000,
        sample_rate=16000,
        spectrogram_shape=[128, 1024],
        overlap=0.75
        )
        
        self.config = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=False,
            gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=1.0)
        )
        
        # load the dataset
        real_waveforms, labels = nsynth_input_fn(
            filenames=glob.glob(args.filenames),
            batch_size=args.batch_size,
            num_epochs=args.num_epochs if args.train else 1,
            shuffle=False if args.train else False,
            pitches=range(24, 85),
            sources=[0]
        )
        # A latent vector is generatrd from a normal distribution
        latents = tf.random.normal([args.batch_size, 256])

        # the latent vector is passed to the generator with labels and the growing_level variable
        # to generate a fake image  
        fake_images = generator(latents, labels, growing_level)

        # extract from waveform loaded from the dataset extract magnitude spectrograms and instantaneous frequencies 
        real_magnitude_spectrograms, real_instantaneous_frequencies = convert_to_spectrogram(
            real_waveforms, **spectral_params)
        
        # reconstract the real image
        real_images = tf.stack([real_magnitude_spectrograms, real_instantaneous_frequencies], axis=1)

        # get fake magnitude spectrogram and instantaneous frequencies from fake_images
        fake_magnitude_spectrograms, fake_instantaneous_frequencies = tf.unstack(fake_images, axis=1)
        # from fake magnitude spectrogram and fake insatntaneous frequencies reconstructa a fake waveform
        fake_waveforms = convert_to_waveform(fake_magnitude_spectrograms, fake_instantaneous_frequencies, **spectral_params)

        # get, real and the fake, feature and logits
        real_features, real_logits = discriminator(real_images, labels, growing_level)
        fake_features, fake_logits = discriminator(fake_images, labels, growing_level)

        # label conditioning 
        # Both the generator and discriminator are conditioned on the labels of the input data
        '''
        Unlike Progressive GAN, our method involves conditioning on an additional source of information. Specifically, we append a one-hot representation of musical pitch to the latent vector, with the
        musically-desirable goal of achieving independent control of pitch and timbre. To encourage the
        generator to use the pitch information, we also add an auxiliary classification (Odena et al., 2017)
        loss to the discriminator that tries to predict the pitch label.
        '''
        real_logits = tf.gather_nd(real_logits, indices=tf.where(labels))
        fake_logits = tf.gather_nd(fake_logits, indices=tf.where(labels))

        # non-saturating loss
        discriminator_losses = tf.nn.softplus(-real_logits)
        discriminator_losses += tf.nn.softplus(fake_logits)
        
        # zero-centerd gradient penalty on data distribution
        if real_gradient_penalty_weight:
            real_gradients = tf.gradients(real_logits, [real_images])[0]
            real_gradient_penalties = tf.reduce_sum(tf.square(real_gradients), axis=[1, 2, 3])
            discriminator_losses += real_gradient_penalties * real_gradient_penalty_weight
        # zero-centerd gradient penalty on generator distribution
        if fake_gradient_penalty_weight:
            fake_gradients = tf.gradients(fake_logits, [fake_images])[0]
            fake_gradient_penalties = tf.reduce_sum(tf.square(fake_gradients), axis=[1, 2, 3])
            discriminator_losses += fake_gradient_penalties * fake_gradient_penalty_weight

        # non-saturating loss
        generator_losses = tf.nn.softplus(-fake_logits)
        # gradient-based mode-seeking loss
        if mode_seeking_loss_weight:
            latent_gradients = tf.gradients(fake_images, [latents])[0]
            mode_seeking_losses = 1.0 / (tf.reduce_sum(tf.square(latent_gradients), axis=[1]) + 1.0e-6)
            generator_losses += mode_seeking_losses * mode_seeking_loss_weight

        generator_loss = tf.reduce_mean(generator_losses)
        discriminator_loss = tf.reduce_mean(discriminator_losses)
        
        # define the generator optimizer, in particular here is used Adam Optimizer
        generator_optimizer = tf.train.AdamOptimizer( learning_rate = generator_lr)
        # define the discirminator optimizer, in particular here is used Adam Optimizer
        discriminator_optimizer = tf.train.AdamOptimizer( learning_rate = discriminator_lr )

        generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

        generator_train_op = generator_optimizer.minimize(
            loss=generator_loss,
            var_list=generator_variables,
            global_step=tf.train.get_or_create_global_step()
        )
        discriminator_train_op = discriminator_optimizer.minimize(
            loss=discriminator_loss,
            var_list=discriminator_variables
        )
        
        self.real_waveforms = real_waveforms
        self.fake_waveforms = fake_waveforms
        self.real_magnitude_spectrograms = real_magnitude_spectrograms
        self.fake_magnitude_spectrograms = fake_magnitude_spectrograms
        self.real_instantaneous_frequencies = real_instantaneous_frequencies
        self.fake_instantaneous_frequencies = fake_instantaneous_frequencies
        self.real_images = real_images
        self.fake_images = fake_images
        self.real_labels = labels
        self.fake_labels = labels
        self.real_features = real_features
        self.fake_features = fake_features
        self.real_logits = real_logits
        self.fake_logits = fake_logits
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.generator_train_op = generator_train_op
        self.discriminator_train_op = discriminator_train_op
    
    """ Add summary to TensorBoard """
    #This function allow us to add a summary this tank tf.summary that is un API for writing summary 
    # that can be visualized on tensoboard
    def add_summary(writer, name, value, global_step):
    
        # define a summary
        summary = tf.Summary()
        # add a value to the summary 
        summary_value = summary.value.add()
        summary_value.simple_value = value
        # add the tag name to the value
        summary_value.tag = name
        # write the value on the summary
        writer.add_summary(summary, global_step=global_step)
        # flush the writer
        writer.flush()
        
    # This is the function of that class, it handle all the training and save the vary informaion that we need 
    def train(self, model_dir,total_steps, save_checkpoint_steps, save_summary_steps, log_tensor_steps):
        
        # Define a Session named sess
        with tf.Session(config=self.config) as sess:
            
            # Initialize the vary variable(global/local/tables) in our graph
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            
            # Define a sever it allow operation as save and restore variables to and from checkpoints. 
            # Define the maximum number of checkpoints to keep (in this case the last 10 checkpoints)
            # Define laso for how many hours of traing keep one checkpoint (in this case 12 hours)
            saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=12)
            
            init = 0
            # Check if a checkpoint exist in directory specified in "model_dir"
            # If it exist continue the traing from that point
            # This is very usefull, because if the training crash or the connection drop we dont loss 
            # all the progress but we can continue the training from the last checkpoint
            if os.path.isdir(model_dir):

                checkpoint = tf.train.latest_checkpoint(model_dir)
                print("CHECKPOINT", checkpoint, type(checkpoint))
                saver.restore(sess, checkpoint)
                init = int(checkpoint.split('-')[-1])
            
            # Save the graph of our model 
            writer = tf.summary.FileWriter(model_dir, sess.graph)
            
            # for each step between init(couls be 0 or the last step of previous checkpoint) and total_steps, do:
            for step in range(init,total_steps+1):
                    
                # Run generator and discriminator
                sess.run(self.discriminator_train_op)
                sess.run(self.generator_train_op)
                
                # At each 100 steps (save_summary_step=100) write on the summary
                # Loss of generetor and discriminator / Real and Fake wavefor
                # In this way we can check on tensoborad the progress of the model during the training
                
                if (step % save_summary_steps) == 0:
                    
                    """
                    Save losses to Tensorboard after 100 steps
                    """
                    # get the generetor and disciriminator loss
                    g_loss=sess.run(self.generator_loss)
                    d_loss=sess.run(self.discriminator_loss)
               
                    # add this information to the summary(on tensorboard)
                    GANSynth.add_summary(writer, 'discriminator_loss', d_loss, step)
                    GANSynth.add_summary(writer, 'generator_loss', g_loss, step)
                    
                    print("STEP--->", step, "  DISCRIMINATOR LOSS = ",d_loss ,"  GENERATOR LOSS = ", g_loss)
                    
                    """
                    Save fake and real audio to Tensorboard after 100 steps
                    """
                    # get real and fake waveform
                    real_waveforms=sess.run(self.real_waveforms)
                    fake_waveforms=sess.run(self.fake_waveforms)
                
                    real = sess.run(tf.summary.audio('real_waveform', real_waveforms,  sample_rate=16000, max_outputs=4))
                    fake = sess.run(tf.summary.audio('fake_waveform', fake_waveforms,  sample_rate=16000, max_outputs=4))
                    
                    # add this information to the summary
                    writer.add_summary(real)
                    writer.add_summary(fake)
                  
                # at each 1000 steps (save_checpoin_steps==1000) save the checkpoint of the model
                
                if (step % save_checkpoint_steps) == 0:
                    """Save the model"""
                    save_path = saver.save(sess, model_dir+'model.ckpt', global_step=step)

                    
    def generate(self, model_dir):

       # define a session
       with tf.Session(config=self.config) as sess:
            
            # Initialize the vary variable(global/local/tables) in our graph
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            
            # define a Saver
            saver = tf.train.Saver()
            
            # check if the directory that contains the saved model exist
            if os.path.isdir(model_dir):
    
                # take the name of the latest checkpoint
                checkpoint = tf.train.latest_checkpoint(model_dir)
                print("CHECKPOINT", checkpoint, type(checkpoint))
                # restor the checkpoint
                saver.restore(sess, checkpoint)

                try:
                    # generate fake audio
                    yield sess.run(self.fake_waveforms)
                except:
                    print("Error during the generation of fake sounds")
                 
                
                
                
