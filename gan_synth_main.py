
import tensorflow as tf
import numpy as np
import functools
import argparse
import glob
import json
import os
from scipy.io import wavfile
from dataset import nsynth_input_fn
from models import GANSynth
from networks import generator, discriminator
from utils import Dict

from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="gan_synth_model_prova/")
parser.add_argument('--filenames', type=str, default="/content/drive/My Drive/Colab Notebooks/GANSynth/nsynth_test.tfrecord")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=None)
parser.add_argument("--total_steps", type=int, default=1000000)
parser.add_argument("--growing_steps", type=int, default=1000000)
parser.add_argument('--train', default=True, action="store_true")
parser.add_argument('--generate', default= False, action="store_true")
args = parser.parse_args()


tf.logging.set_verbosity(tf.logging.INFO)


with tf.Graph().as_default():

    tf.set_random_seed(0)
    growing_level=tf.cast(tf.divide(x=tf.train.create_global_step(),y=args.growing_steps), tf.float32)
    
    gan_synth = GANSynth(
        args = args,
        generator=generator,
        discriminator=discriminator,
        growing_level = growing_level
    )

    if args.train:

        gan_synth.train(
            model_dir=args.model_dir,
            total_steps=args.total_steps,
            save_checkpoint_steps=1000,
            save_summary_steps=100,
            log_tensor_steps=100
        )

    if args.generate:

        os.makedirs("samples_8_new", exist_ok=True)

        generator = gan_synth.generate(
            model_dir=args.model_dir
        )

        num_waveforms = 0
        for waveforms in generator:
            for waveform in waveforms:
                wavfile.write(f"samples_8_new/{num_waveforms}.wav", rate=16000, data=waveform)
                num_waveforms += 1

        print(f"{num_waveforms} waveforms are generated in `samples` directory")
        
        
        
        
        
