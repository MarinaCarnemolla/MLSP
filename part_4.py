  
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
parser.add_argument("--model_dir", type=str, default="gan_synth_model/")
parser.add_argument('--filenames', type=str, default="./nsynth_test.tfrecord")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=None)
parser.add_argument("--total_steps", type=int, default=1000000)
parser.add_argument("--growing_steps", type=int, default=1000000)
parser.add_argument('--train', default=False, action="store_true")
parser.add_argument('--generate', default=True, action="store_true")
args = parser.parse_args()


with tf.Graph().as_default():

    tf.set_random_seed(0)
    growing_level=tf.cast(tf.divide(x=tf.train.create_global_step(),y=args.growing_steps), tf.float32)
    
    gan_synth = GANSynth(
        args = args,
        generator=generator,
        discriminator=discriminator,
        growing_level = growing_level
    )

    os.makedirs("results", exist_ok=True)
    
    generator = gan_synth.generate(
        model_dir=args.model_dir
    )
    
    num_waveforms = 0
    for waveforms in generator:
        print("Arriva ------>", waveforms)
        for waveform in waveforms:
            wavfile.write(f"results/{num_waveforms}.wav", rate=16000, data=waveform)
            num_waveforms += 1
    
    print(f"{num_waveforms} waveforms are generated in `results` directory")
