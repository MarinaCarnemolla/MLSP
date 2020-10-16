import tensorflow as tf
import numpy as np
import functools
import pathlib
import json
import os
from utils import Dict
from tensorflow.contrib.framework.python.ops import audio_ops

def nsynth_input_fn(filenames, batch_size, num_epochs, shuffle,
                    buffer_size=None, pitches=None, sources=None):

    #here we create a lookup index table 
    index_table = tf.contrib.lookup.index_table_from_tensor(sorted(pitches), dtype=tf.int32)

    def parse_example(example):

        features = Dict(tf.parse_single_example(
            serialized=example,
            features=dict(
                path=tf.FixedLenFeature([], dtype=tf.string),
                pitch=tf.FixedLenFeature([], dtype=tf.int64),
                source=tf.FixedLenFeature([], dtype=tf.int64)
            )
        ))

        #Read the element
        waveform = tf.read_file(features.path)
        
        #Here we decode the audio a 16 bit obtaining 64000 dimension 
        # will be scaled to -1.0 to 1.0 in float.
        waveform, _ = audio_ops.decode_wav(
            contents=waveform,
            desired_channels=1,
            desired_samples=64000
        )
        waveform = tf.squeeze(waveform)

        #to each element assign a label according its pitch
        label = index_table.lookup(features.pitch)
        label = tf.one_hot(label, len(pitches))

        #cast pitch and source to int-32
        pitch = tf.cast(features.pitch, tf.int32)
        source = tf.cast(features.source, tf.int32)

        return waveform, label, pitch, source
    
    #Dataset of the Dataset Framework
    dataset = tf.data.TFRecordDataset( filenames=filenames )
    
    #Random shuffle the element of the dataset
    #Fill a buffer with buffer_size elements, then samples element randomly of this buffer
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size = buffer_size or sum([
                len(list(tf.io.tf_record_iterator(filename)))
                for filename in filenames
            ]),
            reshuffle_each_iteration=True
        )
    #Repeat the dataser for n = number of epochs 
    dataset = dataset.repeat( count=num_epochs )
    #Parallelization using multiple threads, appling the function parse_example
    dataset = dataset.map( map_func=parse_example, num_parallel_calls=os.cpu_count())
    
    # filter just acoustic instruments and just pitches between 24-84 (as in the paper)
    dataset = dataset.filter(
        predicate=lambda waveform, label, pitch, source: functools.reduce(
            tf.logical_and, filter(lambda x: x is not None, [
                tf.greater_equal(pitch, min(pitches)) if pitches else pitches,
                tf.less_equal(pitch, max(pitches)) if pitches else pitches,
                tf.reduce_any(tf.equal(sources, source)) if sources else sources,
            ])
        )
    )
    
    dataset = dataset.map(
        map_func=lambda waveform, label, pitch, source: (waveform, label),
        num_parallel_calls=os.cpu_count()
    )
    #Combine the elements of the dataset into batches and drop the remaining elements
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    # Create a dataset that prefetch the element of this dataset. This improve the latency and throughput     
    dataset = dataset.prefetch( buffer_size=1 )

    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
   
    return iterator.get_next()
