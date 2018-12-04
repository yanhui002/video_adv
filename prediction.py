import argparse
import os
import sys
import shutil
import models
import numpy as np
import tensorflow as tf
import scipy.misc
from our_data import DataSet

def calc_gradients(
        test_file,
        model_name,
        input_file_dir,
        output_file_dir,
        max_iter,
        save_freq,
        learning_rate=0.0001,
        targets=None,
        weight_loss2=1,
        data_spec=None,
        batch_size=1,
        seq_len=40,
        noise_file=None):
    """Compute the gradients for the given network and images."""    
    spec = data_spec

    input_image = tf.placeholder(tf.float32, (batch_size, seq_len, spec.crop_size, spec.crop_size, spec.channels))
    input_label = tf.placeholder(tf.int32, (batch_size))
   
    sess = tf.Session()
    probs, variable_set, pre_label,ince_output,pre_node = models.get_model(sess, input_image, model_name) 
    true_label_prob = tf.reduce_sum(probs*tf.one_hot(input_label,101),[1])

    data = DataSet(test_list=test_file, seq_length=seq_len,image_shape=(spec.crop_size, spec.crop_size, spec.channels))
    all_names = []
    all_images = []
    all_labels = []
    print(data.test_data)
    for video in data.test_data:
        frames = data.get_frames_for_sample(video)
        if len(frames) < seq_len:
           continue
        frames = data.rescale_list(frames, seq_len)
        frames_data = data.build_image_sequence(frames)
        all_images.append(frames_data)
        label, hot_labels = data.get_class_one_hot(video[1])
        all_labels.append(label)
        all_names.append(frames)
    total = len(all_names)
    all_indices = range(total)
    num_batch = total/batch_size
    print('process data length:', num_batch)
    
    correct_ori = 0
    
    for ii in range(num_batch):        
        images = all_images[ii*batch_size : (ii+1)*batch_size]
        names = all_names[ii*batch_size : (ii+1)*batch_size]
        labels = all_labels[ii*batch_size : (ii+1)*batch_size]
        indices = all_indices[ii*batch_size : (ii+1)*batch_size]
        for xx in range(len(indices)):
            print(names[xx][0],'label:', labels[xx], 'indice:',indices[xx], 'size:', len(images[xx]), len(images[xx][0]), len(images[xx][0][0]), len(images[xx][0][0][0]))
        #sess.run(tf.initialize_variables(init_varibale_list))
        
        feed_dict = {input_image: images, input_label: labels}
        true_prob, var_pre, var_node = sess.run((true_label_prob, pre_label, pre_node), feed_dict=feed_dict)
        for xx in range(len(indices)):
           if labels[xx] == var_pre[xx]:
              correct_ori += 1
        print('node:', var_node, 'prediction:', var_pre, 'probib', true_prob)
        print('correct_ori:' ,correct_ori)
        print('---------------------------------------------------')


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Use Adam optimizer to generate adversarial examples.')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Directory of dataset.')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory of output image file.')
    parser.add_argument('--model', type=str, required=True,choices=['GoogleNet','Inception2'],
                        help='Models to be evaluated.')
    parser.add_argument('--num_images', type=int, default=sys.maxsize,
                        help='Max number of images to be evaluated.')
    parser.add_argument('--file_list', type=str, default=None,
                        help='Evaluate a specific list of file in dataset.')
    parser.add_argument('--noise_file', type=str, default=None,
                        help='Directory of the noise file.')
    parser.add_argument('--num_iter', type=int, default=300,
                        help='Number of iterations to generate attack.')
    parser.add_argument('--save_freq', type=int, default=300,
                        help='Save .npy file when each save_freq iterations.')
    parser.add_argument('--learning_rate', type=float, default=0.001 * 255,
                        help='Learning rate of each iteration.')
    parser.add_argument('--target', type=str, default=None,
                        help='Target list of dataset.')
    parser.add_argument('--weight_loss2', type=float, default=1.0,
                        help='Weight of distance penalty.')
    parser.add_argument('--not_crop', dest='use_crop', action='store_false',
                        help='Not use crop in image producer.')

    parser.set_defaults(use_crop=True)
    args = parser.parse_args()
    print(args.file_list)
    assert args.num_iter % args.save_freq == 0

    data_spec = models.get_data_spec(model_name=args.model)
    args.learning_rate = args.learning_rate / 255.0 * (data_spec.rescale[1] - data_spec.rescale[0])
    seq_len = 40
    batch_size = 1
    targets = None
    if args.target is not None:
        targets = {}
        with open(args.target, 'r') as f:
            for line in f:
                key, value = line.strip().split()
                targets[key] = int(value)
                
    gradients = calc_gradients(
        args.file_list,
        args.model,
        args.input_dir,
        args.output_dir,
        args.num_iter,
        args.save_freq,
        args.learning_rate,
        targets,
        args.weight_loss2,
        data_spec,
        batch_size,
        seq_len,
        args.noise_file)


if __name__ == '__main__':
    main()
