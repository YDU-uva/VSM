from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
from features import extract_features_omniglot, extract_features_mini_imagenet
from inference import inference_block
from utilities import *  # sample_normal, multinoulli_log_density, print_and_log, get_log_files
from data import get_data
import os

"""
parse_command_line: command line parser
"""


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", choices=["Omniglot", "miniImageNet", 'tieredImageNet', 'cifarfs'],
                        default="miniImageNet", help="Dataset to use")
    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                        help="Whether to run traing only, testing only, or both training and testing.")
    parser.add_argument("--seed", type=int, default=42,
                        help="dataset seeds")
    parser.add_argument("--d_theta", type=int, default=256,
                        help="Size of the feature extractor output.")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Size of the random feature base.")
    parser.add_argument("--shot", type=int, default=1,
                        help="Number of training examples.")
    parser.add_argument("--way", type=int, default=5,
                        help="Number of classes.")
    parser.add_argument("--test_shot", type=int, default=None,
                        help="Shot to be used at evaluation time. If not specified 'shot' will be used.")
    parser.add_argument("--test_way", type=int, default=None,
                        help="Way to be used at evaluation time. If not specified 'way' will be used.")
    parser.add_argument("--tasks_per_batch", type=int, default=4,
                        help="Number of tasks per batch.")
    parser.add_argument("--memory_samples", type=int, default=50,
                        help="Number of samples from q.")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top attention weights.")
    parser.add_argument("--num_classes", type=int, default=64,
                        help="Number of training classes.")
    parser.add_argument("--test_iterations", type=int, default=10,
                        help="test_iterations.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.00025,
                        help="Learning rate.")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of training iterations.")
    parser.add_argument("--checkpoint_dir", "-c", default='./checkpoint',
                        help="Directory to save trained models.")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout keep probability.")
    parser.add_argument("--test_model_path", "-m", default='./checkpoint/best_validation',
                        help="Model to load and test.")
    parser.add_argument("--print_freq", type=int, default=10,
                        help="Frequency of summary results (in iterations).")
    parser.add_argument("--load_dir", "-lc", default='',
                        help="Directory to save trained models.")
    parser.add_argument("--aug", type=bool, default=False,
                        help="data augmentation")

    parser.add_argument("--ori_scale", type=float, default=0.2,
                        help="Dropout keep probability.")
    parser.add_argument("--new_scale", type=float, default=0.8,
                        help="Dropout keep probability.")
    ## hyper_params
    parser.add_argument("--beta", type=float, default=0.0001,
                        help="hyper param for kl_loss")

    args = parser.parse_args()

    # adjust test_shot and test_way if necessary
    if args.test_shot is None:
        args.test_shot = args.shot
    if args.test_way is None:
        args.test_way = args.way

    return args


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.ERROR)

    args = parse_command_line()

    logfile, checkpoint_path_validation, checkpoint_path_final = get_log_files(args.checkpoint_dir, args.mode,
                                                                               args.shot)

    # Load training and eval data
    data = get_data(args.dataset, seed=args.seed)

    # set the feature extractor based on the dataset
    if args.dataset == "miniImageNet" or args.dataset == 'tieredImageNet':
        feature_extractor_fn = extract_features_mini_imagenet
    else:
        feature_extractor_fn = extract_features_omniglot

    # evaluation samples
    eval_samples_train = 15
    eval_samples_test = args.shot

    # testing parameters
    test_iterations = args.test_iterations
    test_args_per_batch = 1  # always use a batch size of 1 for testing

    # tf placeholders
    train_images = tf.placeholder(tf.float32, [None,  # tasks per batch
                                               None,  # shot
                                               data.get_image_height(),
                                               data.get_image_width(),
                                               data.get_image_channels()],
                                  name='train_images')
    test_images = tf.placeholder(tf.float32, [None,  # tasks per batch
                                              None,  # num test images
                                              data.get_image_height(),
                                              data.get_image_width(),
                                              data.get_image_channels()],
                                 name='test_images')
    train_labels = tf.placeholder(tf.float32, [None,  # tasks per batch
                                               None,  # shot
                                               args.way],
                                  name='train_labels')
    test_labels = tf.placeholder(tf.float32, [None,  # tasks per batch
                                              None,  # num test images
                                              args.way],
                                 name='test_labels')
    memory_keys = tf.placeholder(tf.float32, [args.num_classes,  # num memory images
                                              args.d_theta],
                                 name='memory_keys')
    memory_values_mean = tf.placeholder(tf.float32, [args.num_classes,  # num memory images
                                                     args.d_theta],
                                        name='memory_values_mean')
    memory_values_var = tf.placeholder(tf.float32, [args.num_classes,  # num memory images
                                                    args.d_theta],
                                       name='memory_values_mean')
    ori_train_labels = tf.placeholder(tf.float32, [None,  # tasks per batch
                                                   args.way * args.shot],
                                      name='ori_train_labels')
    dropout_keep_prob = tf.placeholder(tf.float32, [], name='dropout_keep_prob')
    # L = tf.constant(args.samples, dtype=tf.float32, name="num_samples")

    L = tf.constant(args.num_samples, dtype=tf.float32, name="num_samples")

    def compute_base_distri(inputs):
        train_inputs, test_inputs, train_outputs, test_outputs, train_ori_labels = inputs

        with tf.variable_scope('shared_features'):

            features_train = feature_extractor_fn(images=train_inputs,
                                                  output_size=args.d_theta,
                                                  use_batch_norm=True,
                                                  dropout_keep_prob=dropout_keep_prob)
            features_test = feature_extractor_fn(images=test_inputs,
                                                 output_size=args.d_theta,
                                                 use_batch_norm=True,
                                                 dropout_keep_prob=dropout_keep_prob)

            support_mean_list = []
            support_var_list = []
            query_mean_list = []
            query_var_list = []
            kl_loss_list = []

            for c in range(args.way):
                support_class_mask = tf.equal(tf.argmax(train_outputs, 1), c)
                support_class_features = tf.boolean_mask(features_train, support_class_mask)
                support_class_labels = tf.boolean_mask(train_ori_labels, support_class_mask)
                support_class_labels_values = tf.to_int32(support_class_labels[0])
                # Pool across dimensions
                support_nu = tf.expand_dims(tf.reduce_mean(support_class_features, axis=0), axis=0)
                memory_keys_choose = tf.concat(
                    (memory_keys[:support_class_labels_values], memory_keys[support_class_labels_values:]), axis=0)
                dotp = tf.matmul(support_nu, memory_keys_choose, transpose_b=True)
                attention_weights = tf.nn.softmax(dotp)
                top_attention_weights = tf.nn.top_k(attention_weights, args.top_k).values
                top_attention_weights = tf.nn.softmax(top_attention_weights)
                top_attention_weights_index = tf.nn.top_k(attention_weights, args.top_k).indices
                list_indices = [[top_attention_weights_index[0][i]] for i in range(args.top_k)]
                memory_values_mean_choose = tf.gather_nd(memory_values_mean, list_indices)
                memory_values_var_choose = tf.gather_nd(memory_values_var, list_indices)
                memory_mean = tf.matmul(top_attention_weights, memory_values_mean_choose)
                memory_var = tf.matmul(top_attention_weights, memory_values_var_choose)
                memory_z = sample_normal(memory_mean, memory_var, args.memory_samples)
                memory_z = tf.reshape(memory_z, (args.memory_samples, args.d_theta))
                support_nu_tile = tf.tile(support_nu, (args.memory_samples, 1))
                memory_support = tf.concat([support_nu_tile, memory_z], axis=-1)
                with tf.variable_scope('z', reuse=tf.AUTO_REUSE):
                    support_mu = inference_block(memory_support, args.d_theta, args.d_theta, 'support_mu')
                    support_logvar = inference_block(memory_support, args.d_theta, args.d_theta, 'support_logvar')

                support_mu_ave = tf.expand_dims(tf.reduce_mean(support_mu, axis=0), axis=0)
                support_logvar_ave = tf.expand_dims(tf.reduce_mean(support_logvar, axis=0), axis=0)
                support_mean_list.append(support_mu_ave)
                support_var_list.append(support_logvar_ave)

                query_class_mask = tf.equal(tf.argmax(test_outputs, 1), c)
                query_class_features = tf.boolean_mask(features_test, query_class_mask)

                # Pool across dimensions
                # query_nu = tf.expand_dims(tf.reduce_mean(query_class_features, axis=0), axis=0)
                query_mu = inference_block(query_class_features, args.d_theta, args.d_theta, 'query_mu')
                query_logvar = inference_block(query_class_features, args.d_theta, args.d_theta, 'query_logvar')
                query_mean_list.append(query_mu)
                query_var_list.append(query_logvar)

                # kl loss
                kl_loss = KL_divergence(support_mu_ave, support_logvar_ave, query_mu, query_logvar)

                kl_loss_list.append(kl_loss)
            support_mean_features = tf.concat(support_mean_list, axis=0)
            support_logvar_features = tf.concat(support_var_list, axis=0)

            kl_loss_all = tf.reduce_sum(tf.concat(kl_loss_list, axis=0))
            z_prototypes = sample_normal(support_mean_features, support_logvar_features, args.num_samples)

            tile_query_label = tf.tile(tf.expand_dims(test_outputs, axis=0), [args.num_samples, 1, 1])
            tile_query_features = tf.tile(tf.expand_dims(features_test, axis=0), [args.num_samples, 1, 1])
            dists = calc_euclidian_dists(tile_query_features, z_prototypes)

            # log softmax of calculated distances
            log_p_y = tf.nn.log_softmax(-dists, axis=-1)
            # n_query = tf.shape(test_outputs)[0]
            # log_p_y = tf.reshape(log_p_y, [args.num_samples, args.way, n_query, -1])

            task_log_py = multinoulli_log_density(inputs=tile_query_label, logits=log_p_y)
            averaged_predictions = tf.reduce_logsumexp(log_p_y, axis=0) - tf.log(L)
            task_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_outputs, axis=-1),
                                                            tf.argmax(averaged_predictions, axis=-1)), tf.float32))
            task_score = tf.reduce_logsumexp(task_log_py, axis=0) - tf.log(L)
            task_loss = -tf.reduce_mean(task_score, axis=0)
            for c in range(args.way):
                support_class_mask = tf.equal(tf.argmax(train_outputs, 1), c)
                support_class_features = tf.boolean_mask(features_train, support_class_mask)

                query_class_mask = tf.equal(tf.argmax(test_outputs, 1), c)
                query_class_features = tf.boolean_mask(features_test, query_class_mask)

                support_class_labels = tf.boolean_mask(train_ori_labels, support_class_mask)
                support_class_labels_values = tf.to_int32(support_class_labels[0])
                class_features = tf.concat([support_class_features, query_class_features], axis=0)
                class_features_mean = tf.tile(tf.expand_dims(tf.reduce_mean(class_features, axis=0), axis=0),
                                              [args.memory_samples, 1])
                memory_keys_choose = tf.expand_dims(memory_keys[support_class_labels_values], axis=0)
                # memory_means_choose = tf.expand_dims(memory_values_mean[support_class_labels_values], axis=0)
                # memory_vars_choose = tf.expand_dims(memory_values_var[support_class_labels_values], axis=0)
                memory_key_representation = graph_attention(class_features, memory_keys_choose,
                                                            memory_keys_choose)
                memory_new_keys = tf.expand_dims(tf.reduce_mean(memory_key_representation, axis=0), axis=0)

                memory_new_means = inference_block(memory_keys_choose, args.d_theta, args.d_theta, 'memory_new_means')
                memory_new_vars = inference_block(memory_keys_choose, args.d_theta, args.d_theta, 'memory_new_vars')
                memory_new_z = sample_normal(memory_new_means, memory_new_vars, args.memory_samples)
                memory_new_z = tf.reshape(memory_new_z, shape=(args.memory_samples, args.d_theta))
                class_features_memory = tf.concat([class_features_mean, memory_new_z], axis=-1)
                with tf.variable_scope('z', reuse=tf.AUTO_REUSE):
                    class_memory_mu = inference_block(class_features_memory, args.d_theta, args.d_theta, 'support_mu')
                    class_memory_logvar = inference_block(class_features_memory, args.d_theta, args.d_theta,
                                                          'support_logvar')

                class_mu_ave = tf.expand_dims(tf.reduce_mean(class_memory_mu, axis=0), axis=0)
                class_logvar_ave = tf.expand_dims(tf.reduce_mean(class_memory_logvar, axis=0), axis=0)

                key_part_1 = memory_keys[:support_class_labels_values]
                key_part_2 = memory_keys[tf.add(support_class_labels_values, 1):]

                memory_concat_keys = tf.concat(
                    [tf.multiply(args.ori_scale, key_part_1), tf.multiply(args.ori_scale, key_part_2),
                     tf.multiply(1 - args.ori_scale, memory_new_keys)], axis=0)

                mean_part_1 = memory_values_mean[:support_class_labels_values]
                mean_part_2 = memory_values_mean[tf.add(support_class_labels_values, 1):]
                memory_concat_values_mean = tf.concat(
                    [tf.multiply(args.ori_scale, mean_part_1), tf.multiply(args.ori_scale, mean_part_2),
                     tf.multiply(1 - args.ori_scale, class_mu_ave)],
                    axis=0)

                var_part_1 = memory_values_var[:support_class_labels_values]
                var_part_2 = memory_values_var[tf.add(support_class_labels_values, 1):]
                memory_concat_values_var = tf.concat(
                    [tf.multiply(args.ori_scale, var_part_1), tf.multiply(args.ori_scale, var_part_2),
                     tf.multiply(1 - args.ori_scale, class_logvar_ave)], axis=0)

        return [kl_loss_all, task_loss, task_accuracy, memory_concat_keys, memory_concat_values_mean,
                memory_concat_values_var]

    batch_base_d_output = tf.map_fn(fn=compute_base_distri,
                                    elems=(train_images, test_images, train_labels, test_labels, ori_train_labels),
                                    dtype=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                                    parallel_iterations=args.tasks_per_batch)

    batch_kl, batch_task_loss, batch_accuracy, batch_memory_keys, batch_memory_mean, batch_memory_var = batch_base_d_output

    batch_loss = batch_task_loss + args.beta * batch_kl

    loss = tf.reduce_mean(batch_loss)
    loss_ce = tf.reduce_mean(batch_task_loss)
    loss_kl = tf.reduce_mean(batch_kl)
    accuracy = tf.reduce_mean(batch_accuracy)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print_and_log(logfile, "Options: %s\n" % args)
        saver = tf.train.Saver()

        if args.mode == 'train' or args.mode == 'train_test':

            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            train_step = optimizer.minimize(loss)  # , var_list=opt_list)

            validation_batches = 1000
            iteration = 0
            best_iteration = 0
            best_validation_accuracy = 0.0
            train_iteration_accuracy = []
            sess.run(tf.global_variables_initializer())
            if args.load_dir:
                saver.restore(sess, save_path=args.load_dir)
            # Main training loop
            memory_key = np.zeros([args.num_classes, args.d_theta])
            memory_value_mean = np.zeros([args.num_classes, args.d_theta])
            memory_value_var = np.zeros([args.num_classes, args.d_theta])
            while iteration < args.iterations:
                train_inputs, test_inputs, train_outputs, test_outputs, ori_train_outputs, ori_test_outputs, exist_classes = \
                    data.get_batch('train', args.tasks_per_batch, args.shot, args.way, eval_samples_train)

                feed_dict = {train_images: train_inputs, test_images: test_inputs,
                             train_labels: train_outputs, test_labels: test_outputs,
                             ori_train_labels: ori_train_outputs,
                             dropout_keep_prob: args.dropout, memory_keys: memory_key,
                             memory_values_mean: memory_value_mean, memory_values_var: memory_value_var}
                _, iteration_loss, iteration_loss_ce, iteration_loss_kl, iteration_accuracy, m_k, m_m, m_v = sess.run(
                    [train_step,
                     loss, loss_ce,
                     loss_kl,
                     accuracy,
                     batch_memory_keys,
                     batch_memory_mean,
                     batch_memory_var],
                    feed_dict)

                train_iteration_accuracy.append(iteration_accuracy)
                memory_key = add_memory(ori_train_outputs, m_k, memory_key)
                memory_value_mean = add_memory(ori_train_outputs, m_m, memory_value_mean)
                memory_value_var = add_memory(ori_train_outputs, m_v, memory_value_var)

                if (iteration > 0) and (iteration % args.print_freq == 0):
                    # compute accuracy on validation set
                    validation_iteration_accuracy = []
                    validation_iteration = 0
                    while validation_iteration < validation_batches:
                        train_inputs, test_inputs, train_outputs, test_outputs, ori_train_outputs, ori_test_outputs, exist_classes = \
                            data.get_batch('validation', test_args_per_batch, args.shot, args.way, eval_samples_test)
                        feed_dict = {train_images: train_inputs, test_images: test_inputs,
                                     train_labels: train_outputs, test_labels: test_outputs,
                                     ori_train_labels: ori_train_outputs,
                                     dropout_keep_prob: 1.0, memory_keys: memory_key,
                                     memory_values_mean: memory_value_mean, memory_values_var: memory_value_var}
                        iteration_accuracy = sess.run(accuracy, feed_dict)
                        validation_iteration_accuracy.append(iteration_accuracy)
                        validation_iteration += 1
                    validation_accuracy = np.array(validation_iteration_accuracy).mean()
                    train_accuracy = np.array(train_iteration_accuracy).mean()

                    # save checkpoint if validation is the best so far
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        saver.save(sess=sess, save_path=checkpoint_path_validation)
                        np.save(checkpoint_path_validation + "_key", memory_key)
                        np.save(checkpoint_path_validation + "_mu", memory_value_mean)
                        np.save(checkpoint_path_validation + "_var", memory_value_var)
                    print_and_log(logfile,
                                  'Iteration: {}, Loss: {:5.3f}, Loss_ce: {:5.3f}, Loss_kl: {:5.3f}, '
                                  'Train-Acc: {:5.3f}, Val-Acc: {:5.3f}, best_iter: {}, Best-Acc: {:5.3f}'
                                  .format(iteration, iteration_loss, iteration_loss_ce,
                                          iteration_loss_kl, train_accuracy, validation_accuracy, best_iteration,
                                          best_validation_accuracy))

                    train_iteration_accuracy = []

                iteration += 1
            # save the checkpoint from the final epoch
            saver.save(sess, save_path=checkpoint_path_final)
            print_and_log(logfile, 'Fully-trained model saved to: {}'.format(checkpoint_path_final))
            print_and_log(logfile, 'Best validation accuracy: {:5.3f}'.format(best_validation_accuracy))
            print_and_log(logfile, 'Best validation model saved to: {}'.format(checkpoint_path_validation))
            np.save(checkpoint_path_final + "_key", memory_key)
            np.save(checkpoint_path_final + "_mu", memory_value_mean)
            np.save(checkpoint_path_final + "_var", memory_value_var)

        def test_model(model_path, load=True):

            memory_key = np.load(model_path + '_key.npy')
            memory_value_mean = np.load(model_path + '_mu.npy')
            memory_value_var = np.load(model_path + '_var.npy')
            if load:
                saver.restore(sess, save_path=model_path)

            test_iteration = 0
            test_iteration_accuracy = []
            while test_iteration < test_iterations:
                train_inputs, test_inputs, train_outputs, test_outputs, ori_train_outputs, ori_test_outputs, exist_classes = \
                    data.get_batch('test', test_args_per_batch, args.test_shot, args.test_way,
                                   eval_samples_test)
                feedDict = {train_images: train_inputs, test_images: test_inputs,
                            train_labels: train_outputs, test_labels: test_outputs,
                            ori_train_labels: ori_train_outputs,
                            dropout_keep_prob: 1.0, memory_keys: memory_key,
                            memory_values_mean: memory_value_mean, memory_values_var: memory_value_var}
                iter_acc = sess.run(accuracy, feedDict)
                test_iteration_accuracy.append(iter_acc)
                test_iteration += 1
            test_accuracy = np.array(test_iteration_accuracy).mean() * 100.0
            confidence_interval_95 = \
                (196.0 * np.array(test_iteration_accuracy).std()) / np.sqrt(len(test_iteration_accuracy))
            print_and_log(logfile, 'Held out accuracy: {0:5.3f} +/- {1:5.3f} on {2:}'
                          .format(test_accuracy, confidence_interval_95, model_path))

        if args.mode == 'train_test':
            print_and_log(logfile, 'Train Shot: {0:d}, Train Way: {1:d}, Test Shot {2:d}, Test Way {3:d}'
                          .format(args.shot, args.way, args.test_shot, args.test_way))
            # test the model on the final trained model
            # no need to load the model, it was just trained
            # tf.reset_default_graph()
            test_model(checkpoint_path_final, load=False)

            # test the model on the best validation checkpoint so far
            test_model(checkpoint_path_validation, )

        if args.mode == 'test':
            test_model(args.test_model_path)

    logfile.close()


if __name__ == "__main__":
    tf.app.run()
