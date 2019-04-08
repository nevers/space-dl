#!/usr/bin/env python3

import re
import os
import csv
import util
import math

import multiprocessing as mp
import tensorflow as tf
import numpy as np

from PIL import Image, ImageDraw

"""
Warning: this script is deprecated and is only included here for documentary purposes.
The model is inefficient and the evaluation metrics are semantically wrong.
"""

URL = "https://nexus.spaceapplications.com/repository/raw-km/infuse/infuse-dl-dataset-v0.0.8.tar.gz"
DATA_DIR = os.path.expanduser("~/.datasets")
MODEL_DIR = ".model/run/"
EVAL_DIR = MODEL_DIR + "eval-acc/"
IMG_H, IMG_W = 406, 528
OUT_H, OUT_W, OUT_C = 5, 7, 3
TRAIN_SAMPLES = 13198
EVAL_SAMPLES = 1226

BATCH_SIZE = 32
LEARN_RATE = 0.01
DROPOUT_RATE = 0.1
TRAIN_STEPS = math.ceil(TRAIN_SAMPLES/BATCH_SIZE)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    util.write_git_description(MODEL_DIR)
    dir = download_dataset(URL, DATA_DIR)

    estimator = create_estimator()
    evaluator = tf.contrib.estimator.InMemoryEvaluatorHook(estimator,
                                                           every_n_iter=TRAIN_STEPS,
                                                           input_fn=lambda: eval_dataset(dir, BATCH_SIZE))
    estimator.train(input_fn=lambda: train_dataset(dir, BATCH_SIZE), hooks=[evaluator])


def create_estimator():
    session_config = tf.ConfigProto(allow_soft_placement=True)
    run_config = tf.estimator.RunConfig(session_config=session_config,
                                        save_summary_steps=20,
                                        save_checkpoints_secs=5*60,
                                        log_step_count_steps=1,
                                        model_dir=MODEL_DIR)

    return tf.estimator.Estimator(model_fn=model_gpu, config=run_config, params={
        "batch_size": BATCH_SIZE,
        "learn_rate": LEARN_RATE,
        "img_h": IMG_H,
        "img_w": IMG_W,
        "dropout_rate": DROPOUT_RATE,
        "dist_threshold": 10,
        "prob_threshold": 0.5,
        "eval_dir": EVAL_DIR
    })


def eval_dataset(dir, batch_size):
    dataset = tf.data.TFRecordDataset([os.path.join(dir, "eval.tfrecords")])
    return dataset.map(parse, num_parallel_calls=24).batch(batch_size).prefetch(batch_size * 10)


def train_dataset(dir, batch_size):
    dataset = tf.data.TFRecordDataset([os.path.join(dir, "train.tfrecords")])
    return dataset.shuffle(128).map(parse, num_parallel_calls=24).repeat().batch(batch_size).prefetch(batch_size * 10)


def parse(record):
    features = {
        'label': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'image': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'name': tf.FixedLenFeature([], tf.string)
    }
    example = tf.parse_single_example(record, features)
    name = example["name"]
    point = tf.to_float(example["label"])

    point_available = tf.logical_not(tf.equal(tf.reduce_max(point), -1))
    label = tf.cond(point_available,
                    lambda: parse_label(point, IMG_W, IMG_H, OUT_W, OUT_H),
                    lambda: tf.zeros([OUT_H, OUT_W, OUT_C], dtype=tf.float32))

    image = tf.to_float(example["image"])
    image = tf.reshape(image, (IMG_H, IMG_W, 1)) / 255
    return image, {"available": point_available, "point": point, "label": label, "name": name}


def parse_label(point, img_w, img_h, out_w, out_h):
    block_dims = tf.constant([img_w/out_w, img_h/out_h])  # [528/7=75.4, 406/5=81.2]
    offset_w, offset_h = tf.unstack(tf.floor(point/block_dims))  # Relative offset of the label in the output image.
    point_x, point_y = tf.unstack((point % block_dims)/block_dims)  # Get the part after the decimal point as a relative pos within a block.
    offset_w, point_x = limit(offset_w, point_x, out_w)
    offset_h, point_y = limit(offset_h, point_y, out_h)
    point = tf.concat([[1.0], [point_x, point_y]], axis=0)  # Add probability bit with value 1.0.
    pixel = tf.reshape(point, [1, 1, 3])  # Reshape to image pixel with 3 channels.
    return tf.image.pad_to_bounding_box(pixel, tf.to_int32(offset_h), tf.to_int32(offset_w), out_h, out_w)


def limit(offset, point, max):
    return tf.cond(offset >= max, lambda: (max - 1.0, 1.0), lambda: (offset, point))


def download_dataset(url, dest):
    name = re.sub(r'^.*/', "", url)
    sub_dir = re.sub(r'(\.[^\.]*){1,2}$', "", name)
    out_dir = os.path.join(dest, sub_dir)
    if os.path.exists(out_dir):
        tf.logging.info("dataset available at: {}".format(out_dir))
        return out_dir
    tf.logging.info("download and extract {} to {}".format(url, dest))
    tf.keras.utils.get_file(origin=url, fname=os.path.join(dest, name), cache_dir=dest, cache_subdir=sub_dir, extract=True)
    return out_dir


def model_gpu(features, labels, mode, params):
    with tf.device("/gpu:0"):
        return model(features, labels, mode, params)


def model(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    layer = conv_layer(features, is_training, filter_size=7, num_filters=64, strides=2, dropout_rate=params["dropout_rate"])  # (?, 200, 261, 64)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2)  # (?, 100, 130, 64)

    layer = conv_layer(layer, is_training, filter_size=3, num_filters=192, strides=1, dropout_rate=params["dropout_rate"])  # (?, 100, 130, 192)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2)  # (?, 50, 65, 192)

    layer = conv_layer(layer, is_training, filter_size=1, num_filters=128, strides=1, dropout_rate=params["dropout_rate"])  # (?, 50, 65, 192)
    layer = conv_layer(layer, is_training, filter_size=3, num_filters=256, strides=1, dropout_rate=params["dropout_rate"])  # (?, 50, 65, 256)
    layer = conv_layer(layer, is_training, filter_size=1, num_filters=256, strides=1, dropout_rate=params["dropout_rate"])  # (?, 50, 65, 256)
    layer = conv_layer(layer, is_training, filter_size=3, num_filters=512, strides=1, dropout_rate=params["dropout_rate"])  # (?, 50, 65, 512)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2)  # (?, 25, 32, 512)

    for _ in range(4):
        layer = conv_layer(layer, is_training, filter_size=1, num_filters=256, strides=1, dropout_rate=params["dropout_rate"])  # (?, 25, 32, 256)
        layer = conv_layer(layer, is_training, filter_size=3, num_filters=512, strides=1, dropout_rate=params["dropout_rate"])  # (?, 25, 32, 512)
    layer = conv_layer(layer, is_training, filter_size=1, num_filters=512, strides=1, dropout_rate=params["dropout_rate"])  # (?, 25, 32, 512)
    layer = conv_layer(layer, is_training, filter_size=3, num_filters=1024, strides=1, dropout_rate=params["dropout_rate"])  # (?, 25, 32, 1024)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2)  # (?, 12, 16, 1024)

    for _ in range(2):
        layer = conv_layer(layer, is_training, filter_size=1, num_filters=512, strides=1, dropout_rate=params["dropout_rate"])  # (?, 12, 16, 512)
        layer = conv_layer(layer, is_training, filter_size=3, num_filters=1024, strides=1, dropout_rate=params["dropout_rate"])  # (?, 12, 16, 1024)
    layer = conv_layer(layer, is_training, filter_size=3, num_filters=1024, strides=1, dropout_rate=params["dropout_rate"])  # (?, 12, 16, 1024)
    layer = conv_layer(layer, is_training, filter_size=3, num_filters=1024, strides=2, dropout_rate=params["dropout_rate"])  # (?, 5, 7, 1024)

    layer = conv_layer(layer, is_training, filter_size=3, num_filters=1024, strides=1, dropout_rate=params["dropout_rate"])  # (?, 5, 7, 1024)
    layer = conv_layer(layer, is_training, filter_size=3, num_filters=1024, strides=1, dropout_rate=params["dropout_rate"])  # (?, 5, 7, 1024)

    layer = conv_layer(layer, is_training, filter_size=1, num_filters=4096, strides=1, dropout_rate=params["dropout_rate"])  # (?, 5, 7, 4096)
    logits = conv_layer(layer, is_training, filter_size=1, num_filters=3, strides=1, dropout_rate=params["dropout_rate"])  # (?, 5, 7, 3)

    loss = get_loss(logits, labels["label"], labels["available"])

    logit_probs, _, _, logit_xy = find_max(logits, params["img_w"], params["img_h"])

    dists = distance(logit_xy, labels["point"])
    weights = tf.equal(logit_probs > params["prob_threshold"], labels["available"])
    mean = tf.metrics.mean(dists, weights=tf.to_float(weights))
    tf.summary.scalar("train/dist_mean", mean[1])

    acc_avail, acc_preds, acc_labels = overall_accuracy(logit_probs, dists, labels["available"], params["prob_threshold"], params["dist_threshold"])
    acc_overall = tf.metrics.accuracy(labels=acc_labels, predictions=acc_preds)
    tf.summary.scalar("train/accuracy_overall", acc_overall[1])

    acc_preds_point, acc_labels_point = boolean_mask_all([acc_preds, acc_labels], labels["available"])
    acc_point = tf.metrics.accuracy(labels=acc_labels_point, predictions=acc_preds_point)
    tf.summary.scalar("train/internal/accuracy_point", acc_point[1])

    acc_pred = tf.metrics.accuracy(labels=tf.to_float(labels["available"]), predictions=acc_avail)
    tf.summary.scalar("train/internal/accuracy_pred", acc_pred[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        #positive_mask = tf.equal(acc_preds, acc_labels)
        #img_data = [features, labels["name"], labels["point"], logit_xy, logit_probs]
        #args = boolean_mask_all(img_data, positive_mask)
        #positive = tf.py_func(util.image_draw, args + ["green"], Tout=tf.uint8)
        #tf.summary.image("eval/positive", positive, max_outputs=1)

        #negative_mask = tf.not_equal(acc_preds, acc_labels)
        #args = boolean_mask_all(img_data, negative_mask)
        #negative = tf.py_func(util.image_draw, args + ["red"], Tout=tf.uint8)
        #tf.summary.image("eval/negative", negative, max_outputs=1)

        #eval_summary_hook = tf.train.SummarySaverHook(save_steps=1,
        #                                              output_dir=params["eval_dir"],
        #                                              summary_op=tf.summary.merge_all())
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss)#,
                                          #evaluation_hooks=[eval_summary_hook])

    assert mode == tf.estimator.ModeKeys.TRAIN
    # trainable_variables = tf.trainable_variables()
    # for v in trainable_variables:
    #     name = v.name.replace(":", "_")
    #     tf.summary.scalar("train_var/var_mean/" + name, tf.reduce_mean(v))
    #     tf.summary.histogram("train_var/var_hist/" + name, v)

    # trainable_variables = [tf.reshape(v, [-1]) for v in trainable_variables]
    # trainable_variables = tf.concat(trainable_variables, axis=0)
    # tf.summary.histogram("train_vars/vars_hist", trainable_variables)

    optimiser = tf.train.AdamOptimizer(learning_rate=params["learn_rate"])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        minimize = optimiser.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=minimize)


def get_loss(logits, labels, available):
    batch_size = tf.shape(logits)[0]

    pred_logits = logits[:, :, :, 0]
    pred_labels = labels[:, :, :, 0]
    pred_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=pred_labels, logits=pred_logits)

    weights = tf.reshape(tf.to_float(available), [batch_size, 1])  # If no points are available, only the probability bits determine the loss
    xy_logits = tf.reshape(logits[:, :, :, 1:3], [batch_size, -1])
    xy_labels = tf.reshape(labels[:, :, :, 1:3], [batch_size, -1])
    xy_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=xy_labels, logits=xy_logits, weights=weights)
    return pred_loss + xy_loss


def distance(logits_xy, labels_xy):
    return tf.norm(labels_xy - logits_xy, axis=1)


def overall_accuracy(logit_probs, dist, labels_avail, prob_threshold, dist_threshold):
    avail = tf.to_float(logit_probs > prob_threshold)  # (?)
    near = avail * tf.to_float(dist <= dist_threshold)  # (?)
    preds = avail + near  # (?)
    labels = tf.to_float(labels_avail) * 2.0  # (?)
    return avail, preds, labels


def find_max(logits, img_w, img_h):
    batch_size, logits_h, logits_w, logits_c = tf.unstack(tf.to_float(tf.shape(logits)))  # ?, 5.0, 7.0, 3.0

    logits = tf.sigmoid(logits)  # limit probs, x and y between 0..1
    probs = logits[:, :, :, 0]  # (?,5,7,3): only get the probability scores
    probs = tf.reshape(probs, [-1, logits_h*logits_w])  # (?, 5*7) reshape to linear array for easy argmaxing
    highest_prob_indexes = tf.to_float(tf.argmax(probs, axis=1))  # (?) per image in the batch, the highest probability

    rel_offsets_y = tf.floor(highest_prob_indexes / logits_w)  # (?, 1)
    rel_offsets_x = tf.floor(highest_prob_indexes % logits_w)  # (?, 1)
    rel_offsets = tf.stack([rel_offsets_x, rel_offsets_y], axis=1)  # (?, 2) stack x and y together.

    batch_range = tf.reshape(tf.range(tf.to_int32(batch_size)), [-1, 1])  # (?, 1) range of numbers 0..batch_size
    highest_prob_indexes = tf.reshape(tf.to_int32(highest_prob_indexes), [-1, 1])  # (?, 1) reshape for concat with range.
    highest_prob_indexes = tf.concat([batch_range, highest_prob_indexes], 1)  # (?, 2)
    probs = tf.reshape(logits, [-1, logits_h*logits_w, logits_c])  # (?, 5*7, 3) reshape for easy gathering.
    pixels = tf.gather_nd(probs, highest_prob_indexes)  # (?, 3)

    rel_pos = pixels[:, 1:3]  # (?, 2)
    probs = pixels[:, 0]  # (?, 1)

    block_dims = [img_w/logits_w, img_h/logits_h]  # (2)
    abs_pos = rel_offsets*block_dims + rel_pos * block_dims
    return probs, rel_offsets, rel_pos, abs_pos


def conv_layer(input, is_training, filter_size, num_filters, activation=tf.nn.relu, strides=1, dropout_rate=0.1):
    padding = "same" if strides == 1 else "valid"
    conv = tf.layers.conv2d(input, num_filters, filter_size, padding=padding, strides=strides, activation=activation)  # (?, 32, 32, 32)
    norm = tf.layers.batch_normalization(conv,  training=is_training)
    drop = tf.layers.dropout(norm, rate=dropout_rate, training=is_training)  # (?, 32, 32, 32)
    return drop


def boolean_mask_all(tensors, mask):
    return [tf.boolean_mask(t, mask) for t in tensors]

if __name__ == "__main__":
    main()
