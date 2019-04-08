#!/usr/bin/env python3

import re
import os
import csv
import util
import math
import itertools
import time
import datetime

import multiprocessing as mp
import tensorflow as tf
import numpy as np

from PIL import Image, ImageDraw
from keras.models import load_model, Model, Input
from keras import backend as K

"""
A convolutional neural network inspired by the YOLO algorithm to support spacecraft docking in space.
It will automatically perform the following operations:
1. download the dataset to ~/.datasets
2. load the the pretrained model from yolov3.h5
3. retrain the second half of the model whilst keeping the first half of the model parameters constant on your GPU.
4. evaluate the model 2 times per epoch on your GPU
5. write training and evalation metrics for tensorboard to .model/{train,eval}
6. write positive and negative evaluation images, annotated with their predictions to .model/eval-img
7. periodically save the model state to .model/model
"""


URL = "https://nexus.spaceapplications.com/repository/raw-km/infuse/infuse-dl-dataset-v0.0.7-rand-eval.tar.gz"
DATA_DIR = os.path.expanduser("~/.datasets")
LOG_DIR = ".model/run/"
MODEL_PATH = ".model/yolov3.h5"

ORIG_IMG_H, ORIG_IMG_W = 406, 528
IMG_H, IMG_W = 416, 544  # The yolo model requires input img dimensions to be a multiple of 32.
OUT_H, OUT_W, OUT_C = 52, 68, 3
TRAIN_SAMPLES = 11539
EVAL_SAMPLES = 2885

BATCH_SIZE = 8
LEARN_RATE = 0.001
DROPOUT_RATE = 0.1
STEPS_PER_EPOCH = math.ceil(TRAIN_SAMPLES/BATCH_SIZE)

pool = mp.Pool(mp.cpu_count())


def main():
    util.write_git_description(LOG_DIR)
    dir = util.download_dataset(URL, DATA_DIR)

    train_ds = train_dataset(dir, BATCH_SIZE)
    eval_ds = eval_dataset(dir, BATCH_SIZE)
    it = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
    train_ds_init = it.make_initializer(train_ds)
    eval_ds_init = it.make_initializer(eval_ds)

    params = {
        "batch_size": BATCH_SIZE,
        "learn_rate": LEARN_RATE,
        "img_h": IMG_H,
        "img_w": IMG_W,
        "dropout_rate": DROPOUT_RATE,
        "dist_threshold": 10,
        "prob_threshold": 0.5,
        "model": MODEL_PATH,
    }

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    util.train_and_eval(session=sess,
                        batch_size=BATCH_SIZE,
                        total_train_samples=TRAIN_SAMPLES,
                        train_ds_init=train_ds_init,
                        eval_ds_init=eval_ds_init,
                        eval_model_steps=STEPS_PER_EPOCH//2,
                        post_eval_hooks=[save_images],
                        log_dir=LOG_DIR,
                        model=lambda: model_gpu(it.get_next(), params))


def save_images(eval_results):
    img_pos, img_neg = eval_results["img_pos"], eval_results["img_neg"]
    dir = "ep{:02}-es{}-".format(eval_results["epoch"], eval_results["step"])
    eval_img_pos_dir = os.path.join(LOG_DIR, "eval-img", dir + "pos")
    eval_img_neg_dir = os.path.join(LOG_DIR, "eval-img", dir + "neg")
    util.mkdir(eval_img_pos_dir, eval_img_neg_dir)

    img_pos["pool"] = pool
    img_pos["color"] = "green"
    img_pos["dir"] = eval_img_pos_dir
    util.draw_image(**img_pos)

    img_neg["pool"] = pool
    img_neg["color"] = "red"
    img_neg["dir"] = eval_img_neg_dir
    util.draw_image(**img_neg)


def eval_dataset(dir, batch_size):
    dataset = tf.data.TFRecordDataset([os.path.join(dir, "eval.tfrecords")])
    return dataset.shuffle(TRAIN_SAMPLES).map(parse, num_parallel_calls=24).batch(batch_size).prefetch(batch_size * 10)


def train_dataset(dir, batch_size):
    dataset = tf.data.TFRecordDataset([os.path.join(dir, "train.tfrecords")])
    return dataset.shuffle(TRAIN_SAMPLES).map(parse, num_parallel_calls=24).repeat().batch(batch_size).prefetch(batch_size * 10)


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
    point = tf.cond(point_available,
                    lambda: point + [(IMG_W - ORIG_IMG_W)/2, (IMG_H - ORIG_IMG_H)/2],
                    lambda: point)

    label = tf.cond(point_available,
                    lambda: parse_label(point, IMG_W, IMG_H, OUT_W, OUT_H),
                    lambda: tf.zeros([OUT_H, OUT_W, OUT_C], dtype=tf.float32))

    image = tf.to_float(example["image"])
    image = tf.reshape(image, (ORIG_IMG_H, ORIG_IMG_W, 1)) / 255
    image = tf.image.resize_image_with_crop_or_pad(image, IMG_H, IMG_W)
    image = tf.image.grayscale_to_rgb(image)
    return {"image": image, "available": point_available, "point": point, "label": label, "name": name}


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


def model_gpu(batch, params):
    with tf.device("/gpu:0"):
        return model(batch, params)


def model(batch, params):
    K.set_learning_phase(1)
    m = load_model(params["model"], compile=False)
    m.load_weights(params["model"], by_name=True)
    m = Model(m.input, m.get_layer("conv2d_75").output)  # Skip last layer (?, out_h, out_w, 255)
    for i, layer in enumerate(m.layers):
        if i == 152:
            assert layer.name == "add_19"
        layer.trainable = (i > 152)
    m = m(batch["image"])

    logits = tf.layers.conv2d(inputs=m, filters=3, kernel_size=1, strides=1, padding="same")  # (?, out_h, out_w, out_c)
    loss = get_loss(logits, batch["label"], batch["available"])
    decayed_lr = tf.train.exponential_decay(learning_rate=params["learn_rate"],
                                            global_step=tf.train.get_or_create_global_step(),
                                            decay_steps=100,
                                            decay_rate=0.95,
                                            staircase=True)
    optimiser = tf.train.AdamOptimizer(learning_rate=decayed_lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        minimize = optimiser.minimize(loss, global_step=tf.train.get_or_create_global_step())

    loss_tag = "metrics/loss"
    tf.summary.scalar(loss_tag, loss)

    logits = tf.sigmoid(logits)  # limit probs, x and y between 0..1
    logit_probs_max, _, _, logit_xy = find_max(logits, params["img_w"], params["img_h"])
    distance = dist(logit_xy, batch["point"])

    prob_correct = tf.to_float(tf.equal(batch["available"], logit_probs_max > params["prob_threshold"]))

    avail_f = tf.to_float(batch["available"])
    point_prob_correct_count = tf.reduce_sum(prob_correct * avail_f) + 0.0001
    dist_mean = tf.reduce_sum(distance * prob_correct * avail_f) / point_prob_correct_count
    dist_mean_tag = "metrics/dist_mean"
    # For all samples with a visible tip and a correct prediction bit, the average distance between
    # the labeled point and estimated point.
    tf.summary.scalar(dist_mean_tag, dist_mean)

    acc, acc_preds, acc_labels = overall_accuracy(logit_probs_max, distance, batch["available"], params["prob_threshold"], params["dist_threshold"])
    acc_overall_mean = tf.reduce_mean(tf.to_float(acc))
    acc_overall_mean_tag = "accuracy/overall_mean"
    # What percentage of the samples have a correct probability bit and when a point is available
    # are estimated within distance. If the sample doesn't have a tip, only the correctness of the
    # probability bit is taken into account.
    tf.summary.scalar(acc_overall_mean_tag, acc_overall_mean)

    acc_point_mean = tf.reduce_sum(tf.to_float(acc) * avail_f) / point_prob_correct_count
    acc_point_mean_tag = "accuracy/point_mean"
    # Of all the samples with a visible tip and a correct probability bit, what percentage was
    # below the distance threshold.
    tf.summary.scalar(acc_point_mean_tag, acc_point_mean)

    acc_prob_mean = tf.reduce_mean(prob_correct)
    acc_prob_mean_tag = "accuracy/prob_mean"
    # What percentage of the labels have a correct probability bit (i.e. within the threshold).
    tf.summary.scalar(acc_prob_mean_tag, acc_prob_mean)

    trainable_variables = tf.trainable_variables()
    for v in trainable_variables:
        name = v.name.replace(":", "_")
        tf.summary.scalar("train_var/var_mean/" + name, tf.reduce_mean(v))
        tf.summary.histogram("train_var/var_hist/" + name, v)

    trainable_variables = [tf.reshape(v, [-1]) for v in trainable_variables]
    trainable_variables = tf.concat(trainable_variables, axis=0)
    tf.summary.histogram("train_vars/vars_hist", trainable_variables)

    img_data = {
        "images": batch["image"],
        "names": batch["name"],
        "label_xy": batch["point"],
        "logit_xy": logit_xy,
        "logit_probs": logits[:, :, :, 0],
        "logit_probs_max": logit_probs_max
    }

    img_pos = util.boolean_mask_all(img_data, acc)
    img_neg = util.boolean_mask_all(img_data, tf.logical_not(acc))

    train_ops = {
        "loss": loss,
        "train_op": minimize,
        "dist_mean": dist_mean,
        "acc_overall_mean": acc_overall_mean,
        "acc_point_mean": acc_point_mean,
        "acc_prob_mean": acc_prob_mean
    }

    eval_ops = {
        "img_pos": img_pos,
        "img_neg": img_neg,
        "_eval_metrics": {
            loss_tag: tf.metrics.mean(loss),
            dist_mean_tag: tf.metrics.mean(distance, weights=prob_correct * avail_f),
            acc_overall_mean_tag: tf.metrics.accuracy(labels=acc_labels, predictions=acc_preds),
            acc_point_mean_tag: tf.metrics.accuracy(labels=acc_labels, predictions=acc_preds, weights=avail_f),
            acc_prob_mean_tag: tf.metrics.mean(prob_correct)
        }
    }

    return train_ops, eval_ops


def get_loss(logits, labels, available):
    batch_size = tf.shape(logits)[0]

    pred_logits = logits[:, :, :, 0]
    pred_labels = labels[:, :, :, 0]
    pred_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=pred_labels, logits=pred_logits)

    weights = tf.reshape(tf.to_float(available), [batch_size, 1])  # If no points are available, only the probability bits determine the loss
    xy_logits = tf.reshape(logits[:, :, :, 1:3], [batch_size, -1])
    xy_labels = tf.reshape(labels[:, :, :, 1:3], [batch_size, -1])
    xy_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=xy_labels, logits=xy_logits, weights=weights)
    return 5 * pred_loss + xy_loss


def dist(logits_xy, labels_xy):
    return tf.norm(labels_xy - logits_xy, axis=1)


def overall_accuracy(logit_probs_max, dist, labels_avail, prob_threshold, dist_threshold):
    avail = tf.to_float(logit_probs_max > prob_threshold)  # (?)
    near = avail * tf.to_float(dist <= dist_threshold)  # (?)
    preds = avail + near  # (?)
    labels = tf.to_float(labels_avail) * 2.0  # (?)
    acc = tf.equal(preds, labels)
    return acc, preds, labels


def find_max(logits, img_w, img_h):
    batch_size, logits_h, logits_w, logits_c = tf.unstack(tf.to_float(tf.shape(logits)))  # ?, 5.0, 7.0, 3.0

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


if __name__ == "__main__":
    main()
