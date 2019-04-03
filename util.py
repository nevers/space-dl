import os
import re
import subprocess
import itertools
import time
import datetime
import math
import colour

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from enum import Enum


def default_log_handler(result):
    result = ["{}={:.1f}".format(key, value) for key, value in result.items() if value]
    return ", ".join(result)


def train_and_eval(session, batch_size, total_train_samples, model,
                   train_ds_init, eval_ds_init,
                   save_summary_steps=50,
                   save_model_steps=100,
                   eval_model_steps=None,
                   post_train_hooks=[],
                   post_eval_hooks=[],
                   log_dir=".model/run/",
                   log_handler=default_log_handler):

    model_dir = os.path.join(log_dir, "model")
    train_dir = os.path.join(log_dir, "train")
    eval_dir = os.path.join(log_dir, "eval")
    model_path = os.path.join(model_dir, "model.ckpt")

    mkdir(model_dir, train_dir, eval_dir)

    log("using log dir: {}".format(log_dir))
    log("using model path: {}".format(model_path))
    log("save summaries every {} steps".format(save_summary_steps))
    log("save model every {} steps".format(save_model_steps))
    if eval_model_steps:
        log("eval model every {} steps".format(eval_model_steps))
    else:
        log("model eval disabled")
    log("acronymns list: ET=Elasped Time, EP=EPoch, ST=STep")

    start_time = time.time()
    with session:
        log("run model function")
        train_ops, eval_ops = model()
        train_summaries_op = tf.summary.merge_all()

        eval_summaries_op = tf.summary.merge_all("eval")
        eval_metrics = eval_ops["_eval_metrics"]
        eval_metrics_update_op = {k: v[1] for k, v in eval_metrics.items()}
        eval_ops = dict(eval_ops)
        del eval_ops["_eval_metrics"]

        eval_metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        eval_metrics_vars_init_op = tf.variables_initializer(eval_metrics_vars)

        session.run([tf.local_variables_initializer(),
                     tf.global_variables_initializer(),
                     train_ds_init])

        train_saver = tf.train.Saver()

        try:
            train_saver.restore(session, model_path)
            model_load_elasped = time.time() - start_time
            log("model restored from {} in {:.2f}s".format(model_path, model_load_elasped))
        except ValueError:
            log("failed to restore model from {}".format(model_path))
            log("start from scratch")

        train_summary_saver = tf.summary.FileWriter(train_dir)
        eval_summary_saver = tf.summary.FileWriter(eval_dir)
        for _ in itertools.count():
            begin_time = time.time()

            result = session.run(train_ops)
            for hook in post_train_hooks:
                hook(result)

            step = session.run(tf.train.get_or_create_global_step())

            curr = step * batch_size
            epoch = curr // total_train_samples
            sample = curr % total_train_samples
            progress = (sample / total_train_samples) * 100

            end_time = time.time()
            elasped_time_f = datetime.timedelta(seconds=math.floor(end_time - start_time))
            batch_time_f = end_time - begin_time

            msg = "ET {} EP {} {:.1f}% ST {} ({:.2f}s)"
            msg = msg.format(elasped_time_f,
                             epoch,
                             progress,
                             step,
                             batch_time_f,
                             **result)

            if log_handler:
                msg += ": " + log_handler(result)
            log(msg)

            if step and step % save_summary_steps == 0:
                log("save summary")
                summaries = session.run(train_summaries_op)
                train_summary_saver.add_summary(summaries, step)
                train_summary_saver.flush()

            if step and step % save_model_steps == 0:
                log("save model")
                train_saver.save(session, model_path)

            if step and eval_model_steps and step % eval_model_steps == 0:
                log("eval model")
                session.run([eval_ds_init, eval_metrics_vars_init_op])
                while True:
                    try:
                        if eval_summaries_op:
                            eval_metrics_update, eval_summaries, result = session.run([eval_metrics_update_op, eval_summaries_op, eval_ops])
                        else:
                            eval_metrics_update, result = session.run([eval_metrics_update_op, eval_ops])

                        result["step"] = step
                        result["epoch"] = epoch
                        for hook in post_eval_hooks:
                            hook(result)
                    except tf.errors.OutOfRangeError:
                        break
                log("save eval summary")

                for tag, val in eval_metrics_update.items():
                    # You can no longer show multiple summaries into the same plot by simply reusing the
                    # same tag. Tensorboard will automatically make those summaries unique by adding "_1"
                    # to the tag name and thus forcing them to show in a separate plot. If you want to
                    # work around this limitation, you can generate the Summary protocol buffers yourself,
                    # and then manually add them to the summary.FileWriter.
                    summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
                    eval_summary_saver.add_summary(summ, step)
                if eval_summaries_op:
                    eval_summary_saver.add_summary(eval_summaries, step)
                eval_summary_saver.flush()
                session.run(train_ds_init)


def trainable_variables_summary(name):
    trainable_variables = tf.trainable_variables()
    for v in trainable_variables:
        name = v.name.replace(":", "_")
        tf.summary.scalar(name + "/var_mean/" + name, tf.reduce_mean(v))
        tf.summary.histogram(name + "/var_hist/" + name, v)

    trainable_variables = [tf.reshape(v, [-1]) for v in trainable_variables]
    trainable_variables = tf.concat(trainable_variables, axis=0)
    tf.summary.histogram(name + "/vars_hist", trainable_variables)


def log(msg):
    time_f = datetime.datetime.fromtimestamp(math.floor(time.time()))
    print("[{}]".format(time_f), msg)


def boolean_mask_all(tensors, mask):
    return {k: tf.boolean_mask(v, mask) for k, v in tensors.items()}


def draw_image(pool, images, logit_probs, logit_probs_max, label_xy, logit_xy, color, names, dir):
    images = images * 255
    images = images.astype(np.uint8)
    logit_xy = logit_xy.astype(np.uint16)
    batch_size = images.shape[0]

    data = [[logit_probs[i],
             logit_probs_max[i],
             logit_xy[i, 0],
             logit_xy[i, 1],
             label_xy[i, 0],
             label_xy[i, 1],
             images[i],
             color,
             names[i],
             dir] for i in range(batch_size)]

    pool.starmap_async(draw_one_image, data)


def draw_one_image(logit_probs, logit_probs_max, logit_x, logit_y, label_x, label_y, image, color, name, dir):
    name = name.decode() if isinstance(name, bytes) else name
    color = color.decode() if isinstance(color, bytes) else color
    blue = colour.Color("blue")
    red = colour.Color("red")
    color_range = list(blue.range_to(red, 256))

    h, w = image.shape[0:2]
    h_out, w_out = logit_probs.shape[0:2]
    y_weight = h // h_out
    x_weight = w // w_out
    image = Image.fromarray(image)
    image = image.convert("RGBA")

    image_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_drawer = ImageDraw.Draw(image_overlay)

    for y in range(h_out):
        for x in range(w_out):
            prob = logit_probs[y, x]
            color_prob = int(prob * 255)
            alpha_prob = int(prob * 175)
            fill = np.array(color_range[color_prob].get_rgb()) * 255
            fill = fill.astype(np.uint8).tolist()
            fill = fill + [alpha_prob]
            pos_x = x * x_weight
            pos_y = y * y_weight
            overlay_drawer.rectangle([(pos_x, pos_y), (pos_x+x_weight, pos_y+y_weight)], fill=tuple(fill))  # fill=(255, 255, 255, 128))

    image = Image.alpha_composite(image, image_overlay)
    text = "{} label: {:.0f},{:.0f}, pred: {:.2f}:{:.0f},{:.0f}".format(name, label_x, label_y, logit_probs_max, logit_x, logit_y)
    drawer = ImageDraw.Draw(image)
    drawer.text((10, 10), text, fill=color)
    drawer.line((label_x - 5, label_y, label_x + 5, label_y), fill="yellow")
    drawer.line((label_x, label_y - 5, label_x, label_y + 5), fill="yellow")
    drawer.line((logit_x - 5, logit_y, logit_x + 5, logit_y), fill=color)
    drawer.line((logit_x, logit_y - 5, logit_x, logit_y + 5), fill=color)

    image.save(os.path.join(dir, name))


def download_dataset(url, dest=os.path.expanduser("~/.datasets")):
    name = re.sub(r'^.*/', "", url)
    sub_dir = re.sub(r'(\.[^\.]*){1,2}$', "", name)
    out_dir = os.path.join(dest, sub_dir)
    if os.path.exists(out_dir):
        tf.logging.info("dataset available at: {}".format(out_dir))
        return out_dir
    tf.logging.info("download and extract {} to {}".format(url, dest))
    tf.keras.utils.get_file(origin=url, fname=os.path.join(dest, name), cache_dir=dest, cache_subdir=sub_dir, extract=True)
    return out_dir


def write_git_description(dir):
    mkdir(dir)
    fnull = open(os.devnull, "w")
    cmd = "git describe --all --long > {}/git-description".format(dir)
    subprocess.call(cmd, shell=True, stdout=fnull, stderr=fnull)


def mkdir(*args):
    for file in args:
        try:
            os.makedirs(file)
        except:
            pass  # dir already exists
