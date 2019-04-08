#!/usr/bin/env python3

import os
import subprocess
import csv
import util
import re

"""
This script:
1. collates each picture in the evaluation data together with the model output, 
   over all evaluation epochs into one montage. Every montage thus shows how 
   the model predictions evolve over each training epoch.
2. collates all pictures per evaluation epoch into a video. In other words, a video
   is created per epcoch that shows the model predictions and behavior at that epoch
   over the full evaluation dataset.
"""
DATA_DIR = os.path.expanduser("~/.datasets/infuse-dl-dataset-v0.0.7-rand-eval/")
TARGET = ".model/run/eval-img"
VIDEO_DIR = ".model/run/eval-video"
MONTAGE_DIR = ".model/run/eval-montage"


def main():
    dirs = [os.path.join(TARGET, dir) for dir in os.listdir(TARGET) if os.path.isdir(os.path.join(TARGET, dir))]
    dirs = sort_alphanumerical(dirs)
    create_videos(dirs)
    create_montage(dirs)


def create_montage(dirs):
    util.mkdir(MONTAGE_DIR)
    eval_files = get_eval_files()
    for eval_file in eval_files:
        files = [os.path.join(dir, eval_file) for dir in dirs]
        files = [file for file in files if os.path.isfile(file)]
        files = " ".join(files)
        target = os.path.join(MONTAGE_DIR, eval_file)
        cmd = f"montage -background none {files} {target}"
        subprocess.call(cmd, shell=True)


def sort_alphanumerical(l):
    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_eval_files():
    with open(os.path.join(DATA_DIR, "eval.csv")) as file:
        reader = csv.reader(file, delimiter=",")
        return [row[0] for row in reader]


def create_videos(dirs):
    util.mkdir(VIDEO_DIR)
    for dir in dirs:
        basename = os.path.basename(dir)
        target = os.path.join(VIDEO_DIR, basename)
        cmd = f"ffmpeg -f image2 -pattern_type glob -i '{dir}/*.png' -t 30 -c:v libx264 -profile:v high -crf 25 -pix_fmt yuv420p {target}.mp4"
        subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()
