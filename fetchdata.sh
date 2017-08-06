#!/bin/bash

# Gets all the big data files we need for our project. You can ignore this if
# you're running demo.py (this script will only run on Unix machines anyway).

set -e

cores=4  # change up if you have more CPU cores

fetchurl() {
    if [ ! -f "$2" ]; then
        echo "Downloading $1 to $2"
        curl "$1" --output "$2" || echo "Couldn't fetch $1; if it's on Partch then " \
            "you'll only be able to get it from within the ANU network"
    else
        echo "$2 already downloaded, skipping"
    fi
}

mkdir -p data

if [ ! -d data/yalefaces ]; then
    echo "Retrieving Yale Faces"
    fetchurl \
        "http://users.cecs.anu.edu.au/~u5568237/engn4528/yalefaces.zip" \
        "data/yalefaces.zip"
    unzip -q -u data/yalefaces.zip -d data/
    # rename everything to .gif
    find data/yalefaces/ -type f \( -name 'subject*' -and -not -name '*.gif' \) -exec mv {} {}.gif \;
    # crop all faces in-place
    for subdir in testset trainingset; do
        ./crop_images.py --size 128 "data/yalefaces/$subdir" "data/yalefaces/$subdir"
    done
else
    echo "Skipping Yale Faces retrieval (use rm -rf data/yalefaces/ and rerun to force download)"
fi

if [ ! -d data/lfw-deepfunneled ]; then
    echo "Retrieving Labelled Faces in the Wild"
    fetchurl \
        "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz" \
        "data/lfw-deepfunneled.tgz"
    tar xf "data/lfw-deepfunneled.tgz" -C data/

    echo "Cropping all images"
    # will chop a few px off each side of 250x250 images so that faces are
    # better centred (needs ImageMagick)
    find data/lfw-deepfunneled/ -name '*.jpg' -print0 | xargs -0i -P "$cores" -n 1 \
         convert "{}" -crop 128x128+61+61 "{}"

    echo "Enforcing train/test split"
    # split each subject directory into "train" or "test" subdirectory using
    # data/train_test_split_lfw.json
    mkdir data/lfw-deepfunneled/train
    mkdir data/lfw-deepfunneled/test
    python <<EOF
import json, shutil
with open('data/train_test_split_lfw.json') as fp:
    split = json.load(fp)
for mode in ['train', 'test']:
    subjs = split[mode + '_subjects']
    for subj in subjs:
        src = 'data/lfw-deepfunneled/' + subj
        dest = 'data/lfw-deepfunneled/' + mode + '/' + subj
        shutil.move(src, dest)
EOF
else
    echo "Skipping LFW fetch/crop (use rm -rf data/lfw-deepfunneled and rerun to force download)"
fi

if [ ! -d data/kfkeypoints ]; then
    echo "Fetching trained models"
    fetchurl \
        "http://users.cecs.anu.edu.au/~u5568237/engn4528/kfkeypoints.zip" \
        "data/kfkeypoints.zip"
    unzip data/kfkeypoints.zip -d data/
else
    echo "Skipping Kaggle facial keypoints download (rm -r data/kfkeypoints and rerun to force download)"
fi

if [ ! -d data/plain-vae-rgb ]; then
    echo "Fetching trained models"
    mkdir -p data/plain-vae-rgb
    fetchurl \
        "http://users.cecs.anu.edu.au/~u5568237/engn4528/plain-vae-rgb/encoder.h5" \
        "data/plain-vae-rgb/encoder.h5"
    fetchurl \
        "http://users.cecs.anu.edu.au/~u5568237/engn4528/plain-vae-rgb/decoder.h5" \
        "data/plain-vae-rgb/decoder.h5"
else
    echo "Skipping model download (rm -r data/models and rerun to force download)"
fi

# make a huge combined dataset with LFW and ANU faces, preprocessed in exactly
# the way we intend to preprocess them
# if [ ! -d data/combined-lfw-anu ]; then
#     echo "Making huge combined LFW/ANU dataset"
#     mkdir -p data/combined-lfw-anu
#     ./crop_images.py data/lfw-deepfunneled data/combined-lfw-anu
#     # throw (pre-cropped) ANU faces into training set, making sure that each
#     # goes into a subject-specific directory
#     for fname in data/anufaces/subjects/*.jpg; do
#         bn="$(basename "$fname")"
#         dest_dir="data/combined-lfw-anu/train/anu-$(echo "$bn" | cut -d _ -f 1)"
#         mkdir -p "$dest_dir"
#         dest_fn="$dest_dir/$bn"
#         cp -v "$fname" "$dest_fn"
#     done
# else
#     echo "Skipping creation of LFW/ANU dataset; delete data/combined-lfw-anu and rerun to make it again"
# fi
