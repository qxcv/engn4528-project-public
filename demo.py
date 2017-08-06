#!/usr/bin/env python3
"""Download required data/models and run our complete system on ANU Faces.
Should be run within an appropriately configured Python 3/Anaconda environment,
per README.md."""

import zipfile
from urllib.request import urlretrieve
import os
from subprocess import run

DATA_URL = 'http://users.cecs.anu.edu.au/~u5568237/engn4528/engn4528-data.zip'
COMMAND_BASE = [
    'python3', './test_on_group_photo.py', 'data/anu-identities.ini',
    'data/anufaces/anuclass01.jpg'
]
OPTIONS = [
    ('Detector with Eigenface recognition', ['--method', 'eigen']),
    ('Detector with VAE recognition', ['--method', 'plainvae']),
    ('Detector with IDVAE recognition', ['--method', 'idvae']),
]
DL_FILE = 'data/.everything-downloaded'


def needs_download():
    # should we download data?
    return not os.path.isfile(DL_FILE)


def do_download():
    # download data if we need to
    print('Downloading (may take a while)')
    fn, _ = urlretrieve(DATA_URL)
    print('Extracting file to data/')
    with zipfile.ZipFile(fn, "r") as fp:
        fp.extractall("data" + os.path.sep)

    # make sure we don't download next time
    with open(DL_FILE, 'w') as fp:
        fp.write('this will stop data from being re-downloaded\n')


if __name__ == '__main__':
    if needs_download():
        do_download()
    else:
        print("Data is there, skipping download")
    print('')
    print('Demos available')
    print('===============')
    print('')
    for idx, opt in enumerate(OPTIONS):
        print('Demo %d: %s' % (idx, opt[0]))
    print('')
    # this will fail if the number is out of range
    selected = int(
        input(
            'Enter a number (0-%d) corresponding to the demo you want to see: '
            % (len(OPTIONS) - 1)))
    sel_name, sel_cmd = OPTIONS[selected]
    full_cmd = COMMAND_BASE + sel_cmd
    print('Selected %s' % sel_name)
    print('Running command %s' % ' '.join(full_cmd))
    run(full_cmd)
    print('Done!')
