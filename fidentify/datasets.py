"""Support code for face recognition datasets. Makes it easy to lazily load
batches of pictures of the same person."""

from abc import ABC, abstractmethod
from csv import DictReader
from os import path

import numpy as np

from fidentify.utils import get_paths


class FaceDataset(ABC):
    """Interface which all face recognition datasets must implement"""

    def __init__(self, image_handler):
        self.image_handler = image_handler

    @abstractmethod
    def person_pictures(self, is_train):
        """Gives dictionary mapping {person ID: [picture ID]}. Individual
        person and picture IDs can be whatever type, they just need to be
        understood by get_picture. is_train indicates whether to draw from
        training set (True) or testing set (False)."""
        pass

    @abstractmethod
    def get_picture(self, picture_id):
        """Load a picture from disk or memory (depends how big dataset is)."""
        pass

    def keypoint_locs(self, picture_id):
        raise NotImplementedError("Only subclasses can implement this")


class LFW(FaceDataset):
    """Labelled Faces in the Wild wrapper."""

    def __init__(self, lfw_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Takes path to LFW directory produced by fetchdata.sh"""
        self.train_dir = path.join(lfw_path, 'train')
        self.test_dir = path.join(lfw_path, 'test')
        self._person_pictures_train = self._walk_people(self.train_dir)
        self._person_pictures_test = self._walk_people(self.test_dir)

        # load everything into memory lazily, caching here
        # we can't do this for bigger datasets
        self._image_cache = {}

    @staticmethod
    def _walk_people(subdir):
        """Get all identities from either train or test directory."""
        subdir = path.abspath(subdir)
        # see http://stackoverflow.com/a/16595356
        subdir_parts = path.normpath(subdir).split(path.sep)
        rv = {}
        for subpath in get_paths(subdir):
            parts = path.normpath(subpath).split(path.sep)
            rel_parts = parts[len(subdir_parts):]
            # should only be two parts
            assert len(rel_parts) == 2, \
                "Expecting path of form '<person>/<image>', not '%s'" \
                % (rel_parts,)
            name, _ = rel_parts
            rv.setdefault(name, []).append(subpath)
        return rv

    def person_pictures(self, is_train):
        if is_train:
            return self._person_pictures_train
        return self._person_pictures_test

    def get_picture(self, picture_id):
        # cache picture on first load
        if picture_id not in self._image_cache:
            self._image_cache[picture_id] \
                = self.image_handler.load_as_colspace(picture_id)
        return self._image_cache[picture_id]


class KaggleFK(FaceDataset):
    """Kaggle Facial Keypoints wrapper."""

    def __init__(self, kfk_dir, used_keypoints, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.used_keypoints = used_keypoints
        self.kfk_dir = kfk_dir
        self.train_kp, self.train_im = self._load_csv(
            path.join(kfk_dir, 'train.csv'))
        self.test_kp, self.test_im = self._load_csv(
            path.join(kfk_dir, 'test.csv'))

    @staticmethod
    def _load_csv(self, csv_path):
        reader = DictReader(csv_path)
        keypoints = []
        images = []
        for row_idx, row in enumerate(reader):
            # convert to 96 * 96 image
            image_ints = list(map(int, row['Image'].split()))
            img = np.array(image_ints, dtype='uint8').reshape((96, 96))
            images.append(img)
            kp_dict = {}
            for kp in self.used_keypoints:
                kp_x = row[kp + '_x']
                kp_y = row[kp + '_y']
                kp_dict[kp] = (kp_x, kp_y)
        return keypoints, images

    @abstractmethod
    def person_pictures(self, is_train):
        if is_train:
            base_ind = 0
            image_list = self.train_im
        else:
            base_ind = len(self.train_im)
            image_list = self.test_im
        return {
            idx: img for idx, img in enumerate(image_list, start=base_ind)
        }

    def _unmap_id(self, ident):
        ntrain = len(self.train_im)
        ntest = len(self.test_im)
        assert (ntrain + ntest) > ident >= 0
        if ident >= ntrain:
            real_id = ident - self.train_im
            return real_id, False
        return ident, True

    @abstractmethod
    def get_picture(self, picture_id):
        real_id, is_train = self._unmap_id(picture_id)
        if is_train:
            return self.train_im[real_id]
        return self.test_im[real_id]

    def keypoint_locs(self, picture_id):
        pass
