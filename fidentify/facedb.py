"""Support code for various face recognition approaches (eigenfaces, VAE,
etc.)"""

from abc import ABC, abstractmethod
import configparser
import os

from scipy.misc import imread
import numpy as np

from fidentify.utils import path_from_root
from fidentify.preprocessing import apply_pipeline


class Person:
    """Data class used by Recogniser to store the identity of a person (i.e.
    their name, all known pictures of them, and descriptors for those
    pictures)."""
    def __init__(self, name):
        self.name = name
        self.pictures = []
        self.descriptors = []

    def add_photo(self, picture, descriptor):
        """Record a face photo and the associated face descriptor for this
        person."""
        self.pictures.append(picture)
        self.descriptors.append(descriptor)


class Recogniser(ABC):
    """Base class for all recognition methods (eigenfaces, VAE, etc.)."""
    def __init__(self, pipeline=None):
        if pipeline is None:
            self.pipeline = tuple()
        else:
            self.pipeline = tuple(pipeline)
        self.people = {}

    @abstractmethod
    def preproc_face_descriptor(self, preproc_image):
        """Make a face descriptor from a preprocessed image. Must be
        implemented by subclasses."""
        pass

    @abstractmethod
    def descriptor_similarity(self, desc1, desc2):
        """Return measure of visual similarity between two faces based on their
        descriptors. Again, must be implemented by subclasses."""
        pass

    @abstractmethod
    def reconstruct_descriptor(self, desc):
        """Turn a descriptor back into an image."""
        pass

    def _preprocess(self, image):
        """Apply preprocessing pipeline to an input image."""
        return apply_pipeline(self.pipeline, image)

    def face_descriptor(self, picture):
        """Calculate a face descriptor for an (unpreprocessed) image."""
        preproc = self._preprocess(picture)
        return self.preproc_face_descriptor(preproc)

    def add_picture(self, person_name, picture):
        """Add a picture of a person and the associated descriptor."""
        if person_name not in self.people:
            self.people[person_name] = Person(person_name)
        descriptor = self.face_descriptor(picture)
        self.people[person_name].add_photo(picture, descriptor)

    def match_people(self, picture, k=1):
        """Find name of k people most closely matching picture."""
        assert k > 0, "doesn't make sense to request %d people" % k
        similarities = {}
        descriptor = self.face_descriptor(picture)
        for person_key, person in self.people.items():
            for person_descriptor in person.descriptors:
                similarity = self.descriptor_similarity(descriptor,
                                                        person_descriptor)
                similarities[person.name] = similarity
        if len(similarities) == 0:
            raise ValueError('There are no descriptors in the database')
        sorted_people = sorted(similarities, key=similarities.get)
        chosen_people = sorted_people[:k]
        return [(person, similarities[person]) for person in chosen_people]

    def load_id_file(self, ids_path):
        """Enrol a whole collection of people from a specially-formatted .ini
        file (see the .inis in data/ for examples)."""
        config = configparser.ConfigParser()
        with open(ids_path, 'r') as fp:
            config.read_file(fp)
        cfg_dir = os.path.dirname(ids_path)
        for person_name in config.sections():
            photo_relpaths = config[person_name]['photos'].split(',')
            for photo_relpath in photo_relpaths:
                photo_abspath = path_from_root(cfg_dir, photo_relpath)
                im = imread(photo_abspath)
                self.add_picture(person_name, im)

    def get_pictures(self, person_name):
        """Get all stored photos of a given person"""
        if person_name not in self.people:
            return []
        person = self.people[person_name]
        return list(person.pictures)


class RandomMatcher(Recogniser):
    """Generates a bunch of random descriptors. Good as a baseline to make sure
    you're actually generating meaningful descriptors."""
    def preproc_face_descriptor(self, preproc_image):
        return np.random.randn()

    def descriptor_similarity(self, desc1, desc2):
        return (desc1 - desc2) ** 2

    def reconstruct_descriptor(self, desc):
        raise NotImplementedError()
