#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import simplejson as json
import hashlib

import sys
from scipy.sparse import csr_matrix

#import mmh3  # TODO: re-evaluate if this is fast enough, Probably want non-cryptographic, awesome if we let them decide
SEED = 0  # TODO: allow them to set the seed



def get_nested(d, path):
    for p in path:
        if p not in d:
            return None
        d = d[p]
    return d



def load_class(full_class_string):
    """
    dynamically load a class from a string
    """

    module_path, class_name = full_class_string.rsplit(".", 1)
    module = importlib.import_module(module_path)
    # Finally, we retrieve the Class
    return getattr(module, class_name)

class FeatureDict(object):
    """

    TODO: figure out how to handle namespaces
    TODO: how to handle name to index
    """

    def __init__(self):
        self.indices = dict()  # Dict[str, list[int]]
        self.values = dict()  # Dict[str, list[float]]
        self._output_namespace = None

    def set_namespace(self, namespace):
        self._output_namespace = namespace

    # if we're already putting a partial here, why not also do the dicts
    # because we need all these functions to add data
    def add_float(self, key, value):
        # TODO: How to enforce namespaces if it's optional
        # How I wrap the thing with a partial
        # NOOOO: TOO many function to wrap with partial, Make instance level attribute???
        # seems like bad design
        index = int(hashlib.sha1('{0}^{1}'.format(key, value)).hexdigest(), 16)
        self.indices.setdefault(self._output_namespace, []).append(index)
        self.values.setdefault(self._output_namespace, []).append(float(value))

        # are namespaces defined in the config or the class
        # DECISION: they are defined in the config
        # this implies that they must be defined outside of stuff

    def add_string(self, key, value=1.0):
        index = int(hashlib.sha1(key).hexdigest(), 16)
        self.indices.setdefault(self._output_namespace, []).append(index)
        self.values.setdefault(self._output_namespace, []).append(1.0)

    def add_vector(self, key, values, namespace):
        # MUST HANDLE THIS FOR images
        # DON"T WANT COLLISIONS BUT WANT THEM SEQUENTIAL

        # human readable probably only want single start and end
        # THIS IS CRITIAL NOW!!!!

        # HOW TO HANDLE TRANSFORMS
        # HOW TO ASSIGN THE INDEX, awesome if order didn't matter like the hashing trick

#        self.indices.setdefault(namespace, []).append(mmh3.hash('{0}^{1}'.format(key, value), SEED))
#        self.values.setdefault(namespace, []).append(1.0)
        # what if define steps for vectors
        raise NotImplementedError('add_vector is not implemented')

    def to_indices_data(self):
        data, indices = [], []
        for key in sorted(self.indices.keys()):
            data.extend(self.indices[key])
            indices.extend(self.values[key])
        return indices, data
        

    def to_np_array(self):
        data, indices = [], []
        for key in sorted(self.indices.keys()):
            data.extend(self.indices[key])
            indices.extend(self.values[key])
        return csr_matrix((data, indices, [0 ,len(indices)]), shape=(1, sys.maxint), dtype=float)


class Transformer(object):

    def __init__(self, config=None):
        pass

    def trasnsform(self, data):
        raise NotImplementedError()


class FeatureFamilyExtractor(object):
    """

    TODO: static vs instance

    DECISION: have configure method
    DECISION: external data is hanlded by configuration
    """

    # TODO: figure out how this is implemented
    __slots__=(
        'dependencies',  # full import path
        'caching',  # None, implies no caching, initially use as boolean
        'inputs',
    )


    def __init__(self, config=None):
        """Must be optional"""
        pass

    # Make each family extractor static
    # Separate external data by itself
    def extract(cls, data, features):
        """
        input: data for computation, configuration, FeatureFamily # for a simplified interface
        type: dict FeatureFamily
        returns: 
        rtype: 
        """
        raise NotImplementedError()


class FieldExtractor(FeatureFamilyExtractor):
    """
    mapping
    """
    def __init__(self, config):
        self.validate_config(config)
        self._string_fields = config.get('string', [])
        self._float_fields = config.get('float', [])
        self._vector_fields = config.get('vector', [])

    @staticmethod
    def validate_config(config):
        valid_keys = set(['string', 'integer', 'vector'])
        assert set(config.keys()).issubset(valid_keys)
        for fields in config.itervalues():
            assert type(fields) is list

    def extract(self, data, feature_dict):
        for in_field, out_field in self._string_fields:
            feature_dict.add_string(out_field, get_nested(data, in_field))

        for in_field, out_field in self._float_fields:
            feature_dict.add_float(out_field, get_nested(data, in_field))

        for in_field, out_field in self._vector_fields:
            feature_dict.add_vector(out_field, get_nested(data, in_field))

# STANDARDIZE ORDER OF CONFIG AND DATA

class Step(object):

    def __init__(self, step):
        self.step = step
        cls = load_class(step['class'])
        self._extractor = cls(step.get('configuration'))
        self._output_namespace = step.get('output_namespace')


class FeatureVectorExtractor(object):

    def __init__(self, config):
        self._config = config
        self._steps = [Step(step) for step in config['steps']]

    def get_features(self, lines, shared_data=None):
        for line in lines:
            input_data = {
                'shared': shared_data,
                'instance': line,
            }
            feature_dict = FeatureDict()
            for step in self._steps:
                feature_dict.set_namespace(step._output_namespace)
                step._extractor.extract(input_data, feature_dict)
            print feature_dict.to_np_array().data



    @staticmethod
    def validate_config(cls, config):
        valid_keys = set(['steps'])
        assert set(config.keys()).issubset(valid_keys())
        # ASSERT MORE HERE

    def to_matrix(cls, feature_dicts):



# TODO: figure this out later
config = {
    'steps': [{
        'output_namespace': 'bookmark',
        'class': 'muse.feature.FieldExtractor',
        'configuration': {  # Make this a hashable dict
            'string': [['instance.url', 'url']],
        },
    }],
}



if __name__ == '__main__':
    lines = []
    with open('data/givealink_nov_2009/2009-11-01.json') as f:
        for line in f:
            lines.append(json.loads(line))

    extractor = FeatureVectorExtractor(config)
    results = extractor.get_features(lines)
    print results
