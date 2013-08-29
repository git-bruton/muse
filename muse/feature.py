#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib

import mmh3  # TODO: re-evaluate if this is fast enough, Probably want non-cryptographic, awesome if we let them decide
SEED = 0  # TODO: allow them to set the seed


def get_nested(d, path):
    for p in path:
        if p not in d:
            return None
        d = d[p]
    return d

class FeatureDict(object):
    """

    TODO: figure out how to handle namespaces
    TODO: how to handle name to index
    """

    def __init__(self):
        self.indices = dict()  # Dict[str, list[int]]
        self.values = dict()  # Dict[str, list[float]]
        self.__namespace = None

    def set_namespace(self, namespace):
        self.__namespace = namespace

    # if we're already putting a partial here, why not also do the dicts
    # because we need all these functions to add data
    def add_float(self, key, value):
        # TODO: How to enforce namespaces if it's optional
        # How I wrap the thing with a partial
        # NOOOO: TOO many function to wrap with partial, Make instance level attribute???
        # seems like bad design
        self.indices.setdefault(self._output_namespace, []).append(mmh3.hash(key, SEED))
        self.values.setdefault(self._output_namespace, []).append(value)

        # are namespaces defined in the config or the class
        # DECISION: they are defined in the config
        # this implies that they must be defined outside of stuff

    def add_string(self, key, value=1.0):
        self.indices.setdefault(self._output_namespace, []).append(mmh3.hash('{0}^{1}'.format(key, value), SEED))
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
        self._integer_fields = config.get('integer', [])
        self._vector_fields = config.get('vector', [])

    @staticmethod
    def validate_config(cls, config):
        valid_keys = set(['string', 'integer', 'vector'])
        assert set(config.keys()).issubset(valid_keys)
        for fields in config.itervalues():
            assert type(fields) is list

    def get_features(self, data, feature_dict):
        for in_field, out_field in self._string_fields:
            feature_dict.add_string(out_field, get_nested())

        for in_field, out_field in self._float_fields:
            feature_dict.add_float(out_field, get_nested())

        for in_field, out_field in self._vector_fields:
            feature_dict.add_vector(out_field, get_nested())

# STANDARDIZE ORDER OF CONFIG AND DATA

class Step(object):

    def __init__(self, step):
        self.step = step
        cls = importlib.import_module(step['class'])
        self._extractor = cls(step.get('configuration'))
        self._output_namespace = step.get('output_namespace')


class FeatureVectorExtractor(object):

    def __init__(self, config):
        self.__config = config
        self.__steps = [Step(step) for step in config['steps']]

    def get_features(self, lines, shared_data=None):
        for line in lines:
            input_data = {
                'shared': shared_data,
                'example': line,
            }
            feature_dict = FeatureDict()
            for step in self.__steps:
                feature_dict.set_namespace(step._output_namespace)
                step.extract(input_data, feature_dict)



    @staticmethod
    def validate_config(cls, config):
        raise NotImplemented()




# TODO: figure this out later
config = {
    'steps': [{
        'name': 'rebounding',
        'output_namespace': 'bball',
        'class': 'basketball.features.rebounding.Rebound',
        'configuration': {  # Make this a hashable dict
        },
    }, {
    }]
}



if __name__ == '__main__':
    print('hi')
