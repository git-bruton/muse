#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import simplejson as json
import hashlib
import math

from scipy.sparse import csr_matrix

#import mmh3  # TODO: re-evaluate if this is fast enough, Probably want non-cryptographic, awesome if we let them decide
SEED = 0  # TODO: allow them to set the seed

MAX_COLS = 9223372036854775807


def get_nested(_dict, full_path, default=None):
    for path in full_path.split('.'):
        if path not in _dict:
            return default
        _dict = _dict[path]
    return _dict

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


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
        self._float_values = dict()  # Dict[str, list[float]]
        self._string_values = dict()
        self._output_namespace = None

    def __repr__(self):
        output = {
            'float': self._float_values,
            'string': self._string_values,
            'output_namespace': self._output_namespace,
        }
        return json.dumps(output, sort_keys=True, cls=SetEncoder)

    def set_namespace(self, namespace):
        self._output_namespace = namespace

    def clear_namespace(self):
        self._output_namespace = None

    def add_float(self, key, value):
        # TODO: How to enforce namespaces if it's optional
        # How I wrap the thing with a partial
        # NOOOO: TOO many function to wrap with partial, Make instance level attribute???
        # seems like bad design
        if self._output_namespace not in self._float_values:
            self._float_values[self._output_namespace] = dict()
        self._float_values[self._output_namespace][key] = value
        #index = int(hashlib.sha1('{0}^{1}'.format(self._output_namespace, key)).hexdigest(), 16)
        #self.indices.setdefault(self._output_namespace, []).append(index)
        #self.values.setdefault(self._output_namespace, []).append(float(value))

        # are namespaces defined in the config or the class
        # DECISION: they are defined in the config
        # this implies that they must be defined outside of stuff

    def iter_float_namespace(self, namespace):
        for key, value in self._float_values[namespace].iteritems():
            yield key, value

    def add_strings(self, key, values):
        if self._output_namespace not in self._string_values:
            self._string_values[self._output_namespace] = dict()
        if key not in self._string_values[self._output_namespace]:
            self._string_values[self._output_namespace][key] = set()
        self._string_values[self._output_namespace][key] |= set(values)
#        index = int(hashlib.sha1('{0}^{1}^{2}'.format(self._output_namespace, key, value)).hexdigest(), 16)
#        self.indices.setdefault(self._output_namespace, []).append(index)
#        self.values.setdefault(self._output_namespace, []).append(1.0)

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
        indices = []
        data = []
        for namespace, key_values in self._float_values.iteritems():
            for key, value in key_values.iteritems():
                index = int(hashlib.sha1('{0}^{1}'.format(self._output_namespace, key)).hexdigest(), 16) % MAX_COLS
                data.append(value)
                indices.append(index)
        for namesapce, key_values in self._string_values.iteritems():
            for key, value in key_values.iteritems():
                index = int(hashlib.sha1('{0}^{1}^{2}'.format(self._output_namespace, key, value)).hexdigest(), 16) % MAX_COLS
                data.append(1.0)
                indices.append(index)
        return indices, data
        
    def to_np_array(self):
        raise NotImplementedError()
        data, indices = [], []
        for key in sorted(self.indices.keys()):
            data.extend(self.indices[key])
            indices.extend(self.values[key])
        return csr_matrix((data, indices, [0 ,len(indices)]), shape=(1, MAX_COLS), dtype=float)


class Transformer(object):

    # TODO: HOW OR WHY DO TRANSFORMS CARE ABOUT DATATYPES
    # DOES IT KNOW WHICH key to add too
    # TODO: ENFORCE ONLY ONE NAMESPACE PER FEATUREDICT (IS THIS NECESSARY????)
    def __init__(self, config=None):
        pass

    @staticmethod
    def new_index(existing_index, transform_name):
        return int(hashlib.sha1('{0}^{1}'.format(existing_index, transform_name)).hexdigest(), 16)

    def trasnsform(self, data):
        raise NotImplementedError()


class PolynomialFloatTransformer(Transformer):
    def __init__(self, config=None):
        ## VALIDATE CONFIG
        if not config:
            self._powers = [0.5, 2.0]
            self._input_namespace = None
        else:
            self._powers = config['powers']
            self._input_namespace = config.get('input_namespace')

    def transform(self, feature_dict):
        # TODO: DO TRANFORMS WRITE TO THE SAME NAMESPACE
        # THEY SHOULD BE SPECIFIED IN THE CONFIG
        new_values = dict()
        for key, value in feature_dict.iter_float_namespace(self._input_namespace):
            for power in self._powers:
                new_values['{0}^{1}'.format(key, power)] = math.pow(value, power)
        for key, value in new_values.iteritems():
            feature_dict.add_float(key, value)

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
        valid_keys = set(['string', 'float', 'vector'])
        assert set(config.keys()).issubset(valid_keys)
        for fields in config.itervalues():
            assert type(fields) is list

    def extract(self, data, feature_dict):
        for in_field, out_field, default in self._string_fields:
            feature_dict.add_strings(out_field, [get_nested(data, in_field, default)])

        for in_field, out_field, default in self._float_fields:
            feature_dict.add_float(out_field, get_nested(data, in_field, default))

        #for in_field, out_field in self._vector_fields:
        #    feature_dict.add_vector(out_field, get_nested(data, in_field))

# STANDARDIZE ORDER OF CONFIG AND DATA

class StepTypes(object):
    extractor = 'extractor'
    transformer = 'transformer'

class Step(object):

    def __init__(self, step):
        self.step = step
        cls = load_class(step['class'])
        self._actor = cls(step.get('configuration'))
        self._output_namespace = step.get('output_namespace')
        if hasattr(self._actor, 'extract'):
            self._type = StepTypes.extractor
        elif hasattr(self._actor, 'transform'):
            self._type = StepTypes.transformer
        else:
            raise ValueError('Wrong type: {0}'.format(type(self._actor)))



class FeatureVectorExtractor(object):

    def __init__(self, config):
        self._config = config
        self._steps = [Step(step) for step in config['steps']]

    def get_features(self, lines, shared_data=None):
        feature_dicts = []
        for line in lines:
            input_data = {
                'shared': shared_data,
                'instance': line,
            }
            fd = FeatureDict()
            feature_dicts.append(fd)
            for step in self._steps:
                if step._type == StepTypes.extractor:
                    fd.set_namespace(step._output_namespace)
                    step._actor.extract(input_data, fd)
                    fd.clear_namespace()
                elif step._type == StepTypes.transformer:
                    fd.set_namespace(step._output_namespace)
                    step._actor.transform(fd)
                    fd.clear_namespace()
                else:
                    raise ValueError('Wrong type')
        print feature_dicts[0]
        return self.to_matrix(feature_dicts)



    @staticmethod
    def validate_config(cls, config):
        valid_keys = set(['steps'])
        assert set(config.keys()).issubset(valid_keys())
        # ASSERT MORE HERE

    @staticmethod
    def to_matrix(feature_dicts):
        indices = []
        data = []
        indptr = [0] 
        for fd in feature_dicts:
            _indices, _data = fd.to_indices_data()
            indices.extend(_indices)
            data.extend(_data)
            indptr.append(len(indices))
        return csr_matrix((data, indices, indptr), shape=(len(feature_dicts), MAX_COLS), dtype=float)


# TODO: figure this out later
config1 = {
    'steps': [{
        'output_namespace': 'bookmark',
        'class': 'muse.feature.FieldExtractor',
        'configuration': {  # Make this a hashable dict
            'string': [['instance.url', 'url']],
        },
    }],
}


# RENAME SHARED TO CONTEXT LIKE AEROSOLVE
config2 = {
    'steps': [{
        'output_namespace': 'user',
        'class': 'muse.feature.FieldExtractor',
        'configuration': {
            'float': [
                ['instance.average_stars', 'average_stars', 0.0],
                ['instance.votes.cool', 'votes.cool', 0.0],
                ['instance.votes.funny', 'votes.funny', 0.0],
                ['instance.votes.useful', 'votes.useful', 0.0],
            ],
        },
    }, {
        'output_namespace': 'user',
        'class': 'muse.feature.PolynomialFloatTransformer',
        'configuration': {
            'input_namespace': 'user',
            'powers': [0.5, 2.0, 5.0],
        },
    }],
}


if __name__ == '__main__':
    lines = []
    # path = 'data/givealink_nov_2009/2009-11-01.json'
    path = 'data/users.json'
    with open(path) as f:
        for line in f:
            lines.append(json.loads(line))

    extractor = FeatureVectorExtractor(config2)
    results = extractor.get_features(lines)
