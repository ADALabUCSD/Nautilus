# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from . import constants

class ParamType:
    ArchitectureTuning = 1
    HyperparameterTuning = 2


class _HP(object):
    def sample_value(self):
        """ randomly samples a value"""
        raise NotImplementedError()


class _HPChoice(_HP):
    def __init__(self, param_type, options):
        self.param_type = param_type
        self.options = options
        self.rand = np.random.RandomState(constants.RANDOM_SEED)

    def sample_value(self):
        return self.rand.choice(self.options, 1)[0]


def hp_choice(param_type, options):
    """ Categorical options.
    :param param_type: Parameter type.
    :param options: List of options.
    """

    if not type(options) == list:
        raise Exception('options has to be of type list.')

    return _HPChoice(param_type, options)


class _HPUniform(_HP):
    def __init__(self, param_type, min, max):
        if min >= max:
            raise Exception('min should be smaller than max')

        self.param_type = param_type
        self.min = min
        self.max = max

        self.rand = np.random.RandomState(constants.RANDOM_SEED)

    def sample_value(self):
        return self.rand.uniform(self.min, self.max, 1)[0]


def hp_uniform(param_type, min, max):
    """ Uniform distribution bounded by min and max.
    :param param_type: Parameter type
    :param min: Minimum value
    :param max: Maximum value
    """
    return _HPUniform(param_type, min, max)


class _HPQUniform(_HP):
    def __init__(self, param_type, min, max, q):
        if min >= max:
            raise Exception('min should be smaller than max')

        if q >= (max - min):
            raise Exception('q should be smaller than (max-min)')
        self.param_type = param_type
        self.min = min
        self.max = max
        self.q = q

        self.rand = np.random.RandomState(constants.RANDOM_SEED)

    def sample_value(self):
        t = round(self.rand.uniform(self.min, self.max, 1)[0] / self.q)
        return t * self.q


def hp_quniform(param_type, min, max, q):
    """ Quantized uniform distribution with a quantum of q, bounded by min and max. Returns a
     value like round(uniform(low, high) / q) * q.
    
    :param param_type: Parameter type
    :param min: Minimum value
    :param max: Maximum value
    :param q: Quantum
    """
    return _HPQUniform(param_type, min, max, q)


class _HPLogUniform(_HP):
    def __init__(self, param_type, min, max):
        if min >= max:
            raise Exception('min should be smaller than max')

        self.param_type = param_type
        self.min = min
        self.max = max
        self.rand = np.random.RandomState(constants.RANDOM_SEED)

    def sample_value(self):
        t = self.rand.uniform(self.min, self.max, 1)[0]
        return np.power(0.1, -t)


def hp_loguniform(param_type, min, max):
    """ Log uniform (base 10) distribution bounded by min and max.

    :param param_type: Parameter type
    :param min: Exponent of the minimum value in base 10 (e.g., -4 for 0.0001).
    :param max: Exponent of the maximum value in based 10.
    """
    return _HPLogUniform(param_type, min, max)


class _HPQLogUnifrom(_HP):
    def __init__(self, param_type, min, max, q):
        if min >= max:
            raise Exception('min should be smaller than max')

        if q >= (max - min):
            raise Exception('q should be smaller than (max-min)')

        self.param_type = param_type
        self.min = -min
        self.max = -max
        self.q = q

        self.rand = np.random.RandomState(constants.RANDOM_SEED)

    def sample_value(self):
        t = round(np.power(0.1, self.rand.uniform(
            self.min, self.max, 1)[0]) / self.q)
        return t * self.q


def hp_qloguniform(param_type, min, max, q):
    """ Quantized log uniform (base 10) distribution with a quantum of q, bounded by min and max. Returns a
     value like round(exp(uniform(low, high)) / q) * q.

    :param param_type: Parameter type
    :param min: Exponent of the minimum value in base 10 (e.g., -4 for 0.0001).
    :param max: Exponent of the maximum value in base 10.
    :param q:   Quantum
    """
    return _HPQLogUnifrom(param_type, min, max, q)
