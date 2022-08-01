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

from __future__ import absolute_import
import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from gurobipy import setParam

RANDOM_SEED = int(os.getenv('NAUTILUS_RANDOM_SEED', '2020'))

# Setting random seeds
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if 'GRB_LICENSE_FILE' not in os.environ:
    os.environ['GRB_LICENSE_FILE'] = os.path.join(Path.home(), 'gurobi.lic')
setParam("Seed", RANDOM_SEED)
setParam("NumericFocus", 3)

USE_MATERIALIZATION_OPTIMIZATION = 'use_materialization_optimization'

USE_MODEL_MERGE_OPTIMIZATION = 'use_model_merge_optimization'

VERY_LARGE_VALUE = float(os.getenv('NAUTILUS_VERY_LARGE_VALUE', '1e15'))
