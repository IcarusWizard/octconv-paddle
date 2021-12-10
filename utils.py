import random
import paddle
import numpy as np

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)