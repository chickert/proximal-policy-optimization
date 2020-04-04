import logging
from functools import wraps
from itertools import chain, product
from time import time
from typing import List, Iterable, Any, Dict

import numpy as np

# Set up logger
logger = logging.getLogger(__name__)


# Function timer
def timer(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        logger.info("Run time for %s: %.2fs", f.__name__, time() - start)
        return result
    return wrapper


# Nice pattern for concatenating lists
def concatenate_lists(lists: Iterable[List[Any]]) -> List[Any]:
    return list(chain(*lists))


def combine_grids(grids: List[Iterable[Any]], keys: Iterable[str]) -> List[Dict[str, Any]]:
    return [dict(zip(keys, perm)) for perm in product(*grids)]


def logit(x: float) -> float:
    return np.log(x / (1 - x))
