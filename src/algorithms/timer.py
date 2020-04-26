import logging
from functools import wraps
from time import time

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

