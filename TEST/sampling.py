import torch

import itertools

list_a = list(range(0, 11))
list_b = list(range(100, 111))


linked = zip(list_a, list_b)

products = itertools.product(*linked)

