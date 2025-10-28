import jax.numpy as jnp
from jax import jit, jacfwd
import numpy as np


class BSpline:

    def __init__(self, knotvector, degree):
        self.knotvector = knotvector
        self.degree = degree


class PeriodicBSpline(BSpline):
    def __init__(self, knotvector, degree):
        super().__init__(knotvector, degree)


class TensorProduct:
    def __init__(self, bsplines):
        self.ndim = len(bsplines)
        self.bsplines = bsplines
        self.degrees = tuple(bspline.degree for bspline in bsplines)
        self.knotvectors = [bspline.knotvector for bspline in bsplines]


class Surface(TensorProduct):
    def __init__(self, bsplines):
        super().__init__(bsplines)
        self.sh_fns = (
            len(self.bsplines[0].knotvector) - self.bsplines[0].degree - 1,
            len(self.bsplines[1].knotvector) - self.bsplines[1].degree - 1,
        )


class HeartValve(Surface):
    def __init__(self, bsplines):
        super().__init__(bsplines)
        self.sh_fns = (
            len(self.bsplines[0].knotvector) - self.bsplines[0].degree - 1,
            len(self.bsplines[1].knotvector) - self.bsplines[1].degree - 1,
        )
