"""Smoothers used in utils.processing smoothers"""

import tsmoothie.smoother

SMOOTHERS = {
    "kalman": tsmoothie.smoother.KalmanSmoother,
    "spline": tsmoothie.smoother.SplineSmoother,
    "binner": tsmoothie.smoother.BinnerSmoother,
    "lowess": tsmoothie.smoother.LowessSmoother,
    "convolution": tsmoothie.smoother.ConvolutionSmoother,
    "decompose": tsmoothie.smoother.DecomposeSmoother,
}
