import numpy as np
import scipy as sp


def cubic_spline_interpolation(t, q, t_eval=None):
    """Perform cubic spline interpolation on the given data points.

    t_eval, if given, evaluates the fitted spline (and its derivatives) at
    a different set of times than the knots it was fit on -- e.g. fitting
    on sparse waypoint times but evaluating on a dense dt-spaced grid.
    Defaults to t itself (fit and evaluate at the same knot times).
    """
    if t_eval is None:
        t_eval = t

    # Create a cubic spline interpolation function
    spline = sp.interpolate.CubicSpline(t, q, bc_type='not-a-knot')

    spline_d = spline.derivative()
    spline_dd = spline.derivative(nu=2)

    # Evaluate the spline at the requested x values
    q_i = spline(t_eval)
    q_d = spline_d(t_eval)
    q_dd = spline_dd(t_eval)

    return q_i, q_d, q_dd