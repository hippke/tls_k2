import warnings
import numpy
from numba import jit


@jit(fastmath=True, parallel=False, nopython=True, cache=True)
def biweight_location_iter(data, c=6.0, ftol=1e-6, maxiter=15):
    """Robust location estimate using iterative Tukey's biweight"""

    delta_center = 1e100
    niter = 0

    # Initial estimate for the central location
    median_data = numpy.median(data)
    mad = numpy.median(numpy.abs(data - median_data))
    center = center_old = median_data

    if mad == 0:
        return center  # Neglecting this case was a bug in scikit-learn

    cmad = 1 / (c * mad)  # one expensive division here, instead of two per loop later
    while niter < maxiter and abs(delta_center) > ftol:  # New estimate for center
        distance = data - center
        weight = (1 - (distance * cmad) ** 2) ** 2  # inliers with Tukey's biweight
        weight[(numpy.abs(distance * cmad) >= 1)] = 0  # outliers as zero
        center += numpy.nansum(distance * weight) / numpy.nansum(weight)
        delta_center = center_old - center  # check convergence threshold
        center_old = center
        niter += 1
    return center


@jit(fastmath=True, parallel=False, cache=True)
def running_segment(t, y, window, c=6, ftol=1e-6, maxiter=15):
    mean_all = numpy.full(len(t), numpy.nan)

    # Move border checks out of loop (reason: large speed gain)
    half_window = window / 2
    lo = numpy.min(t) + half_window
    hi = numpy.max(t) - half_window
    idx_start = 0
    idx_end = 0

    #for i in prange(len(t)-1):  # Parallel numba. Speedup for large data is near linear
    for i in range(len(t)-1):
        if t[i] > lo and t[i] < hi:
            # Nice style would be:
            #   idx_start = numpy.argmax(t > t[i] - window/2)
            #   idx_end   = numpy.argmax(t > t[i] + window/2)
            # But that's too slow (factor 10). Instead, we write:
            while t[idx_start] < t[i] - half_window:
                idx_start += 1
            while t[idx_end] < t[i] + half_window:
                idx_end += 1

            # Get the Tukey-mean for the segment in question
            mean_all[i] = biweight_location_iter(y[idx_start:idx_end], c, ftol, maxiter)
    return mean_all


def get_gaps_indexes(time, gap_threshold):
    """Array indexes where (time) series is interrupted for longer than (window)"""
    gaps = numpy.diff(time)
    gaps_indexes = numpy.where(gaps > gap_threshold)
    gaps_indexes = numpy.add(gaps_indexes, 1) # Off by one :-)
    gaps_indexes = numpy.concatenate(gaps_indexes).ravel()  # Flatten
    gaps_indexes = numpy.append(numpy.array([0]), gaps_indexes)  # Start
    gaps_indexes = numpy.append(gaps_indexes, numpy.array([len(time)+1]))  # End point
    return gaps_indexes


def trend(t, y, window, c=6, ftol=1e-6, maxiter=15):

    # Check and warn if invalid values present
    invalid_ys = len(y) - numpy.count_nonzero(numpy.isfinite(y))
    invalid_ts = len(t) - numpy.count_nonzero(numpy.isfinite(t))
    if invalid_ys > 0 or invalid_ts > 0:
        warnings.warn("Invalid values encountered in data (NaN, None, and/or inf). " 
            "It is recommended to remove these before the detrending."
            )

    gaps_indexes = get_gaps_indexes(t, gap_threshold=window)
    trend = numpy.array([])
    trend_segment = numpy.array([])
    for i in range(len(gaps_indexes)-1):
        lo = gaps_indexes[i]
        hi = gaps_indexes[i+1]
        trend_segment = running_segment(t[lo:hi], y[lo:hi], window, c, ftol, maxiter)
        trend = numpy.append(trend, trend_segment)
    return trend
