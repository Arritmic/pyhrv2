# -*- coding: utf-8 -*-
"""
pyHRV - Time Domain Module
--------------------------

This module provides functions to compute HRV time domain  parameters using R-peak locations
and/or NN interval series extracted from an ECG lead I-like signal (e.g. ECG, SpO2 or BVP sensor data).

Notes
-----
..  Up to v.0.3 this work has been developed within the master thesis
	"Development of an Open-Source Python Toolbox for Heart Rate Variability (HRV)".
..	You find the API reference for this module here:
	https://pyhrv.readthedocs.io/en/latest/_pages/api/time.html
.. 	See 'references.txt' for a full detailed list of references

Author
------
..  Pedro Gomes, pgomes92@gmail.com

Contributors (and former Thesis Supervisors)
--------------------------------------------
..  Hugo Silva, PhD, Instituto de Telecomunicacoes & PLUX wireless biosignals S.A.
..  Prof. Dr. Petra Margaritoff, University of Applied Sciences Hamburg

Last Update
-----------
12-11-2019

:copyright: (c) 2019 by Pedro Gomes
:license: BSD 3-clause, see LICENSE for more details.

"""
# Compatibility
from __future__ import division, print_function

# Third party libraries
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# BioSppy imports
import biosppy
from biosppy.signals.ecg import ecg

# Local imports/pyHRV toolbox imports
import pyhrv


def nni_parameters(nni=None, rpeaks=None):
    """Computes basic statistical parameters from a series of NN intervals (# of intervals, mean, min, max).

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	nni_counter : int
		Number of NN intervals.
	nni_mean : float
		Mean NN interval [ms].
	nni_min : float
		Minimum NN interval [ms].
	nni_max : float
		Maximum NN interval [ms].

	Notes
	-----
	..	Only one type of input data is required.
	.. 	If both 'nni' and 'rpeaks' are provided, 'nni' will be chosen over the 'rpeaks'
	..	NN and R-peak series provided in [s] format will be converted to [ms] format.

	"""
    # Check input
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # output
    args = (int(nn.size), nn.mean(), nn.min(), nn.max())
    names = ('nni_counter', 'nni_mean', 'nni_min', 'nni_max')
    return biosppy.utils.ReturnTuple(args, names)


def nni_differences_parameters(nni=None, rpeaks=None):
    """Computes basic statistical parameters from a series of successive NN interval differences (mean, min, max, standard deviation).

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	nni_diff_mean: float
		Mean NN interval difference [ms].
	nni_diff_min : float
		Minimum NN interval difference [ms].
	nni_diff_max : float
		Maximum NN interval difference [ms].

	Notes
	-----
	..	Only one type of input data is required.
	.. 	If both 'nni' and 'rpeaks' are provided, 'nni' will be chosen over the 'rpeaks'
	..	NN and R-peak series provided in [s] format will be converted to [ms] format.

	"""
    # Check input
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # Get NN interval differences
    nnd = pyhrv.tools.nni_diff(nn)

    # output
    args = (float(nnd.mean()), int(nnd.min()), int(nnd.max()),)
    names = ('nni_diff_mean', 'nni_diff_min', 'nni_diff_max',)
    return biosppy.utils.ReturnTuple(args, names)


def hr_parameters(nni=None, rpeaks=None):
    """Computes basic statistical parameters from a series of Heart Rate (HR) data (mean, min, max, standard deviation).

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	hr_mean : float
		Mean heart rate [bpm].
	hr_min : float
		Minimum heart rate value [bpm].
	hr_max : float
		Maximum heart rate value [bpm].
	hr_std : float
		Standard deviation of the HR series [bpm].

	Notes
	-----
	..	Only one type of input data is required.
	.. 	If both 'nni' and 'rpeaks' are provided, 'nni' will be chosen over the 'rpeaks'
	..	NN and R-peak series provided in [s] format will be converted to [ms] format.

	"""
    # Check input
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # Get heart rate series
    hr = pyhrv.tools.heart_rate(nn)

    # Output
    args = (hr.mean(), hr.min(), hr.max(), hr.std(ddof=1))
    names = ('hr_mean', 'hr_min', 'hr_max', 'hr_std')
    return biosppy.utils.ReturnTuple(args, names)


def sdnn(nni=None, rpeaks=None):
    """Computation of the standard deviation of an NN interval series.

	References: [Electrophysiology1996]
	Docs:		https://pyhrv.readthedocs.io/en/latest/_pages/api/time.html#sdnn-sdnn

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	sdnn : float
		Standard deviation of NN intervals [ms].

	Notes
	-----
	..	Only one type of input data is required.
	.. 	If both 'nni' and 'rpeaks' are provided, 'nni' will be chosen over the 'rpeaks'
	..	NN and R-peak series provided in [s] format will be converted to [ms] format.

	"""
    # Check input
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # Computation of SDNN & Output
    args = [pyhrv.utils.std(nn)]
    names = ['sdnn']
    return biosppy.utils.ReturnTuple(args, names)


def sdnn_index(nni=None, rpeaks=None, full=True, duration=300, warn=True):
    """Computes the mean of the SDNN values of each segment (default: 300s segments).

	References: [Electrophysiology1996]
	Docs:		https://pyhrv.readthedocs.io/en/latest/_pages/api/time.html#sdnn-index-sdnn-index

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].
	full : bool, optional
		If True, returns last segment, even if the cumulative sum of NNI does not reach the 300s (default: False).
	duration : int, optional
		Maximum duration duration per segment in [s] (default: 300s).
	warn : bool, optional
		If True, raise a warning message if a segmentation could not be conducted (duration > NNI series duration)

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	sdnn_index : float
		Mean of the standard deviations of all NN intervals within 5 minutes intervals [ms]

	Notes
	-----
	..	Only one type of input data is required.
	.. 	If both 'nni' and 'rpeaks' are provided, 'nni' will be chosen over the 'rpeaks'
	..	NN and R-peak series provided in [s] format will be converted to [ms] format.

	"""
    # Check input
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # Signal segmentation into 5 min segments
    segments, seg = pyhrv.utils.segmentation(nn, full=full, duration=duration, warn=warn)

    if seg:
        sdnn_values = [sdnn(x)['sdnn'] for x in segments]
        sdnn_index = np.mean(sdnn_values)
    else:
        sdnn_index = float('nan')

    # Output
    args = [sdnn_index]
    names = ['sdnn_index']
    return biosppy.utils.ReturnTuple(args, names)


def sdann(nni=None, rpeaks=None, full=True, overlap=False, duration=300, warn=True):
    """Computes the standard deviation of the mean NNI value of each segment (default: 300s segments).

	References: [Electrophysiology1996], [Lohninger2017]
	Docs:		https://pyhrv.readthedocs.io/en/latest/_pages/api/time.html#sdann-sdann

	Parameters
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].
	full : bool, optional
		If True, returns last segment, even if the cumulative sum of NNI does not reach the 300s (default: False).
	duration : int, optional
		Maximum duration duration per segment in [s] (default: 300s).
	warn : bool, optional
		If True, raise a warning message if a segmentation could not be conducted (duration > NNI series duration)

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	sdnn_index : float
		Standard deviations of the means of all NN intervals within 5 minutes intervals in [ms].

	Notes
	-----
	..	Only one type of input data is required
	.. 	If both 'nni' and 'rpeaks' are provided, 'nni' will be chosen over the 'rpeaks'
	..	NN and R-peak series provided in [s] format will be converted to [ms] format

	"""
    # Check input
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # Signal segmentation into 5 min segments
    segments, seg = pyhrv.utils.segmentation(nn, full=full, duration=duration, warn=warn)

    if seg:
        mean_values = [np.mean(x) for x in segments]
        sdann_ = pyhrv.utils.std(mean_values)
    else:
        sdann_ = float('nan')
        warnings.warn("Signal duration too short for SDANN computation.")

    # Output
    args = [sdann_]
    names = ['sdann']
    return biosppy.utils.ReturnTuple(args, names)


def rmssd(nni=None, rpeaks=None):
    """Computes root mean of squared differences of successive NN Intervals.

	References: [Electrophysiology1996], [Lohninger2017]
	Docs:		https://pyhrv.readthedocs.io/en/latest/_pages/api/time.html#rmssd-rmssd

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	rmssd : float
		RMSSD value in [ms].

	Notes
	-----
	..	Only one type of input data is required
	.. 	If both 'nni' and 'rpeaks' are provided, 'nni' will be chosen over the 'rpeaks'
	..	NN and R-peak series provided in [s] format will be converted to [ms] format

	"""
    # Check input
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # Compute RMSSD
    nnd = pyhrv.tools.nni_diff(nn)
    rmssd_ = np.sum([x ** 2 for x in nnd])
    rmssd_ = np.sqrt(1. / nnd.size * rmssd_)

    # Output
    args = (rmssd_,)
    names = ('rmssd',)
    return biosppy.utils.ReturnTuple(args, names)


def sdsd(nni=None, rpeaks=None):
    """Computation of the standard deviation of differences of successive NN intervals.

	References: [Electrophysiology1996]
	Docs:		https://pyhrv.readthedocs.io/en/latest/_pages/api/time.html#sdsd-sdsd

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	sdsd : float
		Standard deviation of successive differences of NN intervals [ms]

	Notes
	-----
	..	Only one type of input data is required
	.. 	If both 'nni' and 'rpeaks' are provided, 'nni' will be chosen over the 'rpeaks'
	..	NN and R-peak series provided in [s] format will be converted to [ms] format

	"""
    # Check input
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # Compute NN differences
    nnd = pyhrv.tools.nni_diff(nn)

    # Computation of SDNN
    sdsd_ = pyhrv.utils.std(nnd)

    # Output
    args = [sdsd_]
    names = ['sdsd']
    return biosppy.utils.ReturnTuple(args, names)


def nnXX(nni=None, rpeaks=None, threshold=None):
    """Find number of NN interval differences greater than a specified threshold and ratio between number of intervals
	> threshold and total number of NN interval differences.

	References:	[Electrophysiology1996], [Ewing1984]
	Docs:		https://pyhrv.readthedocs.io/en/latest/_pages/api/time.html#nnxx-nnxx

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].
	threshold : int
		Threshold for nnXX values in [ms].

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	nnXX: int
		Number of NN interval differences greater than the specified threshold [-].
	pnnXX : float
		Ratio between nnXX and total number of NN interval differences [-].

	Notes
	-----
	..	Only one type of input data is required
	.. 	If both 'nni' and 'rpeaks' are provided, 'nni' will be chosen over the 'rpeaks'
	..	NN and R-peak series provided in [s] format will be converted to [ms] format
	..	The ``XX`` in the ``nnXX`` and the ``pnnXX`` keys are substituted by the specified threshold (``threshold``).

		For instance, ``nnXX(nni, threshold=30)`` returns the custom ``nn30`` and ``pnn30`` parameters. Using a
		``threshold=30`` as ``nnXX(nni, threshold=35`` returns the custom ``nn35`` and ``pnn35`` parameters.

	"""
    # Check input
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # Check threshold
    if threshold is None:
        raise TypeError("No threshold specified. Please specify a [ms] threshold.")
    if threshold <= 0:
        raise ValueError("Invalid value for 'threshold'. Value must not be <= 0.")

    # Count NN20
    nnd = pyhrv.tools.nni_diff(nn)
    nnxx = sum(i > threshold for i in nnd)
    pnnxx = nnxx / len(nnd) * 100

    # Output
    args = (nnxx, pnnxx)
    names = ('nn%i' % threshold, 'pnn%i' % threshold)
    return biosppy.utils.ReturnTuple(args, names)


def nn50(nni=None, rpeaks=None):
    """Find number of NN interval differences which are greater 50ms (NN50) and ratio between NN50 and total amount of
	NN intervals.

	References: [Electrophysiology1996], [Ewing1984]
	Docs:		https://pyhrv.readthedocs.io/en/latest/_pages/api/time.html#nn50-nn50

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	nn50 : int
		Number of NN interval differences greater 50ms.
	pnn50 : float
		Ratio between NN50 and total number of NN intervals.

	Raises
	------
	TypeError
		If no input data for 'rpeaks' or 'nni' provided.

	Notes
	-----
	..	Only one type of input data is required
	.. 	If both 'nni' and 'rpeaks' are provided, 'nni' will be chosen over the 'rpeaks'
	..	NN and R-peak series provided in [s] format will be converted to [ms] format

	"""
    return nnXX(nni=nni, rpeaks=rpeaks, threshold=50)


def nn20(nni=None, rpeaks=None):
    """Find number of NN interval differences which are greater 20ms (NN20) and ratio between NN20 and total amount of
	NN intervals.

	References: [Electrophysiology1996], [Hutchinson2003], [Mietus2002]
	Docs:		https://pyhrv.readthedocs.io/en/latest/_pages/api/time.html#nn20-nn20

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].

	Returns
	-------
	nn20 : int
		Number of NN interval differences greater 20ms.
	pnn20 : float
		Ratio between NN20 and total number of NN intervals.

	Raises
	------
	TypeError
		If no input data for 'rpeaks' or 'nn' provided.

	Notes
	-----
	..	Only one type of input data is required
	.. 	If both 'nni' and 'rpeaks' are provided, 'nni' will be chosen over the 'rpeaks'
	..	NN and R-peak series provided in [s] format will be converted to [ms] format
	"""
    return nnXX(nni=nni, rpeaks=rpeaks, threshold=20)


def tinn(nni=None, rpeaks=None, binsize=7.8125, plot=True, show=True, figsize=None, legend=True):
    """Computes TINN based on the NN intervals histogram.

    References: [Electrophysiology1996]
    Docs:       https://pyhrv.readthedocs.io/en/latest/_pages/api/time.html#tinn-tinn

    Parameters
    ----------
    nni : array
        NN intervals in [ms] or [s].
    rpeaks : array
        R-peak times in [ms] or [s].
    binsize : int, float
        Bin size of the histogram bins (default: 7.8125ms).
    plot : bool
        If True, creates histogram plot using matplotlib, else uses numpy (data only, no plot).
    show : bool, optional
        If true, shows histogram (default: True).
    figsize : array, optional
        Matplotlib figure size (width, height) (default: (6, 6)).
    legend : bool, optional
        If True, adds legend to the histogram (default: True).

    Returns (biosppy.utils.ReturnTuple)
    -----------------------------------
    tinn_histogram : matplotlib.figure.Figure
        Histogram figure (only if input parameter 'plot' is True).
    tinn_n : float
        N value of the TINN computation.
    tinn_m : float
        M value of the TINN computation.
    tinn : float
        TINN value.

    Raises
    ------
    TypeError (via 'check_input()')
        If no input data for 'rpeaks' or 'nni' is provided.

    Notes
    -----
    ..  Default bin size is set to the recommended bin size of 1/128 (with 128Hz being the minimum recommended sampling
        frequency) as recommended by the HRV guidelines.
    ..  'show' has only an effect if 'plot' is also True.
    ..  'legend' has only an effect if 'plot' is also True.
    ..  'figsize' has only an effect if 'plot' is also True.
    ..  If both 'nni' and 'rpeaks' are provided, 'rpeaks' will be chosen over the 'nni' and the 'nni' data will be computed
        from the 'rpeaks'.

    """
    # Check input
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # Get Histogram data (with or without histogram plot figure)
    if plot:
        fig, ax, D, bins = _get_histogram(nn, figsize=figsize, binsize=binsize, legend=legend, plot=plot)
    else:
        D, bins = _get_histogram(nn, figsize=figsize, binsize=binsize, legend=legend, plot=plot)

    # Identify the peak bin (bin with the maximum histogram value)
    peak_idx = np.argmax(D)
    peak_bin = bins[peak_idx]
    peak_value = D[peak_idx]

    # Define possible N and M bins
    n_bins = bins[bins < peak_bin]
    m_bins = bins[bins > peak_bin]

    # Initialize variables for minimal error calculation
    min_error = np.inf
    N_opt, M_opt = 0, 0

    # Precompute squared differences of D to avoid repeated operations
    D_squared = D ** 2

    # Loop over all possible N and M values
    for N in n_bins:
        for M in m_bins:
            if M <= N:
                # Skip invalid intervals
                continue

            # Calculate slopes for left and right side of the triangle
            left_slope = peak_value / (peak_bin - N)
            right_slope = peak_value / (M - peak_bin)

            # Initialize q using piecewise linear function
            q = np.zeros_like(D)

            # Left segment: where bins range from N to peak_bin
            left_mask = (bins >= N) & (bins <= peak_bin)
            q[left_mask] = left_slope * (bins[left_mask] - N)

            # Right segment: where bins range from peak_bin to M
            right_mask = (bins >= peak_bin) & (bins <= M)
            q[right_mask] = right_slope * (M - bins[right_mask])

            # Compute error (Sum of squared differences)
            # error = sum((D - q)**2)
            # Expanding: (D[i] - q[i])^2 = D[i]^2 - 2 * D[i] * q[i] + q[i]^2
            # We already have D[i]^2 from D_squared, q[i]^2 can be computed,
            # and 2 * D[i] * q[i] as well.
            error = np.sum(D_squared - 2 * D * q + q ** 2)

            # Update minimal error and corresponding N, M values
            if error < min_error:
                min_error = error
                N_opt, M_opt = N, M

    # Compute TINN
    tinn_value = M_opt - N_opt

    # If a plot is required, add the interpolated triangle and other specified plot characteristics
    if plot:
        # Add triangle lines to the histogram plot
        ax.plot([N_opt, peak_bin], [0, peak_value], 'r--', linewidth=1)
        ax.plot([peak_bin, M_opt], [peak_value, 0], 'r--', linewidth=1)

        # Add legend if required
        if legend:
            ax.legend([
                'Histogram $D(NNI)$',
                'Interpolated Triangle',
                f'$D(X)$: {peak_value:.2f}',
                f'Peak Bin: {peak_bin:.2f} ms',
                f'N: {N_opt:.2f} ms',
                f'M: {M_opt:.2f} ms',
                f'TINN: {tinn_value:.2f} ms'
            ], loc=0)

        # Show plot
        if show:
            plt.show()

        # Output results
        return biosppy.utils.ReturnTuple((fig, N_opt, M_opt, tinn_value), ('tinn_histogram', 'tinn_n', 'tinn_m', 'tinn'))
    else:
        # Return computed values without a plot
        return biosppy.utils.ReturnTuple((N_opt, M_opt, tinn_value), ('tinn_n', 'tinn_m', 'tinn'))


def triangular_index(nni=None, rpeaks=None, binsize=7.8125, plot=True, show=True, figsize=None, legend=True):
    """Computes triangular index based on the NN intervals histogram.

	References:	[Electrophysiology1996]
	Docs:		https://pyhrv.readthedocs.io/en/latest/_pages/api/time.html#triangular-index-triangular-index

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].
	binsize : int, float
		Bin size of the histogram bins (default: 7.8125ms).
	plot : bool
		If True, creates histogram plot using matplotlib, else uses numpy (data only, no plot).
	show : bool, optional
		If true, shows histogram (default: True).
	figsize : array, optional
		Matplotlib figure size (width, height) (default: (6, 6)).
	legend : bool, optional
		If True, adds legend to the histogram (default: True).

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	tri_histogram : matplotlib figure object
		Histogram figure (only if input parameter 'plot' is True).
	tri_index : float
		Triangular index.

	Raises
	------
	TypeError
		If no input data for 'rpeaks' or 'nni' provided.

	Notes
	-----
	..	Default bin size set to recommended bin size of 1/128 (with 128Hz being the minimum recommended sampling
		frequency) as recommended by the HRV guidelines.
	..	'show' has only effect if 'plot' is also True.
	.. 	'legend' has only effect if 'plot' is also True.
	..	'figsize' has only effect if 'plot' is also True.

	"""
    # Check input
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # If histogram should be plotted
    if plot:
        # Get histogram values
        fig, ax, D, bins = _get_histogram(nn, figsize=figsize, binsize=binsize, legend=legend, plot=plot)

        # Compute Triangular index: number of nn intervals / maximum value of the distribution
        tri_index = nn.size / D.max()

        # Add legend
        if legend:
            h = mpl.patches.Patch(facecolor='skyblue')
            x = mpl.patches.Patch(facecolor='g', alpha=0.0)
            dx = mpl.patches.Patch(facecolor='g', alpha=0.0)
            tri = mpl.patches.Patch(facecolor='white', alpha=0.0)
            ax.legend(
                [h, x, dx, tri],
                ['Histogram D(NNI)', 'D(X): %i' % D.max(), 'X: %.3f' % bins[np.argmax(D)],
                 'TriIndex: %.3f' % tri_index],
                loc=0
            )

        # Show plot
        if show:
            plt.show()

        # Output
        args = (fig, tri_index,)
        names = ('tri_histogram', 'tri_index',)

    # If histogram should not be plotted
    else:
        D, bins = _get_histogram(nn, figsize=figsize, binsize=binsize, legend=legend, plot=plot)

        # Compute Triangular index: number of nn intervals / maximum value of the distribution
        tri_index = nn.size / D.max()

        # Output
        args = (tri_index,)
        names = ('tri_index',)

    return biosppy.utils.ReturnTuple(args, names)


def _get_histogram(nn=None, plot=True, figsize=None, binsize=None, legend=True):
    """Prepares NNI histogram data for all geometrical functions.

	Parameters
	----------
	nn : array
		NN intervals in [ms] or [s].
	plot : bool
		If True, creates histogram plot using matplotlib, else uses numpy (data only, no plot).
	figsize : array, optional
		Matplotlib figure size (width, height) (default: (6, 6)).
	binsize : int, float
		Bin size of the histogram bins.
	legend : bool
		If True, highlights D(X) marker to the plot to be added to the legends (default=True).

	Returns
	-------
	fig : matplotlib figure object
		Figure of the histogram plot (only if input parameter 'plot' is True).
	vals : array
		Histogram distribution values.
	bins : array
		Histogram bins.

	Raises
	------
	TypeError
		If no input data provided for 'nn'.
	TypeError
		If no input data provided for 'binsize'.

	Notes
	-----
	..	'figsize' has only effect if 'plot' is also True.
	.. 	'legend' has only effect if 'plot' is also True.

	"""
    # Check input data & confirm numpy
    if nn is None:
        raise TypeError("No input data provided for 'nn'.")
    else:
        nn = np.asarray(nn)

    if binsize is None:
        raise TypeError("No input data provided for 'binsize'")

    # Create bins array
    bins = np.arange(0, np.max(nn) + binsize, binsize)

    # Get histogram plot and data
    if plot:
        # Check figsize
        if figsize is None:
            figsize = (6, 6)

        # Prepare plot figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        vals, bins, patches = ax.hist(nn, bins, density=False, align='left', facecolor='skyblue', edgecolor='black')
        bins = bins[:-1]

        # Highlight bin of the histograms maximum value with a different color and prepare legend
        if legend:
            ax.vlines(bins[np.argmax(vals)], 0, (vals.max() * 1.1),
                      linestyles='--', color='g', linewidth=0.6)
            pos = (bins[np.argmax(vals)], vals.max() * 1.11)
            ax.annotate('D(X)', xy=pos, xytext=pos, ha='center', color='g')

        # Configure figure and plot
        ax.axis([nn.min() - (3 * binsize), nn.max() + (3 * binsize), 0, vals.max() * 1.15])
        ax.set_xlabel('NNI Bins [ms]')
        ax.set_ylabel('D(NNI) [-]')
        ax.set_title('NNI Histogram')
        return fig, ax, vals, bins

    else:
        vals, bins = np.histogram(nn, bins, density=False)
        return vals, bins[:-1]


def geometrical_parameters(nni=None, rpeaks=None, binsize=7.815, plot=True, show=True, figsize=None, legend=True):
    """Creates NNI histogram with specified binsize (default: 7.815ms) and computes geometrical parameters (triangular
	index, TINN, N, and M).

	References:	[Electrophysiology1996]
	Docs:		https://pyhrv.readthedocs.io/en/latest/_pages/api/time.html#geometrical-parameters-function-geometrical-parameters


	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].
	binsize : int, float
		Bin size of the histogram bins (default: 7.8125ms).
	plot : bool
		If True, creates histogram plot using matplotlib, else uses numpy (data only, no plot).
	show : bool, optional
		If true, shows histogram (default: True).
	figsize : array, optional
		Matplotlib figure size (width, height) (default: (6, 6)).
	legend : bool, optional
		If True, adds legend to the histogram (default: True).

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	nni_histogram : matplotlib figure object
		Histogram figure (only if input parameter 'plot' is True).
	tri_index : float
		Triangular index.
	tinn_n : float
		N value of the TINN computation.
	tinn_m : float
		M value of the TINN computation.
	tinn : float
		TINN value.

	Raises
	------
	TypeError (via 'check_input()')
		If no input data for 'rpeaks' or 'nni' provided.

	Notes
	-----
	..	Default bin size set to recommended bin size of 1/128 (with 128Hz being the minimum recommended sampling
		frequency) as recommended by the HRV guidelines.
	..	'show' has only effect if 'plot' is also True.
	.. 	'legend' has only effect if 'plot' is also True.
	..	'figsize' has only effect if 'plot' is also True.

	"""

    # Check input
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # Get Histogram data & plot (optional)
    if plot:
        fig, ax, D, bins = _get_histogram(nn, figsize=figsize, binsize=binsize, legend=legend, plot=plot)
    else:
        fig = None

    # Get TINN values without plot figure
    tinn_vals = tinn(nni=nn, rpeaks=rpeaks, binsize=binsize, show=False, legend=False, figsize=figsize, plot=False)

    # Get triangular index without plot figure
    trindex = triangular_index(nni=nn, rpeaks=rpeaks, binsize=binsize, show=False, legend=False, plot=False)['tri_index']

    # Histogram plot & settings
    if plot:
        # Plot triangular interpolation
        N, M = tinn_vals['tinn_n'], tinn_vals['tinn_m']
        ax.plot([N, bins[np.argmax(D)]], [0, D.max()], 'r--', linewidth=0.8)
        ax.plot([bins[np.argmax(D)], M], [D.max(), 0], 'r--', linewidth=0.8)

        # Add Legend
        if legend:
            l1 = mpl.patches.Patch(facecolor='skyblue', label='Histogram D(NNI)')
            l2 = mpl.lines.Line2D([0, 0], [0, 0], linestyle='--', linewidth=0.8, color='r', label='Tri. Interpol.')
            l3 = mpl.patches.Patch(facecolor='g', alpha=0.0, label='D(X): %i' % D.max())
            l4 = mpl.patches.Patch(facecolor='g', alpha=0.0, label='X: %.3f$ms$' % bins[np.argmax(D)])
            l5 = mpl.patches.Patch(facecolor='white', alpha=0.0, label='N: %.3f$ms$' % tinn_vals['tinn_n'])
            l6 = mpl.patches.Patch(facecolor='white', alpha=0.0, label='M: %.3fms' % tinn_vals['tinn_m'])
            l7 = mpl.patches.Patch(facecolor='white', alpha=0.0, label='TINN: %.3fms' % tinn_vals['tinn'])
            l8 = mpl.patches.Patch(facecolor='white', alpha=0.0, label='Tri. Index: %.3f' % trindex)
            ax.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8], loc=0, ncol=1)

        # Show plot
        if show:
            plt.show()

    # Output
    args = (fig, tinn_vals['tinn_n'], tinn_vals['tinn_m'], tinn_vals['tinn'], trindex)
    names = ('nni_histogram', 'tinn_n', 'tinn_m', 'tinn', 'tri_index')
    return biosppy.utils.ReturnTuple(args, names)


def time_domain(nni=None,
                rpeaks=None,
                signal=None,
                sampling_rate=1000.,
                threshold=None,
                plot=True,
                show=False,
                binsize=7.8125):
    """Computes all time domain parameters of the HRV time domain module and returns them in a ReturnTuple object.

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].
	signal : array
		ECG signal.
	sampling_rate : int, float, optional
		Sampling rate used for the ECG acquisition in [Hz] (default: 1000.).
	threshold : int, optional
		Custom threshold in [ms] for the NNXX and pNNXX parameters (default: None).
	plot : bool
		If True, creates histogram plot using matplotlib, else uses numpy (data only, no plot) - (geometrical params).
	figsize : array, optional
		Matplotlib figure size for the histogram (width, height) (default: (6, 6)) - (geometrical params).
	binsize : int, float
		Bin size in [ms] of the histogram bins - (geometrical params).
	legend : bool
		If True, highlights D(X) marker to the plot to be added to the legends (default=True) - (geometrical params).

	Returns
	-------
	results : biosppy.utils.ReturnTuple object
		All time domain results (see list and keys below)

	Returned Parameters
	-------------------
	..	NNI parameters (# of NNI, mean, min, max) in [count] and [ms] (keys: 'nni_counter', 'nni_mean', 'nni_min',
		'nni_max')
	..	NNI differences (mean, min, max, standard deviation) in [ms] (keys: 'nni_diff_mean', 'nni_diff_min',
		'nn_diff_max')
	..	HR parameters (mean, min, max, standard deviation) in [BPM] (keys: 'hr_mean', 'hr_min', 'hr_max', 'hr_std')
	..	SDNN in [ms] (key: 'sdnn')
	..	SDNN index in [ms] (key: 'sdnn_index')
	..	SDANN in [ms] (key: 'sdann')
	..	RMSSD in [ms] (key: 'rmssd')
	..	SDSD in [ms] (key: 'sdsd')
	..	nn50 in [count] & pNN50 in [%] (keys: 'nn50', 'pnn50')
	..	nn20 in [count] & pNN20 in [%] (keys: 'nn20', 'pnn20')
	..	nnXX (XX = custom threshold) if specified (keys: 'nnXX', 'pnnXX')
	..	Triangular Index [-] (key: 'tri_index')
	.. 	TINN in [ms] (key: 'tinn', 'tinn_n', 'tinn_m')
	..	NNI histogram (key: 'nni_histogram')

	Notes
	-----
	..	Results are stored in a biosppy.utils.ReturnTuple object and need to be accessed with the respective keys as
		done with dictionaries (see list of parameters and keys above).
	..	Only one type of input data is required (signal, nni, or rpeaks).
	..	Input data will be prioritized in the following order: 1. signal, 2. nni, 3. rpeaks.
	..	SDNN Index and SDANN: In some cases, the NN interval may start in a segment (or
	..	Default bin size set to recommended bin size of 1/128 (with 128Hz being the minimum recommended sampling
		frequency) as recommended by the HRV guidelines.
	..	'show' has only effect if 'plot' is also True.
	.. 	'legend' has only effect if 'plot' is also True.
	..	'figsize' has only effect if 'plot' is also True.

	Raises
	------
	TypeError
		If no input data for 'nni', 'rpeaks', and 'signal' provided.

	"""
    # Check input
    if signal is not None:
        t, signal, rpeaks = biosppy.ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)[:3]
        rpeaks = t[rpeaks]
    elif nni is None and rpeaks is None:
        raise TypeError('No input data provided. Please specify input data.')

    # Get NNI series
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # Call time domain functions & wrap results in a single biosppy.utils.ReturnTuple object
    results = nni_parameters(nn)
    results = pyhrv.utils.join_tuples(results, hr_parameters(nn))
    results = pyhrv.utils.join_tuples(results, nni_differences_parameters(nn))
    results = pyhrv.utils.join_tuples(results, sdnn(nn))
    results = pyhrv.utils.join_tuples(results, sdnn_index(nn))
    results = pyhrv.utils.join_tuples(results, sdann(nn))
    results = pyhrv.utils.join_tuples(results, rmssd(nn))
    results = pyhrv.utils.join_tuples(results, sdsd(nn))
    results = pyhrv.utils.join_tuples(results, nn50(nn))
    results = pyhrv.utils.join_tuples(results, nn20(nn))

    # Compute custom threshold if required
    if threshold is not None and threshold not in [50, 20]:
        results = pyhrv.utils.join_tuples(results, nnXX(nn, threshold=int(threshold)))

    # Compute geometrical parameters
    results = pyhrv.utils.join_tuples(results, geometrical_parameters(nn, plot=plot, show=show, binsize=binsize))

    # Output
    return results


if __name__ == "__main__":
    """
	Example Script - HRV Time Domain Analysis
	"""
    # Load sample NNI series
    nni = pyhrv.utils.load_sample_nni(series='long')

    # Time Domain results
    print("=========================")
    print("TIME DOMAIN Results")
    print("=========================")

    hr_ = hr_parameters(nni)
    print("HR Results")
    print("> Mean HR:			%f [bpm]" % hr_['hr_mean'])
    print("> Min HR:			%f [bpm]" % hr_['hr_min'])
    print("> Max HR:			%f [bpm]" % hr_['hr_max'])
    print("> Std. Dev. HR:		%f [bpm]" % hr_['hr_std'])

    nni_para_ = nni_parameters(nni)
    print("NN Results")
    print("> Mean NN:			%f [ms]" % nni_para_['nni_mean'])
    print("> Min NN:			%f [ms]" % nni_para_['nni_min'])
    print("> Max NN:			%f [ms]" % nni_para_['nni_max'])

    nni_diff_ = nni_differences_parameters(nni)
    print("∆NN Results")
    print("> Mean ∆NN:			%f [ms]" % nni_diff_['nni_diff_mean'])
    print("> Min ∆NN:			%f [ms]" % nni_diff_['nni_diff_min'])
    print("> Max ∆NN:			%f [ms]" % nni_diff_['nni_diff_max'])

    print("SDNN:				%f [ms]" % sdnn(nni)['sdnn'])
    print("SDNN Index:			%f [ms]" % sdnn_index(nni)['sdnn_index'])
    print("SDANN:				%f [ms]" % sdann(nni)['sdann'])
    print("RMMSD:				%f [ms]" % rmssd(nni)['rmssd'])
    print("SDSD:				%f [ms]" % sdsd(nni)['sdsd'])
    print("NN50:				%i [-]" % nn50(nni)['nn50'])
    print("pNN50: 				%f [%%]" % nn50(nni)['pnn50'])
    print("NN20:				%i [-]" % nn20(nni)['nn20'])
    print("pNN20: 				%f [%%]" % nn20(nni)['pnn20'])

    # Compute geometrical parameters (without plot)
    print("=== Geometrical Parameters")
    geo = geometrical_parameters(nni, plot=True, show=True)
    print("Triangular Index: 	%f [-]" % geo['tri_index'])
    print("TINN:				%f [ms]" % geo['tinn'])
    print("> N:				%f [ms]" % geo['tinn_n'])
    print("> M:				%f [ms]" % geo['tinn_m'])

    # Alternatively use the individual geometrical parameter functions
    triangular_index(nni, plot=False)
    tinn(nni, plot=False)

    # Alternatively use the time_domain() function to compute all time domain parameters using a single function
    time_domain(nni=nni)
