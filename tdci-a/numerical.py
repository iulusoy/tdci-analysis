import numpy as np


def aucofu(wavef):
    """Function to compute the autocorrelation function from the
    given vectors (with respect to the first time step).

    Args:
        wavef (numpy array, complex): The wave function\
        over time.

    Returns:
        numpy array, complex: The autocorrelation\
        function over time.

    """
    # store the time column in a vector and drop from array
    time = wavef[0]
    wavef = np.delete(wavef, [0], axis=0)
    # convert to complex array
    realpart = wavef[0::2]
    imagpart = wavef[1::2]
    wavefc = realpart + 1j * imagpart
    # Now construct overlap between first vector and all others
    aucofu = calc_auto(wavefc)
    return time, aucofu


def calc_auto(wavef):
    """Helper function to compute the vector overlap.

    Args:
        wavef (numpy array, complex): The wave function over time.

    Returns:
        numpy array, complex: The autocorrelation function over time."""
    aucofu = np.zeros(len(wavef[0]), dtype=complex)
    if type(wavef.item(0)) != complex:
        print('Found ', type(wavef.item(0)))
        raise TypeError('calc auto received wrong type of wavefunction data!')
    for i in range(0, len(wavef[0])):
        aucofu[i] = np.sum(np.conjugate(wavef[:, 0]) * wavef[:, i])
    return aucofu


def DFT(wavef, realdft=True):
    """Function to compute the discrete Fourier transform of the
    given function.

    Args:
        wavef (numpy array, real or complex): The data with time data in the\
        first row and the real or complex-valued vectors in the following rows.
        realdft (bool): Denotes if only positive frequency components\
        of the DFT are returned.

    Returns:
        numpy array, real; numpy array, complex: The energy grid points of the\
        Fourier-transformed function and the Fourier-transformed function.
    """
    # do the FT - see https://numpy.org/doc/stable/reference/routines.fft.html
    tmax = len(wavef[0])
    # array is a structured array - entries are arrays themselves
    # data_s = np.zeros((len(wavef)), dtype=np.ndarray)
    if(len(wavef) - 1) > 1:
        print("found more than 1D to FT")
        print("only the last dimension will be plotted")
    for i in range(1, len(wavef)):
        if(realdft):
            # only take the positive frequency components through rfft
            data_s = np.fft.rfft(wavef[i])
        else:
            # take all frequency components through fft
            data_s = np.fft.fft(wavef[i])
    if(realdft):
        data_w = np.fft.rfftfreq(tmax)
    else:
        data_w = np.fft.fftfreq(tmax)
    return data_w, data_s
