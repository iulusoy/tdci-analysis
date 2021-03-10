#!/usr/bin/env python3
# package data_analysis
# brief Contains operations on the data and the data object.

# By ISU, 01/21
# This package handles the analysis of the measured data.
import numpy as np
import input_output as io
import statistical as sl
import numerical as nl
# import cProfile


def get_run_type(name):
    """Determines the run type of the data processing.

    Args:
        name (string): Determines a switch case based on the input file name.\
        Possible options are `expect.t`, `npop.t`, `table.dat`, `efield.t`,\
        `nstate_i.t`.

    Returns:
        function name: The name of the function to call for the selected\
        case."""
    mydict = {'expec.t': run_expec, 'npop.t': run_npop,
              'table.dat': run_table, 'efield.t': run_efield,
              'nstate_i.t': run_nstate}
    case = mydict.get(name, run_abort)
    return case


def run_expec(data, threshd, dir_out):
    """Handles the call to read and plot expect.t data;
    only plots relevant values (values that are not constant).

    Args:
        data (pandas dataframe): The data to process.
        threshd (float): The variance threshold below which data is considered\
        as constant and not included in the output.
        dir_out (string): Output file directory.
        """
    data, indices = sl.check_significance(data, threshd)
    # cProfile.runctx('sl.check_significance(data, threshd)', globals(),
    #                  locals())
    io.output_data(data, indices, dir_out, option='expecval')


def run_npop(data, threshd, dir_out):
    """Handles the call to read npop.t data; discards irrelevant
    columns (columns that remain constant); constructs the correlation
    matrix and prints/plots the result.

    Args:
        data (pandas dataframe): The data to process.
        threshd (float): The variance threshold below which data is considered\
        as constant and not included in the output.
        dir_out (string): Output file directory.
        """
    data, indices = sl.check_significance(data, threshd)
    corrmat = sl.correlation_matrix(data)
    io.output_data(data, indices, dir_out,
                   option='MOpop', data2=corrmat)


def run_table(data, threshd, dir_out):
    """Handles the call to read table.dat data;
    calculates Euclidean distance (L2 norm) of
    the vectors in the table.

    Args:
        data (numpy array): The data to process.
        threshd (float): The variance threshold below which data is considered\
        as constant and not included in the output.
        dir_out (string): Output file directory.
        """
    # no need for the first two columns, and replace the NaNs by zero
    data = np.delete(data, [0, 1], axis=0)
    data = np.nan_to_num(data)
    l2norm = sl.euclidean_distance([0, 2, 4], [1, 3, 5], data)
    indices = []
    io.output_data(l2norm, indices, dir_out, option='transdipmom')


def run_efield(data, threshd, dir_out):
    """Handles the call to read efield.t data; Fourier-transforms
    relevant values (values that are not constant) and plots the
    resulting spectrum.

    Args:
        data (numpy array): The data to process.
        threshd (float): The variance threshold below which data is considered\
        as constant and not included in the output.
        dir_out (string): Output file directory."""
    data, indices = sl.check_significance(data, threshd)
    # todo: how to handle more than one significant column
    data_w, data_s = nl.DFT(data, realdft=True)
    io.output_data(np.stack((data_w, data_s)), indices, dir_out,
                   option='efield')


def run_nstate(data, threshd, dir_out):
    """Handles the call to read nstate_i.t data; calculates, prints
    and plots the autocorrelation function; Fourier-transforms and
    plots the autocorrelation function.

    Args:
        data (numpy array): The data to process.
        threshd (float): The variance threshold below which data is considered\
        as constant and not included in the output.
        dir_out (string): Output file directory.
        """
    time, aucofu = nl.aucofu(data)
    data_w, data_s = nl.DFT(np.stack((time, aucofu)), realdft=False)
    indices = []
    io.output_data(np.stack((time, aucofu)), indices,
                   dir_out, option='aucofu', data2=np.stack((data_w, data_s)))


def run_abort(data, threshd):
    exit("Error: This type of analysis is not implemented")


def main(data_in, dir_in, dir_out, threshd=1.E-5):
    """Main function call if analysis package is to be run as a script.

    Args:
        data_in (string): Input file name.
        dir_in (string): Input file directory.
        dir_out (string): Output file directory.
        threshd (float): The variance threshold below which data is considered\
        as constant and not included in the output."""
    myobjin = io.input_data(data_in, dir_in)
    data = myobjin.read_in()
    runtype = get_run_type(data_in)
    runtype(data, threshd, dir_out)


if __name__ == "__main__":
    # main('expec.t', '../../data/', '../../output/')
    # main('expec.t', 'data/', 'output/')
    # main('efield.t', 'data/', 'output/')
    main('nstate_i.t', 'data/', 'output/')
    # main('table.dat', 'data/', 'output/')
    # main('npop.t', 'data/', 'output/')
