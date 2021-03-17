import numpy as np


def check_significance(data, threshd):
    """Checks which columns are significant based on the set threshold
    for numpy arrays and pandas dataframes. Deletes insignificant
    columns.

    Args:
        data (numpy array/pandas dataframe): The data object.
        threshd (float): The variance threshold below which data is considered as\
        constant and not included in the output.

    Returns:
        numpy array/pandas dataframe, integer list/string list:\
        The data object without insignificant\
        columns and the indices that correspond to the original columns that\
        remain, for plotting/labeling purposes.\
    purposes."""
    # find out type of data object
    if type(data) == np.ndarray:
        # determine which columns are important through the variance
        myvar = np.var(data, axis=1)
        indices = np.nonzero(myvar > threshd)
        data = data[indices]
    else:
        indices = data.var()[data.var() > threshd].index.values
        # indices = [data.columns.get_loc(i) for i in indices]
        data = data.drop(data.var()[data.var() < threshd].index.values, axis=1)
    return data, indices


def euclidean_distance(list_ref, list_comp, vectors):
    """Calculates the Euclidean distance (L2 norm) between pairs of vectors.

    Args:
        list_ref (integer list): A list with the indices of the reference vectors.
        list_comp (integer list): A list with the indices of the vectors to\
        compare to.
        data (numpy array): The data object.

    Returns:
        numpy array: The Euclidean distance (L2 norm) for comparison vs. reference\
        vectors."""
    distances = np.zeros(len(list_ref))
    for i in range(len(list_ref)):
        distances[i] = np.linalg.norm(vectors[list_comp[i]] - vectors[list_ref[i]])
    return distances


def correlation_matrix(data):
    """Calculates the correlation matrix of a dataframe and orders by
    degree of correlation.

    Args:
        data (pandas dataframe): The data object.

    Returns:
        pandas series: The correlation matrix sorted with highest\
        correlation on the top."""
    data = data.drop(["time"], axis=1)  # get rid of time column
    drop_values = set()
    cols = data.columns
    for i in range(0, data.shape[1]):
        for j in range(0, i + 1):
            # get rid of all diagonal entries and the lower triangular
            drop_values.add((cols[i], cols[j]))
    corrmat = data.corr().unstack()
    # sort by absolute values but keep sign
    corrmat = corrmat.drop(labels=drop_values).sort_values(ascending=False,
                                                           key=lambda col: col.abs())
    return corrmat
