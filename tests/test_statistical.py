import pytest
import pandas as pd
import numpy as np
import statistical as sl


class build_df:
    def __init__(self, name):
        self.dfnamepath = name

    def read_df(self):
        self.data = pd.read_csv(self.dfnamepath, r'\s+')
        return self.data


@pytest.fixture(scope='module')
def get_df():
    obj = build_df('./tests/test_df.csv')
    data = obj.read_df()
    return data


@pytest.fixture()
def set_vectors():
    list_ref = [0, 2, 4]
    list_comp = [1, 3, 5]
    vectors = np.array([[-0.2, -1.0, 0.0],
                        [-0.3, -0.9, 0.0],
                        [1.0, 1.1, 0.0],
                        [0.8, 0.9, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.2]])
    dist_ref = np.array([0.14142136, 0.28284271, 0.2])
    return list_ref, list_comp, vectors, dist_ref


def test_check_significance_df(get_df):
    data, index = sl.check_significance(get_df, threshd=1.E-5)
    df_new = get_df.drop(columns=["norm", "<x>", "<y>"])
    index_new = df_new.columns.values
    assert len(index) == len(index_new)
    assert np.array_equal(index, index_new)
    assert data.equals(df_new)


def test_check_significance_array(get_df):
    data, index = sl.check_significance(get_df, threshd=1.E-5)
    df_new = get_df.drop(columns=["norm", "<x>", "<y>"])
    index_new = df_new.columns.values
    def_new = df_new.to_numpy()
    assert len(index) == len(index_new)
    assert np.array_equal(index, index_new)
    assert np.array_equal(data, def_new)


def test_euclidean_distance(set_vectors):
    dist = sl.euclidean_distance(set_vectors[0], set_vectors[1], set_vectors[2])
    assert dist.shape == set_vectors[3].shape
    assert np.allclose(dist, set_vectors[3], atol=1e-08)


def test_correlation_matrix(get_df):
    corrmat = sl.correlation_matrix(get_df)
    corrmat = corrmat.dropna().round(decimals=6)
    d = [0.832628, 0.228204, 0.136257]
    ind = pd.MultiIndex.from_tuples([
                                   ('norm', '<H>'),
                                   ('<z>', '<H>'),
                                   ('norm', '<z>')])
    ser_new = pd.Series(data=d, index=ind)
    assert corrmat.equals(ser_new)
