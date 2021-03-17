import pytest
import numpy as np
import numerical as nl


class build_wf:
    def __init__(self):
        pass

    def init_carray(self, myvar, mydim, mydim2):
        arr = np.ones((mydim, mydim2), dtype=complex)
        return arr * myvar

    def init_cvector(self, myvar, mydim):
        vec = np.ones((mydim), dtype=complex)
        return vec * myvar

    def init_rarray(self, myvar, mydim, mydim2):
        arr = np.ones((mydim, mydim2), dtype=float)
        return arr * myvar


@pytest.fixture(scope='module')
def init_obj():
    # print('Calling setup')
    obj = build_wf()
    return obj
    # if you want to use a teardown method:
    # yield obj
    # print('Cleaning up')


@pytest.fixture()
def get_wf(init_obj):
    test_wf = init_obj.init_carray(1.0, 3, 3)
    test_auto = init_obj.init_cvector(3.0, 3)
    return test_wf, test_auto


@pytest.fixture()
def get_wfr(init_obj):
    return init_obj.init_rarray(1.0, 3, 3)


@pytest.fixture()
def struc(init_obj, request):
    marker = request.node.get_closest_marker("setval")
    if marker == 'None':
        # missing marker
        print('Missing a marker for integrity of wave function object')
    else:
        auto_val = marker.args[0]
        auto_dim = marker.args[1]
        set_zero = marker.kwargs.get('set_zero')
    test_struc = init_obj.init_rarray(1.0, 4, 3)
    tlist = [i * 0.1 for i in range(0, len(test_struc[0]))]
    test_struc[0] = test_struc[0] * tlist
    test_auto = init_obj.init_cvector(auto_val, auto_dim)
    if set_zero:
        test_auto[1:] = 0.0
    return test_struc, test_auto


# @pytest.mark.skip
def test_calc_auto(get_wf):
    """ Test function to test the calculation of the\
    wave function overlap."""
    # test the computation of values
    assert np.array_equal(nl.calc_auto(get_wf[0]), get_wf[1])


# test for type errors
def test_calc_auto_type(get_wfr):
    with pytest.raises(TypeError):
        assert nl.calc_auto(get_wfr)


# test reshuffling of data in aucofu
@pytest.mark.setval(4.0, 3, set_zero=False)
def test_aucofu(struc):
    time, aucofu = nl.aucofu(struc[0])
    assert np.array_equal(time, struc[0][0])
    assert np.array_equal(aucofu, struc[1])


# test DFT
@pytest.mark.setval(3.0, 2, set_zero=True)
def test_DFT_real(struc):
    data_w, data_s = nl.DFT(struc[0])
    assert np.array_equal(data_s, struc[1])


# test DFT
@pytest.mark.setval(3.0, 3, set_zero=True)
def test_DFT_comp(struc):
    data_w, data_s = nl.DFT(struc[0], False)
    assert np.array_equal(data_s, struc[1])
