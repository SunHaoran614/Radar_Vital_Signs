"""PCP's gain/phase symmetry and two analytically known decompositions."""
import numpy as np
import pytest

from src.radar.clutter import clutter_complex_rpca


def mixture():
    x = (np.exp(0.2j * np.arange(32))[:, None]
         * np.exp(0.7j * np.arange(6))[None, :])
    x[[3, 12, 25], [1, 4, 2]] += [4 + 2j, -3 + 4j, 5 - 3j]
    return x.reshape(32, 1, 2, 3).astype(np.complex64)


@pytest.mark.parametrize('gain', [1e-25, 1e-15, 1.0, 1e15, 1e25])
@pytest.mark.parametrize('rotation', [0.0, 0.73, 2.8])
def test_complex_gain_preserves_decomposition(gain, rotation):
    x = mixture()
    factor = gain * np.exp(1j * rotation)
    expected = clutter_complex_rpca(x)
    actual = clutter_complex_rpca((x * factor).astype(np.complex64))
    np.testing.assert_allclose(actual.astype(np.complex128) / factor,
                               expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('gain', [1e-25, 1.0, 1e25])
def test_isolated_entry_is_sparse_optimum(gain):
    # For a single entry, lambda=.5 < 1: sparse cost is less than nuclear cost.
    x = np.zeros((4, 1, 1, 4), dtype=np.complex64)
    x[1, 0, 0, 2] = gain * (2 + 3j)
    actual = clutter_complex_rpca(x)
    np.testing.assert_allclose(actual.astype(np.complex128) / gain,
                               x.astype(np.complex128) / gain, atol=2e-5)


@pytest.mark.parametrize('gain', [1e-25, 1.0, 1e25])
def test_dense_constant_is_low_rank_optimum(gain):
    # 4x4 all-ones: nuclear cost=4, sparse cost=.5*16=8.
    x = np.full((4, 1, 1, 4), gain * (1 + 1j), dtype=np.complex64)
    actual = clutter_complex_rpca(x)
    np.testing.assert_allclose(actual.astype(np.complex128) / gain, 0, atol=2e-5)


def test_zero_is_finite_zero():
    x = np.zeros((8, 1, 2, 3), dtype=np.complex64)
    np.testing.assert_array_equal(clutter_complex_rpca(x), x)
