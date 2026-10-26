import numpy as np

from app.radar_to_scene import biological_mask


def test_biological_mask_drops_weak_poorly_correlated_echo_only():
    dbz = np.ma.array([[8.0, 12.0, 30.0, 45.0, 10.0, 20.0]])
    rho = np.ma.array([[0.5, 0.95, 0.6, 0.6, 0.0, 0.99]], mask=[[0, 0, 0, 0, 1, 0]])
    m = biological_mask(dbz, rho)
    # birds/insects (weak, rho < 0.8) go; rain (rho 0.95) stays; strong cores (hail) stay whatever rho;
    # a gate with no correlation reading stays
    assert m.tolist() == [[True, False, True, False, False, False]]
