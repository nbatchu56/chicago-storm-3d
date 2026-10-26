import numpy as np

from app.radar_to_scene import biological_mask


def test_biological_mask_drops_weak_poorly_correlated_echo_only():
    dbz = np.ma.array([[8.0, 12.0, 30.0, 45.0, 10.0, 20.0]])
    rho = np.ma.array([[0.5, 0.95, 0.6, 0.6, 0.0, 0.99]], mask=[[0, 0, 0, 0, 1, 0]])
    m = biological_mask(dbz, rho)
    # birds/insects (weak, rho < 0.8) go; rain (rho 0.95) stays; strong cores (hail) stay whatever rho;
    # a gate with no correlation reading stays
    assert m.tolist() == [[True, False, True, False, False, False]]


def test_borrow_rhohv_fills_the_pass_without_dual_pol():
    from types import SimpleNamespace as NS
    from app.radar_to_scene import borrow_rhohv
    nan = np.nan
    rho = np.ma.array([[0.6, 0.6], [0.95, 0.95], [nan, nan], [nan, nan]], mask=[[0, 0], [0, 0], [1, 1], [1, 1]])
    radar = NS(fixed_angle={"data": np.array([0.5, 0.5])}, azimuth={"data": np.array([10.0, 190.0, 189.0, 11.0])},
               sweep_start_ray_index={"data": np.array([0, 2])}, sweep_end_ray_index={"data": np.array([1, 3])})
    out = borrow_rhohv(radar, rho)
    assert out[2].tolist() == [0.95, 0.95] and out[3].tolist() == [0.6, 0.6]
