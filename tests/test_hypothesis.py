import numpy as np
import pytest

from hypothesis import given, strategies as st

from pycontest import simulation as sim2d
from pycontest.utils import momentum, E_kin

@given(mass1  = st.floats(min_value=.1, max_value=1e3),
       mass2  = st.floats(min_value=.1, max_value=1e3))
def test_energy_hypothesis(mass1, mass2):
    
    domain = ([-2, 12], [0, 3])
    t_max = 6
    dt = 0.5
    loc_0 = np.array([[0, 1.5],[10, 1.5]])
    vel_0 = np.array([[1, 0], [-1, 0]])
    radius = 1
    mass = [mass1, mass2]

    loc, vel = sim2d.simulation(t_max, dt, mass, radius, loc_0, vel_0, domain)
    
    E_ini = E_kin(vel_0, mass)
    E_end = E_kin(vel, mass)
    
    assert E_ini == E_end
