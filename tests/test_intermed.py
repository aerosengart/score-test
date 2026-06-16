import jax.numpy as jnp
import numpy as np
from stat_module import intermed


def test_quantities(simul_data, nb_model):
    data = simul_data('tests/data/nb_data.yaml')
    model = nb_model('tests/data/nb_model.yaml', data)

    n_targs = data.n_targs
    n_samps = data.n_samps
    exp_shape = (n_targs, n_samps)

    omega_vals     = intermed.compute_omega(model)
    delta_vals     = intermed.compute_delta(model, inv = False)
    delta_inv_vals = intermed.compute_delta(model, inv = True)
    epsilon_vals   = intermed.compute_epsilon(model)
    xi_vals        = intermed.compute_xi(model)
                                         
    delta_inv_omega_vals = delta_inv_vals * omega_vals
    
    c_vals = intermed.compute_c(model)
    r_vals = intermed.compute_r(model)
    a_vals = intermed.compute_a_j(model, j = 0)

    assert omega_vals.shape == exp_shape
    assert delta_vals.shape == exp_shape
    assert delta_inv_vals.shape == exp_shape
    assert epsilon_vals.shape == exp_shape
    assert xi_vals.shape == exp_shape
    assert c_vals.shape == exp_shape
    assert r_vals.shape == (n_targs, n_samps, n_samps)
    assert a_vals.shape == (n_samps, n_samps)

    assert not jnp.isnan(omega_vals).any()
    assert not jnp.isnan(delta_vals).any()
    assert not jnp.isnan(delta_inv_vals).any()
    assert not jnp.isnan(epsilon_vals).any()
    assert not jnp.isnan(xi_vals).any()
    assert not jnp.isnan(c_vals).any()
    assert not jnp.isnan(r_vals).any()
    assert not jnp.isnan(a_vals).any()

    print(model.compute_mu())

    assert jnp.allclose(omega_vals, model.compute_omega())
    p = np.argwhere(np.isnan(epsilon_vals))
    print(p)
    z = np.argwhere(np.isnan(model.compute_epsilon()))
    print(z)

    # print(epsilon_vals[p[0][0], :])

    # print(model.compute_mu()[p[0][0], :])
    # print(model.compute_omega()[p[0][0], :])
    # print(model.compute_delta_inv_omega()[p[0][0], :])
    # print(model.compute_resid()[p[0][0], :])
    # print(model.compute_epsilon()[p[0][0], :])

    # print(model.compute_xi()[p[0][0], :])

    assert jnp.allclose(epsilon_vals, model.compute_epsilon())
    assert jnp.allclose(xi_vals, model.compute_xi())
    assert jnp.allclose(delta_inv_omega_vals, model.compute_delta_inv_omega())
