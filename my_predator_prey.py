# %% [markdown]
# ## SINDy Predator-prey

# %%
import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from my_handy_functions import *

# %%
def time_plot(t, x, main_str):
    plt.plot(t, x, label=['x', 'y'])
    plt.xlabel('t')
    plt.ylabel('x, y')
    plt.legend()
    plt.title(main_str)
    plt.show()

def phase_plot(x, main_str):
    plt.plot(x[:,0], x[:,1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(main_str)
    plt.show()

def get_model():
    ode_lib = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=library_function_names
    )
    opt = ps.STLSQ(threshold=0.2)
    model = ps.SINDy(
        feature_library = ode_lib,
        optimizer = opt,
        feature_names=["x", "y"]
    )
    return(model, opt)

def comparison_plot(x, sim, main_str):
    plt.plot(x[0,0], x[0,1], "ro", label="Initial condition", alpha=0.6, markersize=8)
    plt.plot(x[:,0], x[:,1], "b", label="Exact solution", alpha=0.4, linewidth=1)
    plt.plot(sim[:, 0], sim[:, 1], "k--", label="SINDy model", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(main_str)
    plt.show()

def get_weak_model(t, num_steps, dt):
    ode_lib = ps.WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        spatiotemporal_grid=t,
        K=num_steps,
        H_xt=dt/2
    )

    opt = ps.SR3(threshold=0.05, thresholder="l1", max_iter=1000, normalize_columns=True, tol=1e-1)

    model = ps.SINDy(feature_library=ode_lib, feature_names=["x","y"], optimizer=opt)
    return(model, opt)

# %%
p=[1, 2, 3, 4] # Predator-prey parameters
t_start = 0
t_end = 90
num_steps = 1000

library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]

# ODE
def predator_prey(t, x, p=p):
    return [p[0] * x[0] - p[1] * x[0] * x[1], p[2] * x[0] * x[1] - p[3] * x[1]]

integrator_keywords = {'method':'LSODA', 'atol':1e-10, 'rtol':1e-12}

# %% [markdown]
# Clean data, normal SINDy

# %%
dt = (t_end-t_start)/num_steps
t = np.arange(t_start, t_end, dt)
t_span = (t_start, t_end)

x0 = [1, 2]
x = solve_ivp(predator_prey, t_span, x0, t_eval=t, **integrator_keywords).y.T

# %%
title = "Clean (noiseless) training data"
time_plot(t, x, title)
phase_plot(x, title)

# %%
model, opt = get_model()
model.fit(x, t=dt)
model.print()

# %%
x0_test = [1, 1.5]
t_test = t
t_span_test = t_span
x_test = solve_ivp(predator_prey, t_span_test, x0_test, t_eval=t_test, **integrator_keywords).y.T

sim = model.simulate(x0_test, t=t_test)
comparison_plot(x_test, sim, "Model prediction")

# %% [markdown]
# Noisy data, normal SINDy

# %%
sd = 0.02
x_noisy = lognormal_noisify(x, sd)

# %%
title = f'Noisy training data, sd = {sd}'
time_plot(t, x_noisy, title)
phase_plot(x_noisy, title)

# %%
model_noisy, opt_noisy = get_model()
model_noisy.fit(x_noisy, t=dt)
model_noisy.print()

# %%
sim_noisy = model_noisy.simulate(x0_test, t=t_test)
comparison_plot(x_test, sim_noisy, "Model prediction")

# %% [markdown]
# Clean data, weak SINDy

# %%
weak_model, weak_opt = get_weak_model(t, num_steps, dt)
weak_model.fit(x, t=dt)
print_result(weak_model, ["x", "y"])

# %%
opt.coef_ = weak_opt.coef_
weak_sim = model.simulate(x0_test, t=t_test)
comparison_plot(x_test, weak_sim, "Model prediction")

# %% [markdown]
# Noisy data, weak SINDy

# %%
weak_model_noisy, weak_opt_noisy = get_weak_model(t, num_steps, dt)
weak_model_noisy.fit(x_noisy, t=dt)
print_result(weak_model_noisy, ["x", "y"])

# %%
opt_noisy.coef_ = weak_opt_noisy.coef_
weak_sim_noisy = model_noisy.simulate(x0_test, t=t_test)
comparison_plot(x_test, weak_sim_noisy, "Model prediction")