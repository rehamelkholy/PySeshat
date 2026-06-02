import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ==============================
# Global parameter scaling
# ==============================
SCALE = np.array([10, 2000, 1, 0.1, 180, 180, 90])

# ==============================
# Kepler solver (vectorized)
# ==============================
def kepler(M, e, tol=1e-10, max_iter=50):
    E = M.copy()
    for _ in range(max_iter):
        dE = (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < tol:
            break
    return E

# ==============================
# Thiele–Innes constants
# ==============================
def thiele_innes(a, i, Omega, omega):
    A = a*(np.cos(Omega)*np.cos(omega) -
           np.sin(Omega)*np.sin(omega)*np.cos(i))

    B = a*(np.sin(Omega)*np.cos(omega) +
           np.cos(Omega)*np.sin(omega)*np.cos(i))

    F = a*(-np.cos(Omega)*np.sin(omega) -
           np.sin(Omega)*np.cos(omega)*np.cos(i))

    G = a*(-np.sin(Omega)*np.sin(omega) +
           np.cos(Omega)*np.cos(omega)*np.cos(i))

    return A, B, F, G

# ==============================
# Core orbit model (Cartesian only)
# ==============================
def orbit_xy(t, params):

    P, T0, e, a, Omega, omega, i = params

    # convert to radians
    Omega = np.deg2rad(Omega)
    omega = np.deg2rad(omega)
    i = np.deg2rad(i)

    phase = ((t - T0) / P) % 1.0
    M = 2*np.pi*phase

    E = kepler(M, e)

    A, B, F, G = thiele_innes(a, i, Omega, omega)

    X = A*(np.cos(E) - e) + F*np.sin(E)
    Y = B*(np.cos(E) - e) + G*np.sin(E)

    return X, Y

# ==============================
# Residuals (Cartesian, scaled)
# ==============================
def residuals_xy(params_scaled, t, Xobs, Yobs, sx, sy):

    params = params_scaled * SCALE

    Xmod, Ymod = orbit_xy(t, params)

    resX = (Xobs - Xmod) / sx
    resY = (Yobs - Ymod) / sy

    return np.concatenate([resX, resY])

# ==============================
# Orbit fitting (main solver)
# ==============================
def fit_orbit_xy(t, X, Y, sx, sy, p0):

    p0_scaled = p0 / SCALE

    Tmin = np.min(t)
    Tmax = np.max(t)

    # allow some margin
    margin = 2 * (Tmax - Tmin)

    lower_T0 = Tmin - margin
    upper_T0 = Tmax + margin

    Tspan = 0.5 * margin

    P_min = max(0.1, 0.1 * Tspan)
    P_max = 10 * Tspan

    rho_max = np.max(np.sqrt(X**2 + Y**2))

    a_min = 0.1 * rho_max
    a_max = 5 * rho_max

    p0[4] = p0[4] % 360
    p0[5] = p0[5] % 360

    bounds = (
        np.array([P_min, lower_T0, 0.0, a_min, 0, 0, 0]) / SCALE,
        np.array([P_max, upper_T0, 0.99, a_max, 360, 360, 180]) / SCALE
    )

    result = least_squares(
        residuals_xy,
        p0_scaled,
        args=(t, X, Y, sx, sy),
        method='trf',
        loss='soft_l1',
        f_scale=1.0,
        bounds=bounds
    )

    return result.x * SCALE

# ==============================
# Data transformation
# ==============================
def polar_to_cartesian(data, sigma_theta_deg=0.5):

    t = data[:, 0]
    theta = np.deg2rad(data[:, 1])
    rho = data[:, 2]
    sigma = data[:, 3]

    X = rho * np.cos(theta)
    Y = rho * np.sin(theta)

    # --- error propagation ---
    # Assume sigma is rho uncertainty
    sigma_rho = sigma

    # Assume angular uncertainty ~ small constant (arcsec → radians)
    sigma_theta = np.deg2rad(sigma_theta_deg)  # ~0.5 deg default

    sx = np.sqrt(
        (np.cos(theta) * sigma_rho)**2 +
        (rho * np.sin(theta) * sigma_theta)**2
    )

    sy = np.sqrt(
        (np.sin(theta) * sigma_rho)**2 +
        (rho * np.cos(theta) * sigma_theta)**2
    )

    # numerical floor
    sx = np.maximum(sx, 1e-4)
    sy = np.maximum(sy, 1e-4)

    return t, X, Y, sx, sy

def parse_orbit_file(filename):

    with open(filename, 'r') as f:
        lines = f.readlines()

    elements = {}
    data = []

    for line in lines:

        line = line.strip()

        if not line or line.startswith('C'):
            continue

        tokens = line.split()

        # Orbital elements
        if tokens[0] in ['P','T','e','a','W','w','i']:
            elements[tokens[0]] = float(tokens[1])
            continue

        # Observations (I1 tag)
        if 'I1' in tokens:

            try:
                t = float(tokens[0])
                theta = float(tokens[1])
                rho = float(tokens[2])
                sigma = float(tokens[3])
            except:
                continue

            data.append((t, theta, rho, sigma))

    return elements, np.array(data)

def build_initial_guess(elements):

    return np.array([
        elements['P'],
        elements['T'],
        elements['e'],
        elements['a'],
        elements['W'],
        elements['w'],
        elements['i']
    ])

def compare_parameters(initial, fitted):

    names = ['P','T0','e','a','Omega','omega','i']

    print("\n=== PARAMETER COMPARISON ===\n")
    print(f"{'Param':<10}{'Initial':<15}{'Fitted':<15}{'Rel. Error (%)':<20}")

    for i, name in enumerate(names):

        p0 = initial[i]
        pf = fitted[i]

        rel_err = np.nan if p0 == 0 else abs((pf - p0)/p0)*100

        print(f"{name:<10}{p0:<15.6f}{pf:<15.6f}{rel_err:<20.6f}")

def compute_residuals(t, X_obs, Y_obs, sx, sy, params):

    X_model, Y_model = orbit_xy(t, params)

    resX = (X_obs - X_model) / sx
    resY = (Y_obs - Y_model) / sy

    return resX, resY, X_model, Y_model

def compute_statistics(resX, resY):

    chi2 = np.sum(resX**2 + resY**2)
    N = len(resX) * 2

    chi2_red = chi2 / N
    rms = np.sqrt(np.mean(np.concatenate([resX, resY])**2))

    print("\n=== FIT STATISTICS ===\n")
    print(f"Chi-square        = {chi2:.6f}")
    print(f"Reduced Chi-square= {chi2_red:.6f}")
    print(f"RMS (normalized)  = {rms:.6f}")

    return chi2, chi2_red, rms

def plot_orbit(t, X_obs, Y_obs, params, title="Orbit Fit"):

    # smooth model curve
    t_dense = np.linspace(min(t), max(t), 500)
    X_model, Y_model = orbit_xy(t_dense, params)

    plt.figure(figsize=(6,6))
    plt.scatter(X_obs, Y_obs, label="Observed")
    plt.plot(X_model, Y_model, label="Model")

    plt.gca().invert_xaxis()
    plt.xlabel("X (arcsec)")
    plt.ylabel("Y (arcsec)")
    plt.title(title)
    plt.legend()

    plt.show()

def plot_residuals(t, resX, resY):

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.scatter(t, resX)
    plt.axhline(0, linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Residual X / σ")
    plt.title("X Residuals")

    plt.subplot(1,2,2)
    plt.scatter(t, resY)
    plt.axhline(0, linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Residual Y / σ")
    plt.title("Y Residuals")

    plt.tight_layout()
    plt.show()

def validate_orbit(t, X, Y, sx, sy, params_fit, p0):

    compare_parameters(p0, params_fit)

    resX, resY, _, _ = compute_residuals(
        t, X, Y, sx, sy, params_fit
    )

    compute_statistics(resX, resY)

    plot_orbit(t, X, Y, params_fit)
    plot_residuals(t, resX, resY)