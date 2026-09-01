import numpy as np
import pandas as pd

def Q_R_prime_calc(R_t = None,
            R_i = None,
            M_t = None,
            M_i = None,
            b = None,
            v_c = None
            ):
    
    """
    Calculate QR'.

    Parameters:
    R_t : float
        Target planet radius (in m).
    R_i : float
        Impactor planet radius (in m).
    M_t : float
        Target planet mass, including atmosphere (in kg).
    M_i : float
        Impactor planet mass (in kg).
    b : float
        Impact parameter.
    v_c : float
        Impact velocity (in m/s).
    

    Returns:
    Q_R_prime : float
        Centre of mass modified specific impact energy (in J/kg. Convert to MJ/kg for use in the X_atm_calc function).

        
    --------------------------------------
    M.J.Roche       16/01/26        
    --------------------------------------
    """
    
    B = (R_t + R_i) * b

    condition = B + R_i <= R_t
    l = np.where(condition, 2 * R_i, R_t + R_i - B) # Projected length of the projectile overlapping the target
    alpha = np.where(condition, 1.0, (3 * R_i * l**2 - l**3) / (4 * R_i**3))
    
    M_tot = M_t + M_i # Total system mass
    mu = (M_t * M_i) / M_tot # Reduced mass
    mu_alpha = (alpha * M_t * M_i) / ((alpha * M_i) + M_t) # Reduced mass for the overlapping projectile
    Q_R = (mu * v_c**2) / (2 * M_tot) # Unmodified centre of mass specific impact energy
    Q_R_prime = (mu_alpha * Q_R) / mu

    return Q_R_prime


# Load fitting parameters:
fit_params_NF_mass = pd.read_csv('fit_params_NF_mass.txt', sep = r'\s+', index_col = 0)
fit_params_NF = pd.read_csv('fit_params_NF_loss.txt', sep = r'\s+', index_col = 0)
fit_params_FF = pd.read_csv('fit_params_FF_loss.txt', sep = r'\s+', index_col = 0)  


def X_atm_calc(var):

    """
    Calculate near-, far-, and total atmospheric loss fractions (X_NF, X_FF, X_atm).

    Parameters:
    b : float
        Impact parameter.
    gamma : float
        Impactor–total (refractory) system mass ratio.
    vc : float
        Impact velocity (in multiples of the mutual escape velocity).
    M_t : float
        Refractory (excluding atmosphere) target planet mass (in Earth masses).
    M_i : float
        Refractory impactor planet mass (in Earth masses).
    Q_R_prime : float
        Centre of mass modified specific impact energy (in MJ kg^-1).
    f_atm : float
        Target planet atmospheric mass fraction.
    R_ratio : float
        Impactor planet radius to target planet radius ratio.

    If passing an array of values, pass as a numpy array.

    Returns:
    f_NF : float
        Near-field atmosphere mass fraction.
    f_FF : float
        Far-field atmosphere mass fraction.
    X_NF : float
        Near-field atmospheric loss fraction.
    X_FF : float
        Far-field atmospheric loss fraction.
    X_atm : float
        Total atmospheric loss fraction.

    
    --------------------------------------
    M.J.Roche       01/09/26        
    --------------------------------------
    """

    b = var[0]
    gamma = var[1]
    vc = var[2]
    M_t = var[3]
    M_i = var[4]
    Q_R_prime = var[5]
    f_atm = var[6]
    R_ratio = var[7]

    # Near-field and far-field mass fractions:
    q11, q12, q13, q14, q15, q21, q22, q31, q32, q33, q34, q35, q36, q38, q41, q42, q43, q44, q45, q46, q47, zeta5, zeta6 = fit_params_NF_mass.iloc[0].values
    
    zeta1 = q11 + (q12 * b) + (q13 * b**q14) + (q15 * f_atm)
    zeta2 = q21 + (q22 * b)
    zeta3 = q31 + (q32 * b) + (q33 * b**q34) + (q35 * f_atm) + (q36 * f_atm**2) + (M_t)**q38
    zeta4 = q41 + (q42 * b) + (q43 * b**q44) + (q45 * f_atm) + (q46 * f_atm**2) + (q47 * M_t)

    f_NF = zeta4 / (1 + zeta6 * np.exp(zeta3 * (R_ratio - zeta2)))**zeta1 + zeta5
    f_NF = np.clip(f_NF, 0, 1)
    f_FF = 1 - f_NF

    # Near-field:
    k11, k12, k13, k14, k15, k16, k17, k21, k22, k23, k24, k25, k31, k32, k33, k34, k35 = fit_params_NF.iloc[0].values

    xi1 = k11 + (k12 * gamma) + (k13 * (gamma+0.05)**2) + (k14 * vc**(k15)) + (k16 * (M_t+1)) + (k17 * np.log10(f_atm))
    xi2 = k21 + (k22 * gamma) + (k23 * (gamma+0.05)**2) + (k24 * vc**(k25))
    xi3 = k31 + (k32 * gamma) + (k33 * (gamma+0.05)**2) + (k34 * vc**(k35))
    
    X_NF = f_NF * (xi1 - xi2 * (b + xi3)**2)

    xi1 = k11 + (k12 * gamma) + (k13 * (gamma+0.05)**2) + (k14) + (k16 * (M_t+1)) + (k17 * np.log10(f_atm))
    xi2 = k21 + (k22 * gamma) + (k23 * (gamma+0.05)**2) + (k24)
    xi3 = k31 + (k32 * gamma) + (k33 * (gamma+0.05)**2) + (k34)

    X_NF_vesc = f_NF * (xi1 - xi2 * (b + xi3)**2)

    if np.isscalar(X_NF):
        X_NF = np.clip(X_NF, max(0, X_NF_vesc), f_NF)
    else:
        X_NF = np.clip(np.array(X_NF), np.maximum(0, np.array(X_NF_vesc)), f_NF)


    # Far-field:
    s11, s12, s13, s14, s15, s16, s21, s22, s23, s24, s25, s26, s31, s32, s33, s41, s42, s43, s44, s45, s46 = fit_params_FF.iloc[0].values

    psi1 = s11 + (s12 * (gamma+0.05)**s13) + (s14 * (M_t+0.05)**s15) + (s16 * np.log10(f_atm))
    psi2 = s21 + (s22 * (gamma+0.05)**s23) + (s24 * (M_t+0.05)**s25) + (s26 * np.log10(f_atm))
    psi3 = s31 + (s32 * (gamma+0.05)**s33)
    psi4 = s41 + (s42 * (gamma+0.05)**s43) + (s44 * (M_t+0.05)**s45) + (s46 * np.log10(f_atm))

    X_FF = f_FF * (psi1 * np.exp(-psi2 * (Q_R_prime * (1 + (M_i / M_t)) * (1 - b)**psi4)) + psi3)
    X_FF = np.clip(X_FF, 0, f_FF)


    # Total:
    X_atm = X_NF + X_FF
    X_atm = np.clip(X_atm, 0, 1)


    return f_NF, f_FF, X_NF, X_FF, X_atm