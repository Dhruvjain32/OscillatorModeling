import math
import matplotlib.pyplot as plt
import numpy as np

### Shared system parameters

# Unforced System (Fades completely to 0) ---
force = 0  # No external driving force
mass = 15.625  # Constant mass (kg)
k = 10.0  # Spring constant (N/m)
omega = 2.0  # Driving frequency (unused when force = 0)
gamma = 25.0  # Damping coefficient (N·s/m)

# Use gamma values of 5, 25, and 50 to see the effect of damping on the system's response.

# --- OPTION 2: Forced System (Oscillates forever in steady-state) ---
# force = 5.0
# mass = 15.625
# k = 10.0
# omega = 2.0
# gamma = 25.0

x_0 = 13.0
v_0 = 4.0


def spring_oscillation(mass, force, k, omega, gamma, time):
  # Numerator and denominator for particular solution
  numer_real = force * (k - mass * omega**2)
  numer_imag = force * gamma * omega
  denom = (k - (mass * omega**2)) ** 2 + (gamma * omega) ** 2

  a = numer_real / denom
  b = numer_imag / denom

  # Particular solution
  particular = (a * np.cos(omega * time)) + (b * np.sin(omega * time))

  # Homogeneous solution
  # 1. Critically Damped (Check math.isclose FIRST!)
  if math.isclose(gamma**2, 4 * mass * k, rel_tol=1e-9, abs_tol=1e-9):
    r = -gamma / (2 * mass)
    
    c_1 = x_0 - a
    c_2 = v_0 - (r * c_1) - b * omega
    
    homogeneous = (c_1 + c_2 * time) * np.exp(r * time)

  # 2. Underdamped (gamma^2 < 4mk)
  elif gamma**2 < 4 * mass * k:
    alpha = -gamma / (2 * mass)
    beta = np.sqrt((4 * mass * k) - gamma**2) / (2 * mass)
    
    c_1 = x_0 - a
    c_2 = (v_0 - alpha * c_1 - b * omega) / beta
    
    homogeneous = np.exp(alpha * time) * (c_1 * np.cos(beta * time) + c_2 * np.sin(beta * time))

  # 3. Overdamped (gamma^2 > 4mk)
  elif gamma ** 2 > 4 * mass * k:
    r_1 = (-gamma + np.sqrt(gamma ** 2 - (4 * mass * k))) / (2 * mass)
    
    r_2 = (-gamma - np.sqrt(gamma ** 2 - (4 * mass * k))) / (2 * mass)
    c_1 = ((v_0 - b * omega) - r_2 * (x_0 - a)) / (r_1 - r_2)
    c_2 = x_0 - a - c_1
    
    homogeneous = c_1 * np.exp(r_1 * time) + c_2 * np.exp(r_2 * time)

  print(discriminant := gamma**2 - 4 * mass * k)
  return homogeneous + particular


# Time array (shortened to 20s so the transient damping behavior is clearly visible)
t = np.linspace(0, 20, 1000)

# Calculate all three cases by varying mass 
x_under = spring_oscillation(mass=35.0, force=force, k=k, omega=omega, gamma=gamma, time=t)  # m > 15.625
x_crit = spring_oscillation(mass=15.625, force=force, k=k, omega=omega, gamma=gamma, time=t)  # m == 15.625
x_over = spring_oscillation(mass=5.0, force=force, k=k, omega=omega, gamma=gamma, time=t)  # m < 15.625

# Plot all three on the same graph
plt.figure(figsize=(10, 6))
plt.plot(
    t,
    x_under,
    label='Underdamped (m = 35.0 kg)',
    color='blue',
    linewidth=2,
)
plt.plot(
    t,
    x_crit,
    label='Critically Damped (m = 15.625 kg)',
    color='green',
    linestyle='--',
    linewidth=2,
)
plt.plot(
    t, 
    x_over, 
    label='Overdamped (m = 5.0 kg)', 
    color='red', 
    linewidth=2
)

plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Forced Spring-Mass-Damper: Comparison of Damping Regimes')
plt.grid(True, alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
