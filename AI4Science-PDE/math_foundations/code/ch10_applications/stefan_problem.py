"""
Chapter 10: 1D Stefan Problem (Melting Front)
Using the Enthalpy Method.
"""
import numpy as np

def enthalpy_method_stefan():
    print("--- 1D Stefan Problem (Enthalpy Method) ---")
    
    # Physical parameters
    k_solid = 1.0     # Thermal conductivity of solid
    k_liquid = 1.0    # Thermal conductivity of liquid
    c_solid = 1.0     # Heat capacity of solid
    c_liquid = 1.0    # Heat capacity of liquid
    L = 10.0          # Latent heat of fusion
    T_melt = 0.0      # Melting temperature
    
    # Domain and grid
    nx = 100
    length = 1.0
    dx = length / (nx - 1)
    
    # Time stepping
    dt = 0.0001
    t_final = 0.05
    nt = int(t_final / dt)
    
    # Initial conditions
    # Start with all solid at T = -1, except left boundary T = +2 (hot)
    T = np.full(nx, -1.0)
    T[0] = 2.0
    
    # Enthalpy function H(T)
    # H = c_s * (T - Tm)                for T < Tm
    # H = c_l * (T - Tm) + L            for T > Tm
    # In mushy zone (T = Tm), H is between 0 and L
    
    def temp_to_enthalpy(T_array):
        H = np.zeros_like(T_array)
        mask_solid = T_array < T_melt
        mask_liquid = T_array >= T_melt
        
        H[mask_solid] = c_solid * (T_array[mask_solid] - T_melt)
        H[mask_liquid] = c_liquid * (T_array[mask_liquid] - T_melt) + L
        return H
        
    def enthalpy_to_temp(H_array):
        T_new = np.zeros_like(H_array)
        mask_solid = H_array < 0
        mask_liquid = H_array > L
        mask_mushy = (H_array >= 0) & (H_array <= L)
        
        T_new[mask_solid] = T_melt + H_array[mask_solid] / c_solid
        T_new[mask_liquid] = T_melt + (H_array[mask_liquid] - L) / c_liquid
        T_new[mask_mushy] = T_melt
        return T_new
        
    H = temp_to_enthalpy(T)
    
    print(f"Simulation started: {nt} steps...")
    
    # Time integration (Explicit Euler)
    for n in range(nt):
        H_old = H.copy()
        
        # Calculate conductivities at cell interfaces
        # Simplified: using constant k everywhere for this demo
        k = k_solid
        
        # Update interior points
        # dH/dt = div(k grad T)
        H[1:-1] = H_old[1:-1] + dt * k / dx**2 * (T[2:] - 2*T[1:-1] + T[:-2])
        
        # Boundary conditions
        T[0] = 2.0 # Fixed hot boundary
        H[0] = temp_to_enthalpy(np.array([2.0]))[0]
        
        H[-1] = H[-2] # Zero flux on right
        
        # Recover temperature from enthalpy
        T = enthalpy_to_temp(H)
        
        if n % (nt // 5) == 0:
            # Find phase interface (first index where T <= T_melt)
            solid_idx = np.where(T <= T_melt)[0]
            if len(solid_idx) > 0:
                front_pos = solid_idx[0] * dx
                print(f"Time t={n*dt:.4f}: Melting front at x={front_pos:.3f}")
                
    print("\nSimulation complete.")
    solid_idx = np.where(T <= T_melt)[0]
    final_front = solid_idx[0] * dx if len(solid_idx) > 0 else length
    print(f"Final melting front position: x={final_front:.3f}")

if __name__ == "__main__":
    enthalpy_method_stefan()
