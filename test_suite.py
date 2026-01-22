import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd

# --- IMPORT YOUR ALGORITHM ---
try:
    from ga_improvement import FilterPopulacija, IIRFilterIndividua
except ImportError:
    print("Error: Could not import 'ga_improvement.py'. Make sure it exists in the same folder.")
    exit()

def get_spec_masks(type_name, w_vec):
    """Generates Pass/Stop masks based on Paper Table 1 Specifications."""
    pi = np.pi
    if type_name == 'LPF':
        pass_mask = w_vec <= 0.2 * pi
        stop_mask = w_vec >= 0.3 * pi
    elif type_name == 'HPF':
        stop_mask = w_vec <= 0.7 * pi
        pass_mask = w_vec >= 0.8 * pi
    elif type_name == 'BPF':
        stop_mask = (w_vec <= 0.2 * pi) | (w_vec >= 0.8 * pi)
        pass_mask = (w_vec >= 0.3 * pi) & (w_vec <= 0.7 * pi)
    elif type_name == 'BSF':
        pass_mask = (w_vec <= 0.2 * pi) | (w_vec >= 0.8 * pi)
        stop_mask = (w_vec >= 0.3 * pi) & (w_vec <= 0.7 * pi)
    return pass_mask, stop_mask

def robust_eval_wrapper(ind, spec):
    """
    Wrapper to calculate fitness with proper gain normalization for ALL filter types.
    """
    sos = ind.get_sos_matrix()
    w, h_unscaled = signal.sosfreqz(sos, worN=spec['w'])
    mag_unscaled = np.abs(h_unscaled)
    
    # Robust Normalization: Find peak in passband
    if np.any(spec['pass_mask']):
        max_pass_gain = np.max(mag_unscaled[spec['pass_mask']])
    else:
        max_pass_gain = mag_unscaled[0] + 1e-9
        
    K = 1.0 / (max_pass_gain + 1e-9)
    mag_actual = mag_unscaled * K
    
    # Calculate Fitness
    pb_mag = mag_actual[spec['pass_mask']]
    e_p = np.maximum(0, pb_mag - 1.0) + np.maximum(0, (1.0 - spec['delta1']) - pb_mag)
    
    sb_mag = mag_actual[spec['stop_mask']]
    e_s = np.maximum(0, sb_mag - spec['delta2'])
    
    sse = np.sum(e_p**2) + np.sum(e_s**2)
    ind.fitness = 1.0 / (1.0 + sse)
    return ind.fitness

def run_full_test_suite():
    print("--- STARTING COMPREHENSIVE FILTER TEST SUITE ---")
    
    types = ['LPF', 'HPF', 'BPF', 'BSF']
    results = []
    
    # Setup Plot 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    w_vec = np.linspace(0, np.pi, 512)
    
    for i, f_type in enumerate(types):
        print(f"\nTesting Filter Type: {f_type}...")
        
        p_mask, s_mask = get_spec_masks(f_type, w_vec)
        spec = {
            'w': w_vec,
            'pass_mask': p_mask,
            'stop_mask': s_mask,
            'delta1': 0.1,
            'delta2': 0.032 
        }
        
        # Paper Table 2: LPF/HPF order 4 (2 SOS), BPF/BSF order 6 (3 SOS)
        n_sos = 2 if f_type in ['LPF', 'HPF'] else 3
        
        # --- FIX IS HERE ---
        # Explicitly passing num_fos=0
        ga = FilterPopulacija(
            velicinaPop=100, 
            num_fos=0,       # <-- This was likely missing or misaligned
            num_sos=n_sos, 
            p_cross=0.9, 
            p_mut=0.2, 
            max_gen=1000, 
            elite_size=2
        )
        
        # Inject Robust Evaluation
        original_eval = IIRFilterIndividua.evaluiraj
        IIRFilterIndividua.evaluiraj = robust_eval_wrapper
        
        best_ind, _ = ga.evoluiraj(spec)
        
        # Restore original
        IIRFilterIndividua.evaluiraj = original_eval
        
        # Validation
        sos = best_ind.get_sos_matrix()
        z, p, k = signal.sos2zpk(sos)
        max_pole = np.max(np.abs(p)) if len(p) > 0 else 0
        is_stable = max_pole < 1.0
        
        results.append({
            'Type': f_type,
            'Order': n_sos * 2,
            'Fitness': best_ind.fitness,
            'Stable': is_stable,
            'Max Pole': f"{max_pole:.4f}"
        })
        
        # Plotting
        w, h = signal.sosfreqz(sos, worN=w_vec)
        mag = np.abs(h)
        if np.any(p_mask):
            mag = mag / (np.max(mag[p_mask]) + 1e-9)
        
        ax = axes[i]
        ax.plot(w/np.pi, mag, 'b', label='GA Result')
        ax.fill_between(w_vec/np.pi, 0, 1.2, where=p_mask, color='g', alpha=0.1, label='Pass')
        ax.fill_between(w_vec/np.pi, 0, 1.2, where=s_mask, color='r', alpha=0.1, label='Stop')
        ax.axhline(1.0, color='k', linestyle=':', alpha=0.3)
        ax.axhline(spec['delta2'], color='r', linestyle='--', alpha=0.5)
        ax.set_title(f"{f_type} (Order {n_sos*2})")
        ax.grid(True)
        if i == 0: ax.legend()

    plt.tight_layout()
    plt.show()
    
    print("\n--- TEST SUITE RESULTS ---")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    run_full_test_suite()