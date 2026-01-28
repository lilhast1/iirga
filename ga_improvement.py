import numpy as np
import scipy.special as scpy
from scipy import signal
import matplotlib.pyplot as plt

def get_random_stable_fos():
    b1 = np.random.uniform(-2, 2)
    a1 = np.random.uniform(-0.99, 0.99) 
    return b1, a1

def get_random_stable_sos():
    b1 = np.random.uniform(-2, 2)
    b2 = np.random.uniform(-2, 2)
    while True:
        a1 = np.random.uniform(-2, 2)
        a2 = np.random.uniform(-1, 1)
        if (a2 < 1.0) and (a1 < a2 + 1.0) and (a1 > -(a2 + 1.0)):
            return b1, b2, a1, a2

def clamp_fos(a1):
    return np.clip(a1, -0.99, 0.99)

def clamp_sos(a1, a2):
    a2 = np.clip(a2, -0.99, 0.99)
    limit = a2 + 0.99 
    a1 = np.clip(a1, -limit, limit)
    return a1, a2


class ApstraktnaIndividua:
    def __init__(self, duzinaHromozoma):
        self.duzinaHromozoma = duzinaHromozoma
        self.hromozom = np.zeros(duzinaHromozoma) 
        self.fitness = 0.0

    def getFitness(self):
        return self.fitness

    def setFitness(self, fitness):
        self.fitness = fitness

class IIRFilterIndividua(ApstraktnaIndividua):
    def __init__(self, num_fos, num_sos):
        self.num_fos = num_fos
        self.num_sos = num_sos
        length = (2 * num_fos) + (4 * num_sos)
        super().__init__(length)

    def initialize_stable(self):
        genes = []
        for _ in range(self.num_fos):
            genes.extend(get_random_stable_fos())
        for _ in range(self.num_sos):
            genes.extend(get_random_stable_sos())
        self.hromozom = np.array(genes)

    def get_sos_matrix(self):
        sos = []
        idx = 0
        for _ in range(self.num_fos):
            b1 = self.hromozom[idx]
            a1 = self.hromozom[idx+1]
            idx += 2
            sos.append([1.0, b1, 0.0, 1.0, a1, 0.0])
        for _ in range(self.num_sos):
            b1 = self.hromozom[idx]
            b2 = self.hromozom[idx+1]
            a1 = self.hromozom[idx+2]
            a2 = self.hromozom[idx+3]
            idx += 4
            sos.append([1.0, b1, b2, 1.0, a1, a2])
        return np.array(sos)

    def evaluiraj(self, target_spec):
        """
        [cite_start]Fitness calculation using tolerance masks [cite: 77-78].
        """
        sos = self.get_sos_matrix()
        w, h_unscaled = signal.sosfreqz(sos, worN=target_spec['w'])
        mag_unscaled = np.abs(h_unscaled)
        
        # Normalize DC (w=0) to 1.0 for LPF
        dc_mag = mag_unscaled[0]
        K = 1.0 if dc_mag < 1e-9 else 1.0 / dc_mag
        mag_actual = mag_unscaled * K

        # Pass-band Error
        pb_mask = target_spec['pass_mask']
        pb_mag = mag_actual[pb_mask]
        delta1 = target_spec['delta1']
        
        e_p = np.zeros_like(pb_mag)
        e_p = np.maximum(e_p, pb_mag - 1.0) 
        e_p = np.maximum(e_p, (1.0 - delta1) - pb_mag)

        # Stop-band Error
        sb_mask = target_spec['stop_mask']
        sb_mag = mag_actual[sb_mask]
        delta2 = target_spec['delta2']
        
        e_s = np.zeros_like(sb_mag)
        e_s = np.maximum(e_s, sb_mag - delta2)

        sse = np.sum(e_p**2) + np.sum(e_s**2)
        self.fitness = 1.0 / (1.0 + sse)
        return self.fitness

class FilterPopulacija:
    def __init__(self, velicinaPop, num_fos, num_sos, p_cross, p_mut, max_gen, elite_size):
        self.velicinaPop = velicinaPop
        self.num_fos = num_fos
        self.num_sos = num_sos
        self.p_cross = p_cross
        self.p_mut = p_mut
        self.max_gen = max_gen
        self.elite_size = elite_size
        
        self.populacija = []
        for _ in range(velicinaPop):
            ind = IIRFilterIndividua(num_fos, num_sos)
            ind.initialize_stable()
            self.populacija.append(ind)
            
        self.best_ind = None
        self.best_fit = -1.0

    def selekcijaRTocak(self):
        fits = np.array([ind.getFitness() for ind in self.populacija])
        if np.all(fits == fits[0]):
             probs = np.ones(len(fits)) / len(fits)
        else:
             probs = scpy.softmax(fits) 
        idx = np.random.choice(range(len(self.populacija)), p=probs)
        return self.populacija[idx]

    def crossover(self, parent_x, parent_y):
        if np.random.rand() > self.p_cross:
            return parent_x.hromozom.copy(), parent_y.hromozom.copy()

        delta_f = parent_x.getFitness() - parent_y.getFitness()
        u = 0.5 * delta_f + 0.5
        u = np.clip(u, 0.0, 1.0) 

        child1_h = u * parent_x.hromozom + (1 - u) * parent_y.hromozom
        child2_h = u * parent_y.hromozom + (1 - u) * parent_x.hromozom
        return child1_h, child2_h

    def mutacija(self, hromozom):
        mutated = hromozom.copy()
        idx = 0
        for _ in range(self.num_fos):
            if np.random.rand() < self.p_mut: mutated[idx] += np.random.normal(0, 0.2)
            if np.random.rand() < self.p_mut: 
                mutated[idx+1] += np.random.normal(0, 0.2)
                mutated[idx+1] = clamp_fos(mutated[idx+1])
            idx += 2
        for _ in range(self.num_sos):
            if np.random.rand() < self.p_mut: mutated[idx] += np.random.normal(0, 0.2)
            if np.random.rand() < self.p_mut: mutated[idx+1] += np.random.normal(0, 0.2)
            
            a1_idx, a2_idx = idx + 2, idx + 3
            if np.random.rand() < self.p_mut: mutated[a1_idx] += np.random.normal(0, 0.2)
            if np.random.rand() < self.p_mut: mutated[a2_idx] += np.random.normal(0, 0.2)
            mutated[a1_idx], mutated[a2_idx] = clamp_sos(mutated[a1_idx], mutated[a2_idx])
            idx += 4
        return mutated

    def evoluiraj(self, target_spec):
        fit_hist = []
        for gen in range(1, 1 + self.max_gen):
            for ind in self.populacija:
                ind.evaluiraj(target_spec)
            
            self.populacija.sort(key=lambda x: x.getFitness(), reverse=True)
            if self.populacija[0].getFitness() > self.best_fit:
                self.best_fit = self.populacija[0].getFitness()
                self.best_ind = self.populacija[0]

            new_pop = []
            for i in range(self.elite_size):
                elite = IIRFilterIndividua(self.num_fos, self.num_sos)
                elite.hromozom = self.populacija[i].hromozom.copy()
                elite.setFitness(self.populacija[i].getFitness())
                new_pop.append(elite)

            while len(new_pop) < self.velicinaPop:
                p1 = self.selekcijaRTocak()
                p2 = self.selekcijaRTocak()
                c1_h, c2_h = self.crossover(p1, p2)
                for h in [c1_h, c2_h]:
                    if len(new_pop) < self.velicinaPop:
                        child = IIRFilterIndividua(self.num_fos, self.num_sos)
                        child.hromozom = self.mutacija(h)
                        new_pop.append(child)
            
            self.populacija = new_pop
            fit_hist.append(self.best_fit)
            
            k0 = 100
            term = np.exp((k0 - gen) / (gen if gen > 0 else 1))
            self.p_mut = (0.2 * term) / (1 + term)
            
            if gen % 50 == 0:
                print(f"Gen {gen}: Best Fit = {self.best_fit:.5f}, Mut Rate = {self.p_mut:.4f}")

        return self.best_ind, fit_hist

# --- PLOTTING FUNCTION ---

def plot_results(best_ind, target_spec):
    # Decode to SOS
    sos = best_ind.get_sos_matrix()
    
    # Calculate Frequency Response
    w, h_unscaled = signal.sosfreqz(sos, worN=target_spec['w'])
    mag_unscaled = np.abs(h_unscaled)
    
    # Recalculate K for plotting (same logic as evaluiraj)
    dc_mag = mag_unscaled[0]
    K = 1.0 if dc_mag < 1e-9 else 1.0 / dc_mag
    
    mag = mag_unscaled * K
    gain_db = 20 * np.log10(np.maximum(mag, 1e-5))
    
    # Ideal lines for plotting
    # We construct a visual "ideal" line based on the spec
    h_visual = np.zeros_like(w)
    h_visual[target_spec['pass_mask']] = 1.0
    # Stop band is 0.0, so it stays 0
    
    plt.figure(figsize=(15, 5))

    # Plot 1: Magnitude Response
    plt.subplot(1, 3, 1)
    

    plt.plot(target_spec['w'] / np.pi, h_visual, 'r--', label='Ideal Target', alpha=0.5)
    plt.plot(w / np.pi, mag, 'b-', label='GA Result')
    
    # Draw Tolerance Bounds
    plt.axhline(1.0 - target_spec['delta1'], color='g', linestyle=':', label='Pass Tol')
    plt.axhline(target_spec['delta2'], color='g', linestyle=':', label='Stop Tol')
    
    plt.title("Magnitude Response")
    plt.xlabel("Frequency (xπ rad/sample)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(w / np.pi, gain_db, 'b-', label='GA Result')
    stop_db = 20 * np.log10(target_spec['delta2'])
    plt.axhline(stop_db, color='r', linestyle='--', label=f'Stop Limit ({stop_db:.1f}dB)')
    
    plt.title("Gain Plot (dB)")
    plt.xlabel("Frequency (xπ rad/sample)")
    plt.ylabel("Gain [dB]")
    plt.ylim([-80, 5])
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    
    z, p, k = signal.sos2zpk(sos) 
    uc = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(uc), np.sin(uc), 'k--', alpha=0.3)
    plt.scatter(np.real(z), np.imag(z), marker='o', edgecolors='b', facecolors='none', label='Zeros')
    plt.scatter(np.real(p), np.imag(p), marker='x', color='r', label='Poles')
    plt.title("Stability (Poles & Zeros)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_freq_points = 512
    w_vec = np.linspace(0, np.pi, num_freq_points)
    
    pass_cutoff = 0.4 * np.pi
    stop_cutoff = 0.5 * np.pi
    
    pass_mask = w_vec <= pass_cutoff
    stop_mask = w_vec >= stop_cutoff

    delta1 = 0.15
    delta2 = 0.036 

    spec = {
        'w': w_vec,
        'pass_mask': pass_mask,
        'stop_mask': stop_mask,
        'delta1': delta1,
        'delta2': delta2
    }

    print("Initializing Filter Design with Genetic Algorithm...")
    ga = FilterPopulacija(
        velicinaPop=100, 
        num_fos=0, 
        num_sos=2, 
        p_cross=0.8, 
        p_mut=0.2, 
        max_gen=1000, 
        elite_size=2
    )

    best_individua, fit_hist = ga.evoluiraj(spec)
    
    print("\nOptimization Complete.")
    print(f"Final Fitness: {best_individua.getFitness():.6f}")
    print("SOS Coefficients:\n", best_individua.get_sos_matrix())
    
    # Plot Results
    plot_results(best_individua, spec)

    # Plot Fitness History
    plt.figure()
    plt.plot(fit_hist)
    plt.xlabel("Generations")
    plt.ylabel("Best Fitness")
    plt.title("Fitness Evolution")
    plt.grid(True)
    plt.show()


    # Hyperparameter tuning part
    print("--- Hyperparameter Tuning: New Code (Refactored) ---")
    
    w_vec = np.linspace(0, np.pi, 256)
    spec = {
        'w': w_vec,
        'pass_mask': w_vec <= 0.35 * np.pi,
        'stop_mask': w_vec >= 0.45 * np.pi,
        'delta1': 0.1,
        'delta2': 0.01 
    }

    param_grid = {
        'pop_size': [25, 50, 100],
        'p_cross': [0.7, 0.8, 0.9, 0.95],
        'p_mut': [0.1, 0.2, 0.25, 0.3],
        'max_gen': [200]
    }

    results = []
    
    for pop in param_grid['pop_size']:
        for pc in param_grid['p_cross']:
            for pm in param_grid['p_mut']:
                fits = []
                for _ in range(2):
                    ga = FilterPopulacija(
                        velicinaPop=pop, 
                        num_fos=0, 
                        num_sos=2, 
                        p_cross=pc, 
                        p_mut=pm, 
                        max_gen=param_grid['max_gen'][0], 
                        elite_size=2
                    )
                    
                    best_ind, _ = ga.evoluiraj(spec)
                    fits.append(best_ind.getFitness())
                
                avg_fit = np.mean(fits)
                results.append((pop, pc, pm, avg_fit))
                print(f"Pop: {pop}, Cross: {pc}, Mut: {pm} -> Avg Fit: {avg_fit:.4f}")

    best_params = max(results, key=lambda x: x[3])
    print("\n--- Tuning Complete ---")
    print(f"Best Parameters for New Code:")
    print(f"Population: {best_params[0]}")
    print(f"Crossover:  {best_params[1]}")
    print(f"Mutation:   {best_params[2]}")
    print(f"Best Fitness achieved: {best_params[3]:.4f}")