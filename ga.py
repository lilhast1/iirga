import numpy as np
import scipy.special as scpy
from scipy import signal
import matplotlib.pyplot as plt

class ApstraktnaIndividua:
    def __init__(self, duzinaHromozoma):
        self.duzinaHromozoma = duzinaHromozoma
        # Real-valued chromosome for filter coefficients
        self.hromozom = np.random.uniform(-1, 1, size=duzinaHromozoma)
        self.fitness = 0.0

    def getFitness(self):
        return self.fitness

    def setFitness(self, fitness):
        self.fitness = fitness

class IIRFilterIndividua(ApstraktnaIndividua):
    def __init__(self, order):
        # Order N: N+1 b coefficients, N a coefficients (a0 is fixed to 1)
        self.order = order
        super().__init__(2 * order + 1) 

    def decode(self):
        b = self.hromozom[:self.order + 1]
        a = np.concatenate(([1.0], self.hromozom[self.order + 1:]))
        return b, a

    def is_stable(self):
        _, a = self.decode()
        poles = np.roots(a)
        return all(np.abs(poles) < 1.0)

    def __add__(self, other):
        if isinstance(other, IIRFilterIndividua):
            return self.fitness + other.fitness
        return self.fitness + other
    
    def __radd__(self, other):
        return self + other

    def evaluiraj(self, target_spec):
        b, a = self.decode()
        if not self.is_stable():
            self.fitness = 1e-12 # Penalty for instability
            return self.fitness

        # Calculate frequency response at target frequencies
        w, h_actual = signal.freqz(b, a, worN=target_spec['w'])
        mag_actual = np.abs(h_actual)
        
        # SSE Calculation
        sse = np.sum((target_spec['h_ideal'] - mag_actual)**2)
        self.fitness = 1.0 / (1.0 + sse)
        return self.fitness

class FilterPopulacija:
    def __init__(self, velicinaPop, order, p_cross, p_mut, max_gen, elite_size):
        self.velicinaPop = velicinaPop
        self.order = order
        self.p_cross = p_cross
        self.p_mut = p_mut
        self.max_gen = max_gen
        self.elite_size = elite_size
        self.populacija = [IIRFilterIndividua(order) for _ in range(velicinaPop)]
        self.best_ind = None
        self.best_fit = -1.0

    def selekcijaRTocak(self):
        fits = np.array([ind.getFitness() for ind in self.populacija])
        # Use softmax to prevent issues with very small fitness values
        probs = scpy.softmax(fits)
        return np.random.choice(self.populacija, p=probs)

    def crossover(self, parent_x, parent_y):
        """ a_i = u*a_i^x + (1-u)*a_i^y where u = 0.5 + f_x - f_y """
        if np.random.rand() > self.p_cross:
            return parent_x.hromozom.copy(), parent_y.hromozom.copy()

        u = 0.5 + (parent_x.getFitness() - parent_y.getFitness())
        u = np.clip(u, -0.5, 1.5) # Clamp to keep coefficients sane

        child1_h = u * parent_x.hromozom + (1 - u) * parent_y.hromozom
        child2_h = u * parent_y.hromozom + (1 - u) * parent_x.hromozom
        return child1_h, child2_h

    def mutacija(self, hromozom):
        for i in range(len(hromozom)):
            if np.random.rand() < self.p_mut:
                hromozom[i] += np.random.normal(0, 0.2)
        return hromozom

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
            # Elitism
            for i in range(self.elite_size):
                elite = IIRFilterIndividua(self.order)
                elite.hromozom = self.populacija[i].hromozom.copy()
                elite.setFitness(self.populacija[i].getFitness())
                new_pop.append(elite)

            while len(new_pop) < self.velicinaPop:
                p1 = self.selekcijaRTocak()
                p2 = self.selekcijaRTocak()
                c1_h, c2_h = self.crossover(p1, p2)
                
                for h in [c1_h, c2_h]:
                    if len(new_pop) < self.velicinaPop:
                        child = IIRFilterIndividua(self.order)
                        child.hromozom = self.mutacija(h)
                        new_pop.append(child)
            fit_hist.append(sum(self.populacija) / self.velicinaPop)
            self.populacija = new_pop
            if gen % 20 == 0:
                print(f"Generation {gen} | Best Fitness: {self.best_fit:.6f}")
            self.p_mut = (0.2 * np.exp((100 - gen) / gen)) / (1 + np.exp((100 - gen) / gen))

        return self.best_ind, fit_hist

def plot_results(best_ind, target_spec):
    b, a = best_ind.decode()
    w, h = signal.freqz(b, a, worN=target_spec['w'])
    mag = np.abs(h)
    
    # Gain in dB
    # We clip to -100dB to avoid log10(0)
    gain_db = 20 * np.log10(np.maximum(mag, 1e-5))
    ideal_db = 20 * np.log10(np.maximum(target_spec['h_ideal'], 1e-5))
    
    plt.figure(figsize=(15, 5))

    # Plot 1: Magnitude Response (Linear)
    plt.subplot(1, 3, 1)
    plt.plot(target_spec['w'] / np.pi, target_spec['h_ideal'], 'r--', label='Ideal')
    plt.plot(w / np.pi, mag, 'b-', label='GA Result')
    plt.title("Magnitude Response (Linear)")
    plt.xlabel("Frequency (xπ rad/sample)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Plot 2: Gain Response (dB)
    plt.subplot(1, 3, 2)
    plt.plot(target_spec['w'] / np.pi, ideal_db, 'r--', label='Ideal')
    plt.plot(w / np.pi, gain_db, 'g-', label='GA Result')
    plt.title("Gain Plot (dB)")
    plt.xlabel("Frequency (xπ rad/sample)")
    plt.ylabel("Gain [dB]")
    plt.ylim([-60, 5]) # Focus on relevant range
    plt.grid(True)
    plt.legend()

    # Plot 3: Pole-Zero Plot
    plt.subplot(1, 3, 3)
    z, p, k = signal.tf2zpk(b, a)
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

# --- RUNNING THE GA ---
if __name__ == "__main__":
    # Define a Low-Pass target
    num_freq_points = 512
    w_vec = np.linspace(0, np.pi, num_freq_points)
    # Ideal response: 1.0 in passband (0 to 0.4pi), 0.0 in stopband
    h_ideal = np.where(w_vec < 0.4 * np.pi, 1.0, 0.0)
    spec = {'w': w_vec, 'h_ideal': h_ideal}

    # Setup GA
    ga = FilterPopulacija(
        velicinaPop=100, 
        order=2,        # 5th order IIR filter
        p_cross=0.8, 
        p_mut=0.2, 
        max_gen=1000, 
        elite_size=2
    )

    best_individua, fit_hist = ga.evoluiraj(spec)
    
    # Final coefficients
    b_final, a_final = best_individua.decode()
    print("\n--- Final Filter Coefficients ---")
    print(f"Numerator (b): {b_final}")
    print(f"Denominator (a): {a_final}")
    
    plot_results(best_individua, spec)


    generations = range(len(fit_hist))

    plt.plot(generations, fit_hist)
    plt.xlabel("Generations")
    plt.ylabel("Average fitness")
    plt.title("Average Fitness per Generation")
    plt.grid(True)

    plt.show()

    # Hyperparameter tuning part, added different parameters to test.
    print("--- Hyperparameter Tuning ---")
    
    w_vec = np.linspace(0, np.pi, 256)
    h_ideal = np.where(w_vec < 0.4 * np.pi, 1.0, 0.0)
    spec = {'w': w_vec, 'h_ideal': h_ideal}

    param_grid = {
        'pop_size': [25, 50, 100],
        'p_cross': [0.8, 0.9, 0.92],
        'p_mut': [0.1, 0.2, 0.25, 0.3],  
        'max_gen': [200]      # Lower gen count for speed during tuning
    }

    results = []

    total_runs = len(param_grid['pop_size']) * len(param_grid['p_cross']) * len(param_grid['p_mut'])
    current_run = 0

    for pop in param_grid['pop_size']:
        for pc in param_grid['p_cross']:
            for pm in param_grid['p_mut']:
                current_run += 1
                
                trial_fitnesses = []
                for trial in range(2):
                    # Re-initialize GA with current params
                    ga = FilterPopulacija(
                        velicinaPop=pop, 
                        order=2,       
                        p_cross=pc, 
                        p_mut=pm, 
                        max_gen=param_grid['max_gen'][0], 
                        elite_size=2
                    )
                    best_ind, _ = ga.evoluiraj(spec)
                    trial_fitnesses.append(best_ind.getFitness())
                
                avg_fit = np.mean(trial_fitnesses)
                results.append((pop, pc, pm, avg_fit))
                print(f"Run {current_run}/{total_runs} | Pop: {pop}, Cross: {pc}, Mut: {pm} -> Avg Fit: {avg_fit:.4f}")

    best_params = max(results, key=lambda x: x[3])
    print("\n--- Tuning Complete ---")
    print(f"Best Parameters for current GA:")
    print(f"Population: {best_params[0]}")
    print(f"Crossover:  {best_params[1]}")
    print(f"Mutation:   {best_params[2]}")
    print(f"Best Fitness achieved: {best_params[3]:.4f}")
