import numpy as np
import matplotlib.pyplot as plt
from ga_improvement import FilterPopulacija, plot_results




def situation(num_freq_points, pass_cutoff, stop_cutoff, delta1, delta2):
    w_vec = np.linspace(0, np.pi, num_freq_points)
        
    pass_mask = w_vec <= pass_cutoff
    stop_mask = w_vec >= stop_cutoff

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
        p_cross=0.9, 
        p_mut=0.2, 
        max_gen=1000, 
        elite_size=10
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



def main():
    # situation(512, 0.4 * np.pi, 0.5*np.pi, 0.15, 0.036)

    # situation(512, 0.4 * np.pi, 0.45*np.pi, 0.15, 1e-5) # ne rjesava ovo je pretesko

    # situation(512, 0.4 * np.pi, 0.45*np.pi, 0.15, 1e-4) # nope

    situation(1<<12, 0.4 * np.pi, 0.41*np.pi, 0.15, 0.15) # nope

    
if __name__=='__main__':
    main()