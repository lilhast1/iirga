# IIR Filter Design Optimization via Genetic Algorithm

This repository contains a Python implementation of a **Genetic Algorithm (GA)** designed to optimize **Infinite Impulse Response (IIR)** digital filters. Unlike classical design methods (like Butterworth or Chebyshev) which rely on analog prototypes and bilinear transformations, this approach performs **direct optimization in the z-domain**.

## 🚀 Key Features

*   **Guaranteed Stability:** Uses the "Stability Triangle" geometric constraints to ensure all generated filters are inherently stable without needing computationally expensive pole checks or penalty functions.
*   **SOS Realization:** Implements filters using **Second-Order Sections (SOS)** kaskade, providing high numerical robustness and low sensitivity to coefficient quantization.
*   **Direct Z-Domain Design:** Capable of designing Low-Pass (LPF), High-Pass (HPF), Band-Pass (BPF), and Band-Stop (BSF) filters directly from frequency specifications.
*   **Order Efficiency:** Achieves the same frequency specifications as classical filters but with a **significantly lower filter order** (e.g., a 6th-order GA filter can match a 14th-order Butterworth filter).
*   **Dynamic Evolution:** Features adaptive mutation rates and weighted crossover based on relative fitness.

## 🛠 How It Works

The algorithm treats filter coefficients as a chromosome of real numbers. 

1.  **Fitness Function:** Instead of simple Mean Squared Error (MSE), we use a **Tolerance Mask**. Error is only calculated if the magnitude response violates the pass-band ripple ($\delta_1$) or stop-band attenuation ($\delta_2$) limits.
2.  **Constraint Handling:** During mutation and initialization, coefficients $a_1$ and $a_2$ of each SOS are clamped to the stability region:
    *   $|a_2| < 1$
    *   $|a_1| < 1 + a_2$
3.  **Optimization:** The GA explores the non-linear, multi-modal error surface to find a global minimum that satisfies the design requirements with minimal hardware complexity.

## 📦 Prerequisites

*   Python 3.9+
*   NumPy
*   SciPy
*   Matplotlib

Install dependencies via pip:
```bash
pip install numpy scipy matplotlib
```

## 💻 Usage

Run the main script to design a filter based on the specifications defined in the code:

```bash
python final.py
```

### Configuration
You can modify the `spec` dictionary in the script to target different cut-off frequencies and tolerances:
```python
spec = {
    'w': w_vec,
    'pass_mask': w_vec <= 0.4 * np.pi, # Low-pass example
    'stop_mask': w_vec >= 0.5 * np.pi,
    'delta1': 0.15, # Pass-band tolerance
    'delta2': 0.036 # Stop-band tolerance
}
```

## 📊 Results

The implementation generates four comprehensive plots to evaluate the optimized filter:
1.  **Magnitude Response:** Shows the filter output against the ideal target and tolerance bounds.
2.  **Gain (dB):** Logarithmic visualization of the stop-band attenuation.
3.  **Pole-Zero Plot:** Visual proof of stability (all poles remain inside the unit circle).
4.  **Step Response:** Time-domain behavior showing the settling time and overshoot.

### Comparison with Classical Methods
| Filter Type | GA Order (Our) | Butterworth Order |
| :--- | :---: | :---: |
| Low-Pass | **4** | 9 |
| Band-Pass | **6** | 14 |

## 🎓 Authors
*   **Eldar Buzadžić** - [ebuzadzic1@etf.unsa.ba](mailto:ebuzadzic1@etf.unsa.ba)
*   **Tarik Hastor** - [thastor1@etf.unsa.ba](mailto:thastor1@etf.unsa.ba)

*Faculty of Electrical Engineering, University of Sarajevo.*

## 📜 Citation
If you use this code for research, please cite our paper:
> Buzadžić, E., & Hastor, T. (2026). *Optimizacija dizajna IIR filtera korištenjem genetičkog algoritma*. Faculty of Electrical Engineering Sarajevo.
