# Floquet counterdiabatic protocols for Quantum Annealing on Parity architecture
<p align="center">my MSc thesis</p>

This repository contains the code & data used for my MSc thesis. 

Link to the thesis: (available soon)


## Abstract

Combinatorial Optimization problems can be addressed with quantum computing techniques: the solution of an optimization problem can be encoded in the ground state of a quantum spin system, which in turn can be experimentally prepared through a suitable algorithm like Quantum Annealing.

Still, annealing applications have to face several challenges. Firstly, the limited spin connectivity of current hardware requires the use of clever encoding strategies. 
In this sense, the Parity architecture can encode fully connected optimization problems without requiring on-hardware long-range connectivity. 
Secondly, a quantum system naturally transits toward excited states during its evolution, resulting in a prepared state with a reduced overlap over the true ground state of the problem-encoding Hamiltonian. To overcome this limitation, one needs to find and implement non-trivial Quantum Annealing protocols.

<br>

In this work, we investigate Floquet counterdiabatic protocols for the Parity architecture. This combination of encoding and protocols constitutes a hardware-friendly recipe that does not require additional system controls.

On the one hand, we study the accuracy of the Floquet protocols, eventually finding a suitable criterion to tune the protocol hyperparameters.
On the other hand, we investigate the efficiency of Floquet schedules. 
We identify a narrow driving frequency regime where the Floquet protocol provides an advantage over the standard Quantum Annealing approach.

We explore alternative protocols obtained by extending the derivation of Floquet schedules to higher orders of precision.
Finally, we move away from the Floquet formulation and relax the requirement of preserving the original set of controls over the quantum system.
Our work shows that numerically optimized counterdiabatic protocols using only extra local field controls can be advantageous over standard annealing approaches.


### About the benchmark scripts

The description of the code is partial, as this was not meant to be published & used out of the scope of my project. However, I provide the data (see in `data/`) that has been used for the analysis/plots of the thesis.

The annealing protocols are implemented in `lib/protocols`. The `cdtool` provides routines to optimize the Adiabatic Gauge Potential for our specific use scenario. 

The following scripts execute the required benchmarks for a specific combination of model and protocol.
- `bench-oscgr.py`: benchmark various $\omega\tau/2\pi$ and $\omega_0/2\pi$
- `bench-teff.py`: benchmarks $\tau_{eff}$ from UA schedule
Example usage:
```bash
python3 bench-teff.py -model lhz6m
python3 bench-oscgr.py -protocol claeys-l1 -model lhz6m
python3 bench-oscgr.py -protocol ord3-l1 -model lhz6m
python3 bench-oscgr.py -protocol ord3v2-l1 -model lhz6m
```

**Requirements**: `qutip`, `numpy`, `scipy`, `matplotlib`, `sympy`.


## (very short) Bibliography

* Pieter W. Claeys et al. - **Floquet-Engineering Counterdiabatic Protocols in Quantum Many-Body Systems**, [Phys. Rev. Lett. 123](https://doi.org/10.1103/PhysRevLett.123.090602) (9 Aug. 2019)
* Wolfgang Lechner, Philipp Hauke, and Peter Zoller. - **A quantum annealing architecture with all-to-all connectivity from local interactions**, [Science Advances 1.9](https://doi.org/10.1126/sciadv.1500838) (2015)
* Ieva Čepaitė et al. - **Counterdiabatic Optimized Local Driving**, [PRX Quantum 4](https://doi.org/10.1103/PRXQuantum.4.010312) (1 Jan. 2023)


---

<h5 align="center">MSc thesis<br>AY 2023/2024 - University of Padua</h5>

<p align="center">
  <img src="https://raw.githubusercontent.com/baronefr/baronefr/main/shared/2022_unipd.png" alt="" height="70"/>
  &emsp;
  <img src="https://raw.githubusercontent.com/baronefr/baronefr/main/shared/2022_pod.png" alt="" height="70"/>
</p>
