
import sympy
import qutip

import cdtool.gauge.varmin as varmin


class ansatz_gauge:

    def __init__(self, H_op:list, ansatz_op:list) -> None:
        """Minimize the N coefficients for a general ansatz of N terms."""
        
        N_ham :int = len(H_op)
        N_op :int = len(ansatz_op)

        # define symbols
        alphas = [ sympy.Symbol(f"\\alpha_{ii}", commutative=True) for ii in range(N_op) ]
        Ls = [ sympy.Symbol("L_" + f"{ii}", commutative=True) for ii in range(N_ham) ]
        Ldots = [ sympy.Symbol("\dot{L}_" + f"{ii}", commutative=True) for ii in range(N_ham) ]
        
        G = 0
        associations = {}
        commutator_placeholder = 0
        for idx, HH in enumerate(H_op):
            # define H term symbol
            symbol_name = f"H_{idx}"
            thisH = sympy.Symbol(symbol_name, commutative=False)

            associations[symbol_name] = HH
            
            # add derivative
            G += Ldots[idx] * thisH
            
            # add commutators
            for alph, aa in zip(alphas, ansatz_op):
                placeholder_coeff = sympy.Symbol(f"\\gamma_{commutator_placeholder}", commutative=False)
                G += 1j * Ls[idx] * alph * placeholder_coeff

                associations[ str(placeholder_coeff.name) ] = qutip.commutator(aa, HH)
                commutator_placeholder += 1
        
        self.G = G
        self.associations = associations
        self.ansatz_coefficients = alphas
        self.symbols = Ls + Ldots
        
    def variational_min(self):
        return varmin.varmin(self.G, self.associations, self.ansatz_coefficients, self.symbols)