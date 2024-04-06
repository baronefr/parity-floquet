

import sympy
import qutip

import cdtool.gauge.varmin as varmin


class manual_l1_gauge:
    def __init__(self, H_i, H_p):
        """
        Numerically optimize powerserie AGP (l=1) for the problem A*H_x + B*H_p.
        """
        theta_1 = qutip.commutator(H_i, qutip.commutator(H_i,H_p))
        theta_2 = qutip.commutator(H_p, qutip.commutator(H_i,H_p))

        # define symbols
        alpha1 = sympy.Symbol("\\alpha_1", commutative=True)
        beta = sympy.Symbol("\\beta", commutative=True) # = A Bdot - B Adot
        
        A = sympy.Symbol('A', commutative=True)
        B = sympy.Symbol('B', commutative=True)
        Adot = sympy.Symbol("C", commutative=True)
        Bdot = sympy.Symbol("D", commutative=True)

        # define non commuting symbols
        Hi = sympy.Symbol("H_i", commutative=False)
        Hp = sympy.Symbol("H_p", commutative=False)
        t1 = sympy.Symbol("\\theta_1", commutative=False)
        t2 = sympy.Symbol("\\theta_2", commutative=False)

        # symbolic math
        self.G = Adot*Hi + Bdot*Hp + alpha1*beta*(A*t1 + B*t2)

        # associate non commuting symbols to objects
        self.associations = {"H_i" : H_i, "H_p" : H_p, "\\theta_1" : theta_1, "\\theta_2": theta_2}

        self.ansatz_coefficients = [ alpha1 ]
        self.symbols = [A,B,Adot,Bdot]
        self.suggested_subs = [ [beta, A*Bdot - B*Adot] ]

    def variational_min(self, subs:bool=True):
        if subs:  ss = self.suggested_subs
        else:     ss = None
        return varmin.varmin(self.G, self.associations, self.ansatz_coefficients, self.symbols, subs=ss)




class manual_l2_gauge:
    def __init__(self, H_i, H_p):
        """
        Numerically optimize powerserie AGP (l=2) for the problem A*H_x + B*H_p.
        """
        
        theta_1 = qutip.commutator(H_i, qutip.commutator(H_i,H_p))
        theta_2 = qutip.commutator(H_p, qutip.commutator(H_i,H_p))
        
        eta_1 = qutip.commutator(H_i, qutip.commutator(H_i,theta_1))
        eta_2 = qutip.commutator(H_i, qutip.commutator(H_i,theta_2))
        eta_3 = qutip.commutator(H_i, qutip.commutator(H_p,theta_1))
        eta_4 = qutip.commutator(H_i, qutip.commutator(H_p,theta_2))
        eta_5 = qutip.commutator(H_p, qutip.commutator(H_i,theta_1))
        eta_6 = qutip.commutator(H_p, qutip.commutator(H_i,theta_2))
        eta_7 = qutip.commutator(H_p, qutip.commutator(H_p,theta_1))
        eta_8 = qutip.commutator(H_p, qutip.commutator(H_p,theta_2))

        # define symbols
        alpha1 = sympy.Symbol("\\alpha_1", commutative=True)
        alpha2 = sympy.Symbol("\\alpha_2", commutative=True)
        beta = sympy.Symbol("\\beta", commutative=True) # = A Bdot - B Adot
        
        A = sympy.Symbol('A', commutative=True)
        B = sympy.Symbol('B', commutative=True)
        Adot = sympy.Symbol("\dot{A}", commutative=True)
        Bdot = sympy.Symbol("\dot{B}", commutative=True)

        # define non commuting symbols
        Hi = sympy.Symbol("H_i", commutative=False)
        Hp = sympy.Symbol("H_p", commutative=False)
        t1 = sympy.Symbol("\\theta_1", commutative=False)
        t2 = sympy.Symbol("\\theta_2", commutative=False)
        n1 = sympy.Symbol("\\eta_1", commutative=False)
        n2 = sympy.Symbol("\\eta_2", commutative=False)
        n3 = sympy.Symbol("\\eta_3", commutative=False)
        n4 = sympy.Symbol("\\eta_4", commutative=False)
        n5 = sympy.Symbol("\\eta_5", commutative=False)
        n6 = sympy.Symbol("\\eta_6", commutative=False)
        n7 = sympy.Symbol("\\eta_7", commutative=False)
        n8 = sympy.Symbol("\\eta_8", commutative=False)

        # symbolic math
        self.G = Adot*Hi + Bdot*Hp + alpha1*beta*(A*t1 + B*t2) +\
            alpha2*beta*( (A**3)*n1 + A*A*B*n2 + B*A*A*n3 + B*B*A*n4+\
                A*A*B*n5 + A*B*B*n6 + B*B*A*n7 + (B**3)*n8 
            )
        
        # associate non commuting symbols to objects
        self.associations = {
            "H_i" : H_i, "H_p" : H_p, 
            "\\theta_1" : theta_1, "\\theta_2": theta_2,
            "\\eta_1" : eta_1, "\\eta_2" : eta_2, "\\eta_3" : eta_3,
            "\\eta_4" : eta_4, "\\eta_5" : eta_5, "\\eta_6" : eta_6,
            "\\eta_7" : eta_7, "\\eta_8" : eta_8
        }
        
        self.ansatz_coefficients = [ alpha1 ]
        self.symbols = [A,B,Adot,Bdot]
        self.suggested_subs = [ [beta, A*Bdot - B*Adot] ]


    def variational_min(self, subs:bool=True):
        if subs:  ss = self.suggested_subs
        else:     ss = None
        return varmin.varmin(self.G, self.associations, self.ansatz_coefficients, self.symbols, subs=ss)
