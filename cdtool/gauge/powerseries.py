
import sympy
import qutip
import itertools
import math

import cdtool.gauge.varmin as varmin



def compute_nested_commutator(operators:list, op_indexes:list, order:int, cache:dict) -> dict:

    assert order > 0, 'unexpected order number'
    output : dict = {}

    for comb in itertools.product(op_indexes, repeat=order+1):

        if len(comb) > 2:  other = cache[ comb[1:] ]
        else:              other = operators[ comb[1] ]
        val = qutip.commutator( operators[comb[0]], other )

        if val.data.nnz == 0:  val = 0

        output[comb] = val

    return output


def make_symbolic_coeff(tpl, symb, symb_deriv):
    lst = [ symb_deriv[ tpl[-1] ] ] + [ symb[ kk ] for kk in tpl[:-1] ]
    return math.prod( lst )




class powerseries_gauge:

    def __init__(self, H_op:list, l:int) -> None:
        N_ham :int = len(H_op)

        H_indexes = [ ii for ii in range(N_ham) ]
        H_symbols = [ sympy.Symbol(f"H_{ii}", commutative=False) for ii in H_indexes ]
        
        self.H_coeffs = [ sympy.Symbol(f"C_{ii}", commutative=True, real=True) for ii in range(N_ham) ]
        self.H_dot_coeffs = [ sympy.Symbol(f"D_{ii}", commutative=True, real=True) for ii in range(N_ham) ]

        alphas = [ sympy.Symbol(f"\\alpha_{ii+1}", commutative=True, real=True) for ii in range(l) ]

        self.associations :dict = {}
        self.plhold_count :int = 0

        G = 0
        for dcoeff, symb, op in zip(self.H_dot_coeffs, H_symbols, H_op):
            G += dcoeff * symb
            self.associations[ str(symb.name) ] = op

        drive_operators = []
        commutators = {}
        for ii in range(1, 2*l+1):
            commutators = compute_nested_commutator(H_op, H_indexes, order=ii, cache=commutators)

            if ii%2 == 1:
                #print(f'retrieving l={(ii+1)//2} driver')
                drive_operators.append( commutators )
            else:
                #print(f'assembling l={(ii)//2} [H,A]')
                addG = self.__assemble_term( commutators )
                G += alphas[(ii)//2 - 1]*addG # TODO the i?

        self.G = G       
        self.drive_operators = drive_operators
        self.ansatz_coefficients = alphas
        self.symbols = self.H_coeffs + self.H_dot_coeffs


    def __assemble_term(self, commutators:dict):
        """Assemble the symbolic term [H, A(commutators)] for commutators encoded dict."""
        partG = 0

        for el in commutators.keys():
            operator = commutators[el]
            if isinstance(operator, int):
                if operator == 0:
                    continue

            # create placeholder symbol and associate to commutator
            plhold = sympy.Symbol(f"\\gamma_{self.plhold_count}", commutative=False)
            self.plhold_count += 1

            self.associations[ str(plhold.name) ] = operator

            coeff = make_symbolic_coeff(el, self.H_coeffs, self.H_dot_coeffs)
            partG += coeff * plhold

        return partG


    def variational_min(self):
        return varmin.varmin(self.G, self.associations, self.ansatz_coefficients, self.symbols)
    
    def get_drivers(self, l:int) -> list:
        dd = self.drive_operators[l-1]
        output = []
        for el in dd.keys():
            operator = dd[el]
            if isinstance(operator, int):
                if operator == 0: continue
            output.append( [make_symbolic_coeff(el, self.H_coeffs, self.H_dot_coeffs), operator] )
        return output
    
    def make_driving(self, drive_input:callable):
        drivers = self.get_drivers(l=1)
        