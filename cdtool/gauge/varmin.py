
import sympy
import math

# %% routines for symbolical manipulation of G

def separate_bycommutative(term):
    """Given an expression, separates commutative from non-commutative factors."""
    comm = []
    noncomm = []
    for factor in term.args:
        if factor.is_commutative:
            comm.append(factor)
        else:
            noncomm.append(factor)
    return comm, noncomm

def handle_powers(ll):
    """Map any  [t^n]  to  [t,t,...] (n times)."""
    ret = []
    for element in ll:
        if isinstance(element, sympy.Pow):
            for ii in range(element.exp):
                ret.append(element.base)
        else:
            ret.append(element)
    return ret

def compute_trace(list_of_terms:list, associations:dict):
    """Takes the associated elements from association dictionary, multiplies and returns the trace."""
    mults = [ associations[ ii.name ] for ii in list_of_terms ]
    return (math.prod(mults)).tr()

def trace_S(S, associations:dict):
    """Compute the trace of S using the operator associations."""
    tracedS = 0

    for term in S.args:
        if isinstance(term, sympy.Mul):
            comm, noncomm = separate_bycommutative(term)
            noncomm = handle_powers( noncomm )
            coeff = math.prod(comm)
            tracedS += coeff*compute_trace(noncomm, associations)
        else:
            raise Exception(f'unknown instance of term {term}')
    return tracedS






class varmin:

    def __init__(self, G, associations:dict, minvar:list, symbols:list, subs:list=None):

        S = (G**2).expand()
        tracedS = trace_S(S, associations)

        if subs is not None:
            # NOTE: subs is expected to be a list of 2-element lists:
            #   [   [ symbol, value_to_substitute],
            #       [ symbol, value_to_substitute]   ]
            for ss in subs:
                tracedS = tracedS.subs(ss[0], ss[1])

        # minimize over the coefficients minvar
        equations = [ sympy.Eq( sympy.diff( tracedS, vv).expand(), 0 ) for vv in minvar ]

        patch_l2 = len(minvar) == 2

        if patch_l2:
            print('executing patched l=2')
            sol_a1 = sympy.solve( equations[0], minvar[0])
            expr_a2 = equations[1].subs(minvar[0], sol_a1[0])
            sol_a2 = sympy.solve( expr_a2, minvar[1])
            sol_a1 = sol_a1[0].subs(minvar[1], sol_a2[0])
            sol = { minvar[0]: sol_a1, minvar[1]: sol_a2[0] }
        else:
            sol = sympy.solve(equations, minvar)

        # arrange solutions by name instead of symbol
        sol_byname = {}
        for aa in minvar:  sol_byname[aa.name] = sol[aa]

        self.G = G
        self.S = S
        self.trS = tracedS
        self.equations = equations
        self.solution = sol
        self.solution_byname = sol_byname

        self.minvar = minvar
        self.symbols = symbols

    def lambdify(self, symbols=None, modules:str='numpy') -> list:
        if symbols is None: symb = self.symbols
        else:               symb = symbols
        return [ sympy.lambdify(symb, self.solution[aa], modules=modules) for aa in self.minvar ]
    