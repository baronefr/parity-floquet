#  tools for numerical CD simulation

import math
import sympy
import qutip


# %% routines for symbolical manipulation of G

def separate_bycommutative(term):
    """Given a term, separates commutative from non-commutative parts."""
    comm = []
    noncomm = []
    for factor in term.args:
        if factor.is_commutative:
            comm.append(factor)
        else:
            noncomm.append(factor)
    return comm, noncomm

def handle_powers(ll):
    """If any element in ll is a power, it is separated in the output list."""
    ret = []
    for element in ll:
        if isinstance(element, sympy.Pow):
            for ii in range(element.exp):
                ret.append(element.base)
        else:
            ret.append(element)
    return ret

def compute_trace(list_of_terms, associations):
    """Takes the associated elements from association dictionary, multiplies and returns the trace."""
    mults = [ associations[ ii.name ] for ii in list_of_terms ]
    return (math.prod(mults)).tr()

def trace_S(S, associations):
    tracedS = 0

    for term in S.args:
        if isinstance(term, sympy.Mul):
            comm, noncomm = separate_bycommutative(term)
            noncomm = handle_powers( noncomm )
            #print('term', term, ' -> ', noncomm)
            coeff = math.prod(comm)
            
            tracedS += coeff*compute_trace(noncomm, associations)
        else:
            raise Exception('unknown')
    return tracedS



# %% SYMBOLIC SOLVERS

def Floquet_l1_AB(H_i, H_p, lambdify_sequence :str = 'coeff'):
    """
    Numerically optimize AGP for Floquet (l=1), problem A*H_x + B*H_p.
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

    # associate non commuting symbols to objects
    associations = {"H_i" : H_i, "H_p" : H_p, "\\theta_1" : theta_1, "\\theta_2": theta_2}
    
    # symbolic math
    G = Adot*Hi + Bdot*Hp + alpha1*beta*(A*t1 + B*t2)
    S = (G**2).expand()

    tracedS = trace_S(S, associations)
    tracedS = tracedS.subs(beta, A*Bdot - B*Adot)
    
    # minimize over the coefficients alpha_i
    #sympy.collect( sympy.diff( tracedS, alpha1 ).expand(), alpha1)
    var_to_solve = [ alpha1 ]
    diff_eqs = [ sympy.Eq( sympy.diff( tracedS, vv).expand(), 0 ) for vv in var_to_solve ]
    sol = sympy.solve(diff_eqs, var_to_solve)
    sol[alpha1]
    
    # lambdify the solutions
    if lambdify_sequence == 'coeff':
        alpha_1_lambda = sympy.lambdify([A,B], sol[alpha1], modules='numpy')
    elif lambdify_sequence == 'full':
        alpha_1_lambda = sympy.lambdify([A,B,Adot,Bdot], sol[alpha1], modules='numpy')
    else:
        raise Exception('unknown lambdify args')
    
    return tracedS, [alpha_1_lambda]



def Floquet_l1_ABC(H_x, H_z, H_c, lambdify_sequence :str = None):
    """
    Numerically optimize AGP for Floquet (l=1), problem A*H_x + B*H_z + C*H_c.
    """
    
    # compute numerical commutators
    tmp1 = qutip.commutator(H_x,H_z)
    tmp2 = qutip.commutator(H_x,H_c)
    
    theta_1 = qutip.commutator(H_x, tmp1)
    theta_2 = qutip.commutator(H_z, tmp1)
    theta_3 = qutip.commutator(H_c, tmp1)
    theta_4 = qutip.commutator(H_x, tmp2)
    theta_5 = qutip.commutator(H_z, tmp2)
    theta_6 = qutip.commutator(H_c, tmp2)
    
    # define symbols
    alpha1 = sympy.Symbol("\\alpha_1", commutative=True)
    b1 = sympy.Symbol("\\beta_1", commutative=True) # = A Bdot - B Adot
    b2 = sympy.Symbol("\\beta_2", commutative=True) # = A Cdot - C Adot
    
    A = sympy.Symbol('A', commutative=True)
    B = sympy.Symbol('B', commutative=True)
    C = sympy.Symbol('C', commutative=True)
    Adot = sympy.Symbol("\dot{A}", commutative=True)
    Bdot = sympy.Symbol("\dot{B}", commutative=True)
    Cdot = sympy.Symbol("\dot{C}", commutative=True)
    
    # define non commuting symbols
    Hx = sympy.Symbol("H_x", commutative=False)
    Hz = sympy.Symbol("H_z", commutative=False)
    Hc = sympy.Symbol("H_c", commutative=False)
    t1 = sympy.Symbol("\\theta_1", commutative=False)
    t2 = sympy.Symbol("\\theta_2", commutative=False)
    t3 = sympy.Symbol("\\theta_3", commutative=False)
    t4 = sympy.Symbol("\\theta_4", commutative=False)
    t5 = sympy.Symbol("\\theta_5", commutative=False)
    t6 = sympy.Symbol("\\theta_6", commutative=False)
    
    # associate non commuting symbols to numerical objects
    associations = {"H_x" : H_x, "H_z" : H_z, "H_c" : H_c, 
        "\\theta_1" : theta_1, "\\theta_2" : theta_2, "\\theta_3" : theta_3,
        "\\theta_4" : theta_4, "\\theta_5" : theta_5, "\\theta_6" : theta_6
    }
    
    # symbolic math
    G = Adot*Hx + Bdot*Hz + Cdot*Hc + \
        alpha1*b1*(A*t1 + B*t2 + C*t3) +\
        alpha1*b2*(A*t4 + B*t5 + C*t6)
    S = (G**2).expand()

    tracedS = trace_S(S, associations)
    tracedS = tracedS.subs(b1, A*Bdot - B*Adot)
    tracedS = tracedS.subs(b2, A*Cdot - C*Adot)
    
    # minimize over the coefficients alpha_i
    #sympy.collect( sympy.diff( tracedS, alpha1 ).expand(), alpha1)
    var_to_solve = [ alpha1 ]
    diff_eqs = [ sympy.Eq( sympy.diff( tracedS, vv).expand(), 0 ) for vv in var_to_solve ]
    sol = sympy.solve(diff_eqs, var_to_solve)
    sol[alpha1]
    
    # lambdify the solutions
    alpha_1_lambda = sympy.lambdify([A,B,C,Adot,Bdot,Cdot], sol[alpha1], modules='numpy')
    return tracedS, [alpha_1_lambda]



def Floquet_l2_AB(H_i, H_p, lambdify_sequence : str = 'coeff'):
    """
    Numerically optimize AGP for Floquet (l=2), problem A*H_x + B*H_p.
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

    # associate non commuting symbols to objects
    associations = {
        "H_i" : H_i, "H_p" : H_p, 
        "\\theta_1" : theta_1, "\\theta_2": theta_2,
        "\\eta_1" : eta_1, "\\eta_2" : eta_2, "\\eta_3" : eta_3,
        "\\eta_4" : eta_4, "\\eta_5" : eta_5, "\\eta_6" : eta_6,
        "\\eta_7" : eta_7, "\\eta_8" : eta_8
    }
    
    # symbolic math
    G = Adot*Hi + Bdot*Hp + alpha1*beta*(A*t1 + B*t2) +\
        alpha2*beta*( (A**3)*n1 + A*A*B*n2 + B*A*A*n3 + B*B*A*n4+\
            A*A*B*n5 + A*B*B*n6 + B*B*A*n7 + (B**3)*n8 
        )
    S = (G**2).expand()

    tracedS = trace_S(S, associations)
    tracedS = tracedS.subs(beta, A*Bdot - B*Adot)
    
    # minimize over the coefficients alpha_i
    #sympy.collect( sympy.diff( tracedS, alpha1 ).expand(), alpha1)
    var_to_solve = [ alpha1, alpha2 ]
    diff_eqs = [ sympy.Eq( sympy.diff( tracedS, vv).expand(), 0 ) for vv in var_to_solve ]
    sol = sympy.solve(diff_eqs, var_to_solve)
    
    # lambdify the solutions
    if lambdify_sequence == 'coeff':
        alpha_1_lambda = sympy.lambdify([A,B], sol[alpha1], modules='numpy')
        alpha_2_lambda = sympy.lambdify([A,B], sol[alpha2], modules='numpy')
    elif lambdify_sequence == 'full':
        alpha_1_lambda = sympy.lambdify([A,B,Adot,Bdot], sol[alpha1], modules='numpy')
        alpha_2_lambda = sympy.lambdify([A,B,Adot,Bdot], sol[alpha2], modules='numpy')
    else:
        raise Exception('unknown lambdify args')
    
    return tracedS, [alpha_1_lambda, alpha_2_lambda]






# %% NUMERICAL CD hamiltonian computation

def CD_l2_agp(H_i, H_p):
    
    # >> first order gauge
    A_gauge_1 = qutip.commutator(H_i,H_p)
    
    # needed for second order stuff
    theta_1 = qutip.commutator(H_i, A_gauge_1)
    theta_2 = qutip.commutator(H_p, A_gauge_1)
    
    # >> second order gauge
    A_gauge_2a = qutip.commutator(H_i,theta_1)
    A_gauge_2b = qutip.commutator(H_i,theta_2)
    A_gauge_2c = qutip.commutator(H_p,theta_1)
    A_gauge_2d = qutip.commutator(H_p,theta_2)
    
    return A_gauge_1, [A_gauge_2a, A_gauge_2b, A_gauge_2c, A_gauge_2d]






# %% general CD driving

def auto_ansatz(H_i, H_p, ansatz, lambdify_sequence :str = 'full'):
    """Minimize the coefficient for a general ansatz."""
    raise Exception('deprecated')
    zeta_1 = qutip.commutator(ansatz, H_i)
    zeta_2 = qutip.commutator(ansatz, H_p)

    # define symbols
    alpha1 = sympy.Symbol("\\alpha_1", commutative=True)

    A = sympy.Symbol('A', commutative=True)
    B = sympy.Symbol('B', commutative=True)
    Adot = sympy.Symbol("\dot{A}", commutative=True)
    Bdot = sympy.Symbol("\dot{B}", commutative=True)

    # define non commuting symbols
    Hi = sympy.Symbol("H_i", commutative=False)
    Hp = sympy.Symbol("H_p", commutative=False)
    z1 = sympy.Symbol("\\zeta_1", commutative=False)
    z2 = sympy.Symbol("\\zeta_2", commutative=False)

    # associate non commuting symbols to objects
    associations = {"H_i" : H_i, "H_p" : H_p, "\\zeta_1" : zeta_1, "\\zeta_2": zeta_2}
    
    # symbolic math
    G = (Adot*Hi + Bdot*Hp) + 1j*alpha1*( A*z1 + B*z2 )
    S = (G**2).expand()

    tracedS = trace_S(S, associations)
    
    # minimize over the coefficients alpha_i
    var_to_solve = [ alpha1 ]
    diff_eqs = [ sympy.Eq( sympy.diff( tracedS, vv).expand(), 0 ) for vv in var_to_solve ]
    sol = sympy.solve(diff_eqs, var_to_solve)
    
    # lambdify the solutions
    if lambdify_sequence == 'coeff':
        alpha_1_lambda = sympy.lambdify([A,B], sol[alpha1], modules='numpy')
    elif lambdify_sequence == 'full':
        alpha_1_lambda = sympy.lambdify([A,B,Adot,Bdot], sol[alpha1], modules='numpy')
    else:
        raise Exception('unknown lambdify args')
    
    return sol[alpha1], [alpha_1_lambda]



def auto_ansatz_morecoeffs(H_i, H_p, ansatz_op:list, lambdify_sequence :str = 'full'):
    """Minimize the N coefficients for a general ansatz of N terms."""
    raise Exception('deprecated')
    N_op = len(ansatz_op)

    # define symbols
    alphas = [ sympy.Symbol(f"\\alpha_{ii}", commutative=True) for ii in range(N_op) ]
    
    gammas_op = [ qutip.commutator(aa, H_i) for aa in ansatz_op ]
    zetas_op = [ qutip.commutator(aa, H_p) for aa in ansatz_op ]
    
    A = sympy.Symbol('A', commutative=True)
    B = sympy.Symbol('B', commutative=True)
    Adot = sympy.Symbol("\dot{A}", commutative=True)
    Bdot = sympy.Symbol("\dot{B}", commutative=True)

    # define non commuting symbols
    Hi = sympy.Symbol("H_i", commutative=False)
    Hp = sympy.Symbol("H_p", commutative=False)
    gammas_symb = [ sympy.Symbol(f"\\gamma_{ii}", commutative=False) for ii in range(N_op) ]
    zetas_symb = [ sympy.Symbol(f"\\zeta_{ii}", commutative=False) for ii in range(N_op) ]
    
    # associate non commuting symbols to objects
    associations = {"H_i" : H_i, "H_p" : H_p }
    for ss, op in zip(gammas_symb, gammas_op):
        associations[ str(ss.name) ] = op
    for ss, op in zip(zetas_symb, zetas_op):
        associations[ str(ss.name) ] = op
    
    # symbolic math
    G = Adot*Hi + Bdot*Hp
    for asy, comm in zip(alphas, gammas_symb):
        G += 1j*A*asy*comm
    for asy, comm in zip(alphas, zetas_symb):
        G += 1j*B*asy*comm
    
    # note:  ansatz is supposed to have a 1j coefficient!
    S = (G**2).expand()
    tracedS = trace_S(S, associations)
    
    # minimize over the coefficients alpha_i
    var_to_solve = alphas
    diff_eqs = [ sympy.Eq( sympy.diff( tracedS, vv).expand(), 0 ) for vv in var_to_solve ]
    sol = sympy.solve(diff_eqs, var_to_solve)
    
    # lambdify the solutions
    if lambdify_sequence == 'coeff':
        alphas_lambda = [ sympy.lambdify([A,B], sol[aa], modules='numpy') for aa in alphas ]
    elif lambdify_sequence == 'full':
        alphas_lambda = [ sympy.lambdify([A,B,Adot,Bdot], sol[aa], modules='numpy') for aa in alphas ]
    else:
        raise Exception('unknown lambdify args')
    
    return sol, alphas_lambda

# %%


def auto_ansatz_full(H_op:list, ansatz_op:list, lambdify_sequence :str = 'full'):
    """Minimize the N coefficients for a general ansatz of N terms."""
    
    return_sym = {}
    
    N_ham = len(H_op)
    N_op = len(ansatz_op)

    # define symbols
    alphas = [ sympy.Symbol(f"\\alpha_{ii}", commutative=True) for ii in range(N_op) ]
    Ls = [ sympy.Symbol(f"L_{ii}", commutative=True) for ii in range(N_ham) ]
    Ldots = [ sympy.Symbol("\dot{L}_" + f"{ii}", commutative=True) for ii in range(N_ham) ]
    
    G = 0
    associations = {}
    commutator_placeholder = 0
    for idx, HH in enumerate(H_op):
        # H term symbol
        symbol_name = f"H_{idx}"
        thisH = sympy.Symbol(symbol_name, commutative=False)
        associations[symbol_name] = HH
        
        # add derivative
        G += Ldots[idx] * thisH
        
        # add commutators
        for alph, aa in zip(alphas, ansatz_op):
            placeholder_coeff = sympy.Symbol(f"\\gamma_{commutator_placeholder}", commutative=False)
            associations[ str(placeholder_coeff.name) ] = qutip.commutator(aa, HH)
            G += 1j * Ls[idx] * alph * placeholder_coeff
            commutator_placeholder += 1
    
    return_sym['G'] = return_sym
    
    # note:  ansatz is supposed to have a 1j coefficient!
    S = (G**2).expand()
    tracedS = trace_S(S, associations)
    
    return_sym['trS'] = tracedS
    
    # minimize over the coefficients alpha_i
    var_to_solve = alphas
    diff_eqs = [ sympy.Eq( sympy.diff( tracedS, vv).expand(), 0 ) for vv in var_to_solve ]
    sol = sympy.solve(diff_eqs, var_to_solve)
    
    sol_names = {}
    for aa in alphas:  sol_names[aa.name] = sol[aa]
    return_sym['sol'] = sol_names
    
    # lambdify the solutions
    if lambdify_sequence == 'coeff':
        alphas_lambda = [ sympy.lambdify(Ls, sol[aa], modules='numpy') for aa in alphas ]
    elif lambdify_sequence == 'full':
        alphas_lambda = [ sympy.lambdify(Ls+Ldots, sol[aa], modules='numpy') for aa in alphas ]
    else:
        raise Exception('unknown lambdify args')
    
    return return_sym, alphas_lambda


