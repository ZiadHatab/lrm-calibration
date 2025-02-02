"""
@author: Ziad (zi.hatab@gmail.com, https://github.com/ZiadHatab)

Implementation of the line-reflect-match (LRM) calibration method.

There are numerous references available in the literature. Below are a few.
The implementation here is based on my interpretation of the method with similar
constraints typically imposed in TRL calibration.
Essentially, it is similar to having an infinitely long transmission line with 
losses (which is the match standard).

Constraints:
    - The match standard must be symmetric
    - If non-zero length is used, the line must have characteristic impedance 
      identical to the match standard
    - The reflect standard must be symmetric but can be unknown. A rough estimate 
      is needed to resolve the ± sign ambiguity
    - If a non-zero length line is used, gamma*length is required to describe 
      the line (i.e., the line must be fully defined)

References:
Note: Each reference implements the method differently. Some allow any fully defined
transmission device and load, as long as they are fully defined.

[1] H.-J. Eul and B. Schiek, "Thru-Match-Reflect: One Result of a Rigorous Theory 
    for De-Embedding and Network Analyzer Calibration," 18th European Microwave 
    Conference, Stockholm, Sweden, 1988, pp. 909-914. 
    doi: https://doi.org/10.1109/EUMA.1988.333924

[2] M. Wollensack, J. Hoffmann, D. Stalder, J. Ruefenacht and M. Zeier, 
    "VNA tools II: Calibrations involving eigenvalue problems," 89th ARFTG 
    Microwave Measurement Conference (ARFTG), Honololu, HI, USA, 2017, pp. 1-4. 
    doi: https://doi.org/10.1109/ARFTG.2017.8000832

[3] W. Zhao et al., "A Unified Approach for Reformulations of LRM/LRMM/LRRM 
    Calibration Algorithms Based on the T-Matrix Representation," Applied Sciences, 
    vol. 7, no. 9, p. 866, 2017. 
    doi: https://doi.org/10.3390/app7090866
"""

import numpy as np
import skrf as rf

# constants
c0 = 299792458
P = np.array([[0,1], [1, 0]])

def s2t(S):
    # convert S-parameters to T-parameters
    T = S.copy()
    T[0,0] = -(S[0,0]*S[1,1]-S[0,1]*S[1,0])
    T[0,1] = S[0,0]
    T[1,0] = -S[1,1]
    T[1,1] = 1
    return T/S[1,0]

def t2s(T):
    # convert T-parameters to S-parameters
    S = T.copy()
    S[0,0] = T[0,1]
    S[0,1] = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    S[1,0] = 1
    S[1,1] = -T[1,0]
    return S/T[1,1]

def mobius(T, z):
    # mobius transformation
    t11, t12, t21, t22 = T[0,0], T[0,1], T[1,0], T[1,1]
    return (t11*z + t12)/(t21*z + t22)

def mobius_inv(T, z):
    # inverse mobius transformation
    t11, t12, t21, t22 = T[0,0], T[0,1], T[1,0], T[1,1]
    return (t12 - t22*z)/(t21*z - t11)

def sqrt_unwrapped(z):
    # take the square root of a complex number with unwrapped phase.
    return np.sqrt(abs(z))*np.exp(0.5*1j*np.unwrap(np.angle(z)))

def correct_switch_term(S, GF, GR):
    '''
    correct switch terms of measured S-parameters at a single frequency point
    GF: forward (sourced by port-1)
    GR: reverse (sourced by port-2)
    '''
    S_new = S.copy()
    S_new[0,0] = (S[0,0]-S[0,1]*S[1,0]*GF)/(1-S[0,1]*S[1,0]*GF*GR)
    S_new[0,1] = (S[0,1]-S[0,0]*S[0,1]*GR)/(1-S[0,1]*S[1,0]*GF*GR)
    S_new[1,0] = (S[1,0]-S[1,1]*S[1,0]*GF)/(1-S[0,1]*S[1,0]*GF*GR)
    S_new[1,1] = (S[1,1]-S[0,1]*S[1,0]*GR)/(1-S[0,1]*S[1,0]*GF*GR)
    return S_new

def solve_at_one_freq(line_S, reflect_S, match_S, reflect_est, gamma_length=0):
    """Calculate LRM error boxes at a single frequency point.

    Parameters
    ----------
    line_S : 2x2 complex array
        measured S-parameters of the line standard 
    reflect_S : 2x2 complex array
        measured S-parameters of the symmetric reflect standard
    match_S : 2x2 complex array
        measured S-parameters of the match standard
    reflect_est : complex
        estimated reflection coefficient of the symmetric reflect standard
    gamma_length : complex, optional
        product of propagation constant and physical length of the line standard.
        Default is 0 (zero-length line).
    
    Returns
    -------
    A, B : 2x2 complex arrays
        Left and right error boxes
    k : complex
        7th error term (transmission error term)
    """
    # extract relevant s-parameters
    match_S11 = match_S[0,0]
    match_S22 = match_S[1,1]
    reflect_S11 = reflect_S[0,0]
    reflect_S22 = reflect_S[1,1]
    line_T = s2t(line_S)

    # calculate normalized error terms from match and line standards
    a12 = match_S11
    b21 = -match_S22
    a21_a11 = mobius(P@line_T@P, match_S22)
    b12_b11 = -mobius_inv(line_T, match_S11)

    # build normalized error boxes
    A_ = np.array([[1,a12],[a21_a11,1]])
    A_inv = np.linalg.inv(A_)
    B_ = np.array([[1,b12_b11],[b21,1]])
    B_inv = np.linalg.inv(B_)
    
    # solve for product term a11b11 from line measurements (k is re-solved at end)
    ka11b11,_,_,k = (A_inv@line_T@B_inv).flatten()
    a11b11 = ka11b11/k  
    a11b11 = a11b11*np.exp(2*gamma_length)  # this to remove the effect of the line length

    if np.isnan(reflect_S[0,0]):
        # this is used if no reflect measurement available
        # will give inaccurate s11 calibration, but s21 is unaffected.
        a11 = np.sqrt(a11b11) 
        b11 = a11
    else:
        a11Gamma = mobius(A_inv, reflect_S11)
        b11Gamma = mobius(P@B_@P, reflect_S22)
        a11_b11 = a11Gamma/b11Gamma  # ratio term a11/b11
        a11 = np.sqrt(a11_b11*a11b11)
        reflect_meas = a11Gamma/a11  # measured reflect standard with sign ambiguity
        # resolve sign ambiguity
        a11 = a11 if abs(reflect_meas - reflect_est) < abs(reflect_meas + reflect_est) else -a11
        b11 = a11b11/a11

    # denormalize error boxes
    A = A_@np.diag([a11, 1])
    B = np.diag([b11, 1])@B_
    
    # solve for 7th error term k using reciprocity
    k = np.sqrt(np.linalg.det(line_T)/np.linalg.det(A)/np.linalg.det(B))
    err1 = abs( k*A@np.diag([np.exp(-gamma_length), np.exp(gamma_length)])@B - line_T ).sum()
    err2 = abs( k*A@np.diag([np.exp(-gamma_length), np.exp(gamma_length)])@B + line_T ).sum()
    k  = k if err1 < err2 else -k

    return A, B, k

class LRM:
    """
    Line-Reflect-Match (LRM) calibration method.

    Parameters
    ----------
    line: 2-port skrf network
        Line standard measurement
    match: 2-port skrf network  
        Match standard measurement
    reflect: 2-port skrf network, optional
        Reflect standard measurement (see Notes)
    reflect_est: complex scalar or array, optional 
        Estimated reflection coefficient (see Notes)
    gamma_length: complex scalar or array, optional
        Product of propagation constant and line length (needed for non-zero length line)
    switch_term: list of 1-port skrf networks, optional
        Switch terms [forward, reverse]

    Notes
    -----
    - At minimum, the line and match standards are required for calibration
    - The reflect standard is optional but required for accurate S11 calibration 
    - If the reflect standard is used, reflect_est must be provided (can be frequency dependent)
    - For non-zero length lines, gamma_length must be specified (can be frequency dependent)
    - All standard measurements must be performed at identical frequency points
    """
    def __init__(self, line, match, reflect=None, reflect_est=None, gamma_length=0, switch_term=None):
        self.f  = line.frequency.f
        self.line_S = line.s.squeeze()    # s-parameters of the line standard
        self.match_S = match.s.squeeze()  # s-parameters of the match standard
        if reflect is not None:
            self.reflect_S = reflect.s.squeeze()
            self.reflect_est = reflect_est*np.ones_like(self.f)
        else:
            self.reflect_S = np.nan*np.ones_like(self.line_S)
            self.reflect_est = np.nan*np.ones_like(self.f)
        
        self.gamma_length = gamma_length*np.ones_like(self.f)

        if switch_term is not None:
            self.switch_term = np.array([x.s.squeeze() for x in switch_term])
        else:
            self.switch_term = np.array([self.f*0 for x in range(2)])
    
    def run(self):
        # This runs the calibration procedure
        A = []
        B = []
        k = []
        print('\nLRM is running...')
        for inx, f in enumerate(self.f):
            line_S  = self.line_S[inx]
            match_s = self.match_S[inx]
            reflect_S = self.reflect_S[inx]
            reflect_est = self.reflect_est[inx]
            gamma_length = self.gamma_length[inx]
            
            # correct for switch term
            sw = self.switch_term[:,inx]
            line_S = correct_switch_term(line_S, sw[0], sw[1]) 

            A_, B_, k_ = solve_at_one_freq(line_S, reflect_S, match_s, reflect_est, gamma_length)
            A.append(A_)
            B.append(B_)
            k.append(k_)
            print(f'Frequency: {f*1e-9:.2f} GHz ... DONE!')

        #self.A  = np.array(A)
        #self.B  = np.array(B)
        self.X  = np.array([np.kron(b.T, a) for a,b in zip(A, B)])
        self.k  = np.array(k)
        self.error_coef() # compute the 12 error terms
    
    def apply_cal(self, NW, left=True):
        '''
        Apply calibration to a 1-port or 2-port network.
        NW:   the network to be calibrated (1- or 2-port).
        left: boolean: define which port to use when 1-port network is given. If left is True, left port is used; otherwise right port is used.
        '''
        nports = np.sqrt(len(NW.port_tuples)).astype('int') # number of ports
        # if 1-port, convert to 2-port (later convert back to 1-port)
        if nports < 2:
            NW = rf.two_port_reflect(NW)

        # apply cal
        P = np.array([[1,0,0,0], [0, 0,1,0], [0,1, 0,0], [0,0,0,1]])  # permute matrix
        q = np.array([[0,1],[1,0]])
        S_cal = []
        for x,k,s,sw in zip(self.X, self.k, NW.s, self.switch_term.T):
            s    = correct_switch_term(s, sw[0], sw[1]) if np.any(sw) else s  # switch term correction
            """
            Correction based on the bilinear fractional transformation.
            R. A. Speciale, "Projective Matrix Transformations in Microwave Network Theory," 
            1981 IEEE MTT-S International Microwave Symposium Digest, Los Angeles, CA, USA, 
            1981, pp. 510-512, doi: 10.1109/MWSYM.1981.1129979
            """
            A = np.array([[x[2,2],x[2,3]],[x[3,2],1]])
            B = np.array([[x[1,1],x[3,1]],[x[1,3],1]])
            Zero = A*0
            E = P.T@np.block([[A*k, Zero],[Zero, q@np.linalg.inv(B)@q]])@P
            E11,E12,E21,E22 = E[:2,:2], E[:2,2:], E[2:,:2], E[2:,2:]
            S_cal.append( np.linalg.inv(s@E21-E11)@(E12-s@E22) )
        S_cal = np.array(S_cal).squeeze()
        freq  = NW.frequency
        
        # revert to 1-port device if the input is a 1-port device
        if nports < 2:
            if left: # left port
                S_cal = S_cal[:,0,0]
            else:  # right port
                S_cal = S_cal[:,1,1]
        return rf.Network(frequency=freq, s=S_cal.squeeze())
    
    def error_coef(self):
        '''
        [4] R. B. Marks, "Formulations of the Basic Vector Network Analyzer Error Model including Switch-Terms," 50th ARFTG Conference Digest, 1997, pp. 115-126.
        [5] Dunsmore, J.P.. Handbook of Microwave Component Measurements: with Advanced VNA Techniques.. Wiley, 2020.

        Left port error terms (forward direction):
        EDF: forward directivity
        ESF: forward source match
        ERF: forward reflection tracking
        ELF: forward load match
        ETF: forward transmission tracking
        EXF: forward crosstalk
        
        Right port error terms (reverse direction):
        EDR: reverse directivity
        ESR: reverse source match
        ERR: reverse reflection tracking
        ELR: reverse load match
        ETR: reverse transmission tracking
        EXR: reverse crosstalk
        
        Switch terms:
        GF: forward switch term
        GR: reverse switch term

        NOTE: the k in my notation is equivalent to Marks' notation [4] by this relationship: k = (beta/alpha)*(1/ERR).
        '''
        self.coefs = {}
        # forward 3 error terms. These equations are directly mapped from eq. (3) in [4]
        EDF =  self.X[:,2,3]
        ESF = -self.X[:,3,2]
        ERF =  self.X[:,2,2] - self.X[:,2,3]*self.X[:,3,2]
        
        # reverse 3 error terms. These equations are directly mapped from eq. (3) in [4]
        EDR = -self.X[:,1,3]
        ESR =  self.X[:,3,1]
        ERR =  self.X[:,1,1] - self.X[:,3,1]*self.X[:,1,3]
        
        # switch terms
        GF = self.switch_term[0]
        GR = self.switch_term[1]

        # remaining forward terms
        ELF = ESR + ERR*GF/(1-EDR*GF)  # eq. (36) in [4].
        ETF = 1/self.k/(1-EDR*GF)      # eq. (38) in [4], after substituting eq. (36) in eq. (38) and simplifying.
        EXF = 0*ESR  # setting it to zero, since we assumed no cross-talk in the calibration. (update if known!)

        # remaining reverse terms
        ELR = ESF + ERF*GR/(1-EDF*GR)    # eq. (37) in [4].
        ETR = self.k*ERR*ERF/(1-EDF*GR)  # eq. (39) in [4], after substituting eq. (37) in eq. (39) and simplifying.
        EXR = 0*ESR  # setting it to zero, since we assumed no cross-talk in the calibration. (update if known!)

        # forward direction
        self.coefs['EDF'] = EDF
        self.coefs['ESF'] = ESF
        self.coefs['ERF'] = ERF
        self.coefs['ELF'] = ELF
        self.coefs['ETF'] = ETF
        self.coefs['EXF'] = EXF
        self.coefs['GF']  = GF
        # reverse direction
        self.coefs['EDR'] = EDR
        self.coefs['ESR'] = ESR
        self.coefs['ERR'] = ERR
        self.coefs['ELR'] = ELR
        self.coefs['ETR'] = ETR
        self.coefs['EXR'] = EXR
        self.coefs['GR']  = GR
        # consistency check between 8-terms and 12-terms model. Based on eq. (35) in [4].
        # This should equal zero, otherwise there is inconsistency between the models (can arise from bad switch term measurements).
        self.coefs['check'] = abs( ETF*ETR - (ERR + EDR*(ELF-ESR))*(ERF + EDF*(ELR-ESF)) )
        return self.coefs 

    def reciprocal_ntwk(self):
        '''
        Return left and right error-boxes as skrf networks, assuming they are reciprocal.
        '''
        freq = rf.Frequency.from_f(self.f, unit='hz')
        freq.unit = 'ghz'

        # left error-box
        S11 = self.coefs['EDF']
        S22 = self.coefs['ESF']
        S21 = sqrt_unwrapped(self.coefs['ERF'])
        S12 = S21
        S = np.array([ [[s11,s12],[s21,s22]] for s11,s12,s21,s22 
                                in zip(S11,S12,S21,S22) ])
        left_ntwk = rf.Network(s=S, frequency=freq, name='Left error-box')
        
        # right error-box
        S11 = self.coefs['EDR']
        S22 = self.coefs['ESR']
        S21 = sqrt_unwrapped(self.coefs['ERR'])
        S12 = S21
        S = np.array([ [[s11,s12],[s21,s22]] for s11,s12,s21,s22 
                                in zip(S11,S12,S21,S22) ])
        right_ntwk = rf.Network(s=S, frequency=freq, name='Right error-box')
        right_ntwk.flip()
        return left_ntwk, right_ntwk
    
    def renorm_impedance(self, Z_new, Z0=50):
        '''
        Re-normalize reference calibration impedance.
        Z_new: new ref. impedance (can be array if frequency dependent)
        Z0: old ref. impedance (can be array if frequency dependent)
        '''
        # ensure correct array dimensions if scalar is given (frequency independent).
        N = len(self.k)
        Z_new = Z_new*np.ones(N)
        Z0    = Z0*np.ones(N)
        
        G = (Z_new-Z0)/(Z_new+Z0)
        X_new = []
        K_new = []
        for x,k,g in zip(self.X, self.k, G):
            KX_new = k*x@np.kron([[1, -g],[-g, 1]],[[1, g],[g, 1]])/(1-g**2)
            X_new.append(KX_new/KX_new[-1,-1])
            K_new.append(KX_new[-1,-1])

        self.X = np.array(X_new)
        self.k = np.array(K_new)
        self.error_coef() # update the 12 error terms

if __name__ == '__main__':
    pass
    
# EOF