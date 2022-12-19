COMMENT

Hyperpolarization-activated channel; mixed cation current; Hodgkin-Huxley style kinetics.  

Based on results from Magee, 1998. (J. Neurosci. 18(19):7613-7624. 1 October, 1998.

Authors: Tim Mickus, Bill Kath, Nelson Spruston: Northwestern University, 2000.
Modification of original Iq model by Nelson Spruston, used in Stuart & Spruston, 1998.
That file was originally modified from one provided by Michele Migliore.

Modified 8/16/02 to work with CVODE (one day, I hope) - Nelson

ENDCOMMENT

TITLE H current

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    THREADSAFE
    SUFFIX hq : cannot use simply h, NEURON will not insert mechanism otherwise
    NONSPECIFIC_CURRENT i
    RANGE i, gbar, vhalf, hh
    GLOBAL inf,tau,a0
}

UNITS {
    (mA)    = (milliamp)
    (uS)    = (microsiemens)
    (nA)    = (nanoamp)
    (mV)    = (millivolt)
    (mM)    = (milli/liter)
    (S)     = (siemens)
    (pS)    = (picosiemens)
    (um)    = (micron)
    (J)     = (joules)
}

PARAMETER {
    gbar    = 8e-4         (mho/cm2)     : maximum conductance
                                         : in rat hippocampal CA1 pyramidal neurons ranges from 8e-4 to 10e-4 S/cm2 in dend, 1e-4 to 2e-4 S/cm2 in soma (Magee, 1998, p. 7615)
    erevh   = -30          (mV)          : (Nelson Spruston used -35); note: this should be adjusted 
                                         : based on the Na+/K+ permeability and the GHK equation
    vhalf   = -80          (mV)          : small change from -82 from Magee, table 1, for 120 mM Na+ external solution, see h_AN_mod.txt for explanation 
    a0      = 0.00057      (/ms)         : this is essentially a scale factor for the time constant
    zeta    = 7            (1)
    ab      = 0.4          (1)
    qten    = 4.5          (1)           : Magee value 4.5 activation, 4.7 deactivation
    v                      (mV)
    dt                     (ms)
    temp    = 33           (degC)        : reference temperature from Magee 1998
    gas     = 8.315        (J/degC)      : universal gas constant (joules/mol/K)
    farad   = 9.648e4      (coul)        : Faraday's constant (coulombs/mol)
}

ASSIGNED {
    i                      (mA/cm2)
    inf
    tau                    (ms)
    celsius                (degC)        : actual temperature for simulation, defined in Neuron, usually about 35
}

STATE { 
    hh
}

INITIAL {
    rate(v)
    hh=inf
}

FUNCTION alpha(v(mV)) (1/ms) {
    alpha = a0*exp((0.001)*(-ab)*zeta*(v-vhalf)*farad/(gas*(273.16+celsius))) 
}

FUNCTION beta(v(mV)) (1/ms) {
    beta = a0*exp((0.001)*(1-ab)*zeta*(v-vhalf)*farad/(gas*(273.16+celsius)))
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    i = gbar*hh*(v-erevh)
}

DERIVATIVE state {
    rate(v)
    hh' = (inf-hh)/tau
}

PROCEDURE rate(v (mV)) {     : callable from hoc
: This procedure is evaluated for each segment together with the DERIVATIVE block, 
: therefore inf and tau can be GLOBAL without disturbing the spatial variation of
: the channel kinetics.

    LOCAL a, b, q10
    q10 = qten^((celsius-temp)/10(degC))
    a = q10*alpha(v)
    b = q10*beta(v)
    inf = a/(a+b)
    tau = 1/(a+b)
    if (tau<2) {tau=2}
}
