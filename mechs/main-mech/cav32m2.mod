TITLE Cav3.2 low threshold T-type calcium current
:
: Model based on experimental data of
: M. Iftinca, et al., Temperature dependence of T-type calcium channel gating, 
: Neuroscience (2006), doi: 10.1016/j.neuroscience.2006.07.010 
:
: Note: Cav3.1 kinetics are valid for 37 deg C and 1.5 mM Ca2+
: Use of Q10 to extrapolate to other temperatures should be avoided.
: Implementation uses Hodgkin-Huxley type formalism and Goldman-Hodgkin-Katz
: constant field equations to model rectification due to the large concentration
: difference between [Ca2+]in and [Ca2+]out, i.e. Ohmic current-voltage
: relationship is not applicable.
:
: This implementation uses kinetics of the form:
: ica = pcabar * m * m * h * ghk(v, cai, cao)
:
: Adrian Negrean, negreanadrian@gmail.com

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    THREADSAFE
    SUFFIX it_cav32m2
    USEION ca READ cai,cao WRITE ica
    RANGE pcabar, ica
    GLOBAL m_inf, h_inf, tau_m, tau_h
}

UNITS {
    (molar) = (1/liter)
    (mV)    = (millivolt)
    (mA)    = (milliamp)
    (mM)    = (millimolar)
    FARADAY = (faraday) (coulomb)
    R       = (k-mole) (joule/degC)
}

PARAMETER {
    v                   (mV)
    cai                 (mM)
    cao                 (mM)
    celsius             (degC)
    pcabar    = 2e-4    (cm/s)    : maximum membrane permeability
}

STATE {
    m h
}

ASSIGNED {
    ica      (mA/cm2)
    m_inf
    tau_m    (ms)
    h_inf
    tau_h    (ms)
}

BREAKPOINT {
    SOLVE castate METHOD cnexp
    ica = pcabar * m * m * h * ghk(v, 1e-6, cao) : fixed internal calcium
}

DERIVATIVE castate {
    evaluate_fct(v)

    m' = (m_inf - m) / tau_m
    h' = (h_inf - h) / tau_h
}

INITIAL {
    evaluate_fct(v)
    m = m_inf
    h = h_inf
}

PROCEDURE evaluate_fct(v(mV)) {
: This procedure is evaluated for each segment together with the DERIVATIVE block, 
: therefore m_inf, h_inf, tau_m and tau_h can be GLOBAL without disturbing the spatial variation of
: the channel kinetics.

    UNITSOFF
    m_inf = 1.0 / ( 1.0 + exp(-(v+51.5)/5.8) )
    h_inf = 1.0 / ( 1.0 + exp((v+73.7)/9.1) )
    tau_m = 0.743 + 5.938 / (exp(-(v+70.45)/12.85) + exp((v+70.45)/12.85))
    tau_h = 1.0 + 11.65*exp(-(v+60.0)/233.0)
    UNITSON
}

FUNCTION ghk(v(mV), ci(mM), co(mM)) (.001 coul/cm3) {
: high cao charge moves inward, negative potential charge moves inward

    LOCAL z, eci, eco
    z = (1e-3)*2*FARADAY*v/(R*(celsius+273.15))
    eco = co*efun(z)
    eci = ci*efun(-z)
    ghk = (.001)*2*FARADAY*(eci - eco)
}

FUNCTION efun(z) {
    if (fabs(z) < 1e-4) {
        efun = 1 - z/2
    }else{
        efun = z/(exp(z) - 1)
    }
}

UNITSON
