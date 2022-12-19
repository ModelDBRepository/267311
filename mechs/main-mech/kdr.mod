TITLE K-DR channel
: from Klee Ficker and Heinemann
: modified to account for Dax et al.
: M.Migliore 1997

NEURON {
    SUFFIX kdr
    USEION k READ ek WRITE ik 
    RANGE gkdr, gbar, ik
    RANGE ninf, ntau
    GLOBAL nscale
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
}

PARAMETER {
    ek              (mV)
    gbar = 0.003    (mho/cm2)
    vhalfn  = 13    (mV)
    a0n     = 0.02  (/ms)
    zetan   = -3    (1)
    gmn     = 0.7   (1)
    nmin    = 1     (ms)
    q10     = 1
    nscale  = 1
    temp    = 24    (degC) : temperature at which gating parameters were determined; gating at other temp is adjusted through q10 
    v               (mV)
    celsius         (degC)
}

STATE {
    n
}

ASSIGNED {
    ik (mA/cm2)
    ninf
    gkdr (mho/cm2)
    ntau (ms)
}

INITIAL {
    rates(v)
    n = ninf
    gkdr = gbar*n
    ik = gkdr*(v-ek)
}        

BREAKPOINT {
    SOLVE states METHOD cnexp
    gkdr = gbar*n
    ik = gkdr*(v-ek)
}

DERIVATIVE states {
    rates(v)
    n' = (ninf-n)/ntau
}

FUNCTION alpn(v(mV)) {
    alpn = exp(1.e-3*zetan*(v-vhalfn)*9.648e4(degC/mV)/(8.315*(273.16+celsius))) 
}

FUNCTION betn(v(mV)) {
    betn = exp(1.e-3*zetan*gmn*(v-vhalfn)*9.648e4(degC/mV)/(8.315*(273.16+celsius))) 
}

PROCEDURE rates(v (mV)) { :callable from hoc
    LOCAL a,qt
    qt = q10^((celsius-temp)/10(degC))
    a = alpn(v)
    ninf = 1/(1+a)
    ntau = betn(v)/(qt*a0n*(1+a))
    if (ntau<nmin) {ntau = nmin}
    ntau = ntau/nscale
}












