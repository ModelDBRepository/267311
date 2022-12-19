# Copyright (c) 2022 Adrian Negrean
# negreanadrian@gmail.com
#
# Software released under MIT license, see license.txt for conditions
import os, platform
import numpy as np, re, copy
from collections.abc import Iterable

import neuron as nrn
from neuron import h as nrnh
min_nrn_ver = (7, 8, 2)
if not tuple(map(int, nrn.version.split('.'))) >= min_nrn_ver:
    raise EnvironmentError(
        "NEURON version too low, required at least {}".format('.'.join(str(n) for n in min_nrn_ver)))

from nrnengine import util

# global flag to signal whether NMODL compiled, external membrane mechanisms, were loaded to the environment
mech_loaded = False

class PointProc():
    """
    Point process wrapper class
    """
    def __init__(self):
        self.pproc = None # set this to the NEURON point process object

# checked 8/21/21
class IClamp(PointProc):
    """
    NEURON IClamp wrapper.
    """
    def __init__(self, seg, waveform):
        """
        Initializes current clamp.
        
        Parameters
        ----------
        seg : str, nrn.nrn.Segment 
            NEURON segment object or segment name into which the current will be injected.
        waveform : dict
            Current waveform specification. Type of waveform input is determined by the mandatory 'type' dict key, which can be:
                'step':
                    A simple current step defined by parameters:
                        'delay': float
                            Delay in [ms] before onset of current injection.
                        'dur': float
                            Duration of current injection in [ms].
                        'amp': float
                            Amplitude in [nA] of current step.
                'waveform':
                    Arbitrary waveform current injection. Has keys:
                        'data': iterable of float
                            Current waveform with same sampling rate as the simulation time step.
                        'dt': float
                            Waveform sampling time step in [ms].
        """
        # convert to NEURON segment object if needed
        seg = get_secseg(h = nrnh, secseg = seg)

        self.pproc = nrnh.IClamp(seg)

        if waveform['type'] == 'step':
            self.pproc.delay = waveform['delay']
            self.pproc.dur = waveform['dur']
            self.pproc.amp = waveform['amp']
        elif waveform['type'] == 'waveform':
            pvec = nrnh.Vector().from_python(waveform['data'])
            self.pproc.dur = 1e9 # necessary so that current injection in determined by the input data
            pvec.play(self.pproc, self.pproc._ref_amp, waveform['dt'], True)
        else:
            raise ValueError("Waveform type not implemented.")

# checked 8/21/21
def get_attr_from_str(obj, attr_str, path_marker = '.'):
    """
    Obtains the attribute of an object given a string attr_str of the form '<attr1><path_marker><attr2><path_marker> ...'
    If attr_str is None or '' then it returns the original object.
    
    Parameters
    ----------
    obj : object
        Python object.
    attr_str : str
        Attribute of the form '<attr1><path_marker><attr2><path_marker> ...'
    path_marker : str
        Character to use as path marker to obtain object attribute.
    
    Returns
    -------
    o : object
        Object attribute.
    """
    if attr_str:
        attrlist = attr_str.split(path_marker)
        o = obj
        for attr in attrlist:
            o = getattr(o, attr)
        return o
    else:
        return obj

def record_hoc_par(h, recinfo):
    """
    Record segment or point process parameters.

    Parameters
    ----------
    h : hoc.HocObject
        Neuron interpreter.

    recinfo : dict of lists
        Dict keys are nrn.Segment or nrn.Segment mechanism level parameters of the form "<seg par>" or "<seg mech>.<mech par>"
        or point process parameters.
        If parameter does not exist, an empty np.array will be added. Dict values are list of str with section or segment names
        nrn.Section or nrn.Segment objects, or point process objects.

    Returns
    -------
    out : dict of dict
        Recording vectors. First level keys are segment, segment mechanism or point process parameters. Second level keys are
        names of sections, segments or point processes in which the first level key parameters are measured. Values are list of NEURON vectors.
    """
    out = {}
    # pname - segment or segment mechanism parameter name
    # objs - list of objects, e.g. section or segments as str, nrn.Section or nrn.Segment or point processes
    for pname, objs in recinfo.items():
        attrs = pname.split('.')
        attrs[-1] = '_ref_' + attrs[-1]
        out[pname] = {}
        # ensure all string named sections and segments are converted to objects
        objs = [get_secseg(h = nrnh, secseg = obj) for obj in objs]
        for obj in objs:
            if isinstance(obj, nrn.nrn.Section):
                _objs = list(obj)
            # for segments and point processes
            else:
                _objs = [obj]
            recv = []
            for _obj in _objs:
                o = _obj
                for attr in attrs:
                    o = getattr(o, attr)
                # record segment attribute
                vec = h.Vector()
                vec.record(o)
                recv.append(vec)
            out[pname][get_hoc_name(h = h, obj = obj)] = recv

    return out

# checked 7/30/21
def load_hoc_file(h, fpath):
    """
    Loads a single HOC file.

    Parameters
    ----------
    h : hoc.HocObject
        Neuron interpreter.
    
    fpath : str
        Absolute path to .hoc file

    Returns
    -------
    None
    """
    if not os.path.isfile(fpath):
        raise Exception("File '{}' does not exist.".format(fpath))
    # delete all existing sections (including mechanisms and point processes) from hoc.
    # note: for some NEURON versions earlier than 8.0 this seems to fail when creating sections with h.Section()
    assert h('forall delete_section()')
    # execute hoc commands from file
    print("opening HOC file '{}'...".format(fpath))
    if not h('xopen("' + fpath + '")'):
        raise Exception("Execution of " + fpath + " failed.")
    print('done.')

# checked 7/31/21
def get_nsecseg(h):
    """
    Returns the number of sections and segments present in NEURON HOC.

    Parameters
    ----------
    h : hoc.HocObject
        Neuron interpreter.

    Returns
    -------
    tuple
        Number of section and segments as (nsec, nseg).
    """
    nsec = 0
    nseg = 0    
    for sec in h.allsec():
        nsec += 1
        nseg += sec.nseg
    
    return (nsec, nseg)

# checked 7/31/21
def resegment(h, freq = 100, d_lambda = 0.1, maxLseg = 2, use_d_lambda = True):
    """
    Resegment all hoc sections to increase spatial accuracy. If sections have a non-zero specific membrane capacitance,
    then the 'd-lambda' rule may be used otherwise, resegmentation is done to ensure that all segments are shorter than max_Lseg.
    In both cases each section is divided into an odd number of segments. For more info on this, see:
        Hines, M.L. and Carnevale, N.T.
        NEURON: a tool for neuroscientists.
        The Neuroscientist 7:123-135, 2001.
    Note: Resegmentation must be done each time Ra or cm are modified if cm is not 0.
    
    Parameters
    ----------
    h : hoc.HocObject
        Neuron interpreter.
    freq : float
        Frequency at which AC length constant will be computed in [Hz].
    d_lambda : float
        Length constant fraction.
    maxLseg : float
        Maximum segment length in [um] which is enforced if the section has zero capacitance.
    use_d_lambda : bool
        If True and sections have a non-zero capacitance, then the d_lambda rule is used for resegmentation,
        otherwise the maxLseg rule is used for both sections having a non-zero or zero capacitance.

    Returns
    -------
    None
    """
    for sec in h.allsec():
        # ensure all segments within a section have either zero or non-zero capacitance
        all_non_zero_cm = True
        all_zero_cm = True
        for seg in sec:
            if seg.cm == 0:
                all_non_zero_cm = False
            else:
                all_zero_cm = False
        assert all_non_zero_cm or all_zero_cm
        
        if sec.cm == 0 or not use_d_lambda:
            nseg = int(sec.L/maxLseg)
            if nseg < 1:
                nseg = 1
            if not nseg % 2:
                nseg += 1   # make sure the number of segments is always odd so that sec(0.5) has a segment    
            sec.nseg = nseg
        else:
            # calculate the spatial constant at the given frequency
            if h.n3d(sec=sec) < 2:
                lambda_f = 1e5*np.sqrt(sec.diam/(4*np.pi*freq*sec.Ra*sec.cm))
            else:
                # use all 3-d points to get a better approximate lambda
                x1 = h.arc3d(0, sec=sec)
                d1 = h.diam3d(0, sec=sec)
                lam = 0
                for i in range(1, int(h.n3d(sec=sec))):
                    x2 = h.arc3d(i, sec=sec)
                    d2 = h.diam3d(i, sec=sec)
                    lam += (x2 - x1)/np.sqrt(d1 + d2)
                    x1 = x2
                    d1 = d2
                #  length of the section in units of lambda
                lam *= np.sqrt(2) * 1e-5*np.sqrt(4*np.pi*freq*sec.Ra*sec.cm)
                lambda_f = sec.L/lam
                
            sec.nseg  = int((sec.L/(d_lambda*lambda_f)+0.9)/2.0)*2 + 1    # also make sure the number of segments is always odd so that sec(0.5) has a segment

# checked 8/05/21
def get_secseg(h, secseg):
    """
    Converts string section or segment names to nrn.nrn.Section or nrn.nrn.Segment objects.
    Any other object is left unchaged.

    Parameters
    ----------
    h : hoc.HocObject
        Neuron interpreter.
    secseg : nrn.nrn.Section, nrn.nrn.Segment, str
        Section or segment object or name such as 'soma[0](0.5)'.

    Returns
    -------
    nrn.nrn.Section or nrn.nrn.Segment object, or original object.
    """
    if isinstance(secseg, str):
        # get section path, e.g. Cell[0].dend[72]
        sec_path = re.sub('(\([0-9]*\.?[0-9]+\))', '', secseg)
        if not sec_path:
            raise Exception("No section was given.")
        # get segment location, e.g. (0.5)
        seg_locs = re.findall('\(([0-9]*\.?[0-9]+)\)', secseg)
        if seg_locs:
            seg_loc = float(seg_locs[0])
        else:
            seg_loc = None

        # split sec_path using '.' to indicate next object
        sec = sec_path
        remaining_sec = sec_path
        obj = h
        while True:
            # get section which may be either indexed or not, i.e. 'dend' or 'dend[72]'
            idx = remaining_sec.find('.')
            if idx > -1:
                sec = remaining_sec[:idx]
            else:
                sec = remaining_sec
            # get section name and index
            try:
                sec_groups =  re.match('(\w+)(\[([0-9]+)?\])?', sec).groups()
            except Exception as e:
                print("Cannot format section or segment '{}'.".format(sec))
                raise e
            # access object
            obj = getattr(obj, sec_groups[0])
            if sec_groups[2]:
                obj = obj[int(sec_groups[2])]

            # stop if no more sections
            if idx == -1:
                break
            else:
                remaining_sec = remaining_sec[idx+1:]

        if seg_loc is not None:
            # get segment
            out = obj(seg_loc)
        else:
            # get section
            out = obj
    else:
        out = secseg
    
    return out

# checked 8/05/21
def is_mechpar_global(h, par, path_mrkr = '.'):
    """
    Checks if mechanism parameter is defined as global.

    Parameters
    ----------
    h : hoc.HocObject
        NEURON hoc interpreter object.

    par : str
        Mechanism parameter of the form '<mech name><path_mrkr><parameter>' e.g. 'pas.g' for passive leak conductance.
        If segment parameter is provided e.g. 'cm' it returns False.

    path_mrkr: str
        Symbol to separate mechanism from parameter.
    """
    par_split = par.split(path_mrkr)
    if len(par_split) == 1:
        return False
    elif len(par_split) == 2:
        seg_attr = par_split[0]
        mech_attr = par_split[1]
        # if accessing throws a type error, then catch it and use it to define segment attribute as RANGE.
        try: 
            is_global = hasattr(h, mech_attr+'_'+seg_attr)
        except TypeError:
            is_global = False
        return is_global
    else:
        raise Exception("Segment attribute/mechanism '{}' could not be formatted.".format(par))

# checked 8/05/21
def meas_seg_distance(h, ref_seg, to_seg, use_path = True):
    """
    Calculates the distance between a reference segment ref_seg and target to_seg.

    Parameters
    ----------
    h : hoc.HocObject
        NEURON hoc interpreter object.
    ref_seg : nrn.nrn.Segment or str
        Reference segment object or name, e.g. 'dend[10](0.7)' from which distance is measured.
    to_seg : nrn.nrn.Segment object or str
        Target segment object or name, e.g. 'dend[10](0.7)' to which distance is measured.
    use_path : bool
        If True, path distance is used, otherwise direct distance between the two segments.
    
    Returns
    -------
    float
        Distance between segments.
    """
    # if segment is given as string e.g. 'soma[0](0.5)' then covert to nrn.nrn.Segment object
    ref_seg = get_secseg(h, ref_seg)
    to_seg = get_secseg(h, to_seg)

    if use_path:
        ref_section = ref_seg.sec
        # make the reference section the currently accessed section
        ref_section.push()
        # set the distance measurement reference to ref_section(0)
        h.distance()
        # measure path distance from ref_section(0) to to_seg
        distance = np.abs(h.distance(to_seg.x, sec = to_seg.sec) - h.distance(ref_seg.x, sec = ref_section))
        # prevent stack overflow
        h.pop_section()
        return distance
    else:
        return _get_line_length(get_segment_coords(h, ref_seg), get_segment_coords(h, to_seg))

# checked 8/05/21
def get_segment_coords(h, seg):
    """
    Returns the cartesian centroid coordinates of all segments or a single segment with given normalized distance within the section.
    
    Parameters
    ----------
    h : hoc.HocObject
        NEURON hoc interpreter object.
    seg : nrn.nrn.Segment or str
        Segment.
    
    Returns
    -------
    coords : numpy 1D array
        XYZ segment centroid coordinates if x is specified or a 2D array with all the segment centroid coordinates.
    """
    ref_seg = get_secseg(h, seg)
    # NEURON's interpolation function is used because using numpy interp consistently fails.
    
    # create vectors to store node anatomical coordinates and normalized lengths
    n3d = int(h.n3d(sec = seg.sec))
    xNodesVect = h.Vector(n3d)
    yNodesVect = h.Vector(n3d)
    zNodesVect = h.Vector(n3d)
    lNodesVect = h.Vector(n3d)
    
    # create vectors to store segment coordinates
    lSegMiddleVect = h.Vector(seg.sec.nseg)
    xSegVect = h.Vector(seg.sec.nseg)
    ySegVect = h.Vector(seg.sec.nseg)
    zSegVect = h.Vector(seg.sec.nseg)
    
    L = h.arc3d(n3d-1, sec = seg.sec)
    for i in range(0, n3d):
        xNodesVect.x[i] = h.x3d(i, sec = seg.sec)
        yNodesVect.x[i] = h.y3d(i, sec = seg.sec)
        zNodesVect.x[i] = h.z3d(i, sec = seg.sec)
        lNodesVect.x[i] = h.arc3d(i, sec = seg.sec)/L
    
    # generate evenly spaced normalized segment center coordinates
    lSegMiddleVect.indgen(1.0/seg.sec.nseg).add(1.0/(2*seg.sec.nseg))
    
    # interpolate
    xSeg = xSegVect.interpolate(lSegMiddleVect, lNodesVect, xNodesVect).to_python()
    ySeg = ySegVect.interpolate(lSegMiddleVect, lNodesVect, yNodesVect).to_python()
    zSeg = zSegVect.interpolate(lSegMiddleVect, lNodesVect, zNodesVect).to_python()
    
    # package result
    segIdx = int(np.ceil(seg.x*seg.sec.nseg))
    if segIdx == 0:
        segIdx = 1
    coords = np.empty([1, 3])
    coords[0,0] = xSeg[segIdx-1]
    coords[0,1] = ySeg[segIdx-1]
    coords[0,2] = zSeg[segIdx-1]
    return coords

# checked 8/05/21
def _get_line_length(p1, p2):
    """
    Calculates the length of a segment between points p1 and p2 defined in cartesian coordinates.
    
    IN
    p1, p2 = n dimensional row vectors, tuples or lists with cartesian coordinates.

    OUT
    length = scalar distance between points p1 and p2.  
    """
    # check vector shapes
    assert _check_coordinate_vect_shapes([p1, p2])
    
    # ensure p1 and p2 are numpy ndarrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    length = 0
    for i in range(0, p1.size):
        length += (p1[0, i] - p2[0, i])**2 
    
    length = np.sqrt(length)
    return length
    
# checked 8/05/21
def _check_coordinate_vect_shapes(vectors):
    """
    Checks if numpy vectors in a list are all row vectors with the same number of columns.
    
    IN
    vectors = list of numpy row vectors, tuples or lists.
    
    OUT
    True if all vectors are row vectors of the same size, False otherwise
    """
    # get vector shapes
    vec_shapes = []
    for vec in vectors:
        vec_shapes.append(vec.shape)
    
    # perform checks
    for shape in vec_shapes:
        # check if vectors are matrices
        if not len(shape) == 2:
            return False
        # check if vectors are row vectors
        if not shape[0] == 1:
            return False        
    
    # check if vectors have the same number of columns
    nCols = vec_shapes[0][1]
    for shape in vec_shapes:
        if not nCols == shape[1]:
            return False
            
    return True

# checked 8/05/21
def set_seg(h, par, sec_list = {}, subtree_sec_list = {}, path_mrkr = '.'):
    """
    Assign segment parameter values.

    Parameters
    ----------
    h : hoc.HocObject
        NEURON hoc interpreter object.

    par : dict
        Dict keys can be:
            - section properties e.g. length 'L', axial resistivity 'Ra', number of segments 'nseg'.
            - segment range attributes of the form '<segment attribute>' e.g. 'diam', capacitance 'cm'.
            - mechanism range or global parameters '<mech name><path marker><parameter name>' e.g. passive leak conductance 'pas.g'.
        Dict values can be:
            - instance of int, float or iterable except dict: parameter is modified in all hoc sections.
            - instance of dict: parameter is modified in select sections named as dict keys.

            If iterable and first element is instance of str, range parameter is adjusted in segments in a distance dependent way, otherwise,
            elements are assumed to be of numeric type and they are applied to each segment of a section. If the number of segments does not match the
            number of elements, values are interpolated.
        Distance dependent models:    
            a) Linear ramp with start value at reference segment and path-distance dependent change with given slope: 
               ('lin1', refseg, val_at_refseg, slope, cutoff[optional])
               refseg           Reference segment used for path-distance measurement. Can be string such as 'soma[0](0.5)' or nrn.nrn.Segment object.
               val_at_refseg    Attribute value at the reference segment.
               slope            Rate of attribute value change in <attribute unit>/[um].
               cutoff           Optional value above/below which the linear function will level off.
               
            b) Linear ramp with start value at reference segment and path-distance dependent change reaching given value at specified distance:
               ('lin2', refseg, val_at_refseg, val_at_dist, dist)
               refseg           Reference segment used for path-distance measurement. Can be string such as 'soma[0](0.5)' or nrn.nrn.Segment object.
               val_at_refseg    Attribute value at the reference segment.
               val_at_dist      Attribute value at distance given by dist.
               dist             Path distance at which attribute reaches value given by val_at_dist.
               
            c) Localized linear ramp:
               ('loclin', refseg, val_at_refseg, val_at_end_dist, start_dist, end_dist)
               refseg           Reference segment used for path-distance measurement. Can be string such as 'soma[0](0.5)' or nrn.nrn.Segment object.
               val_at_refseg    Attribute value at the position of refseg which is maintained up to a distance of start_dist.
               val_at_end_dist  Attribute value at a distance of end_dist from refseg which is maintained at distances beyond end_dist.
               start_dist       Path-distance from refseg over which the attribute value is constant and has the value equal to val_at_refseg.
                                Beyond this distance and up to end_dist the value of the attribute is changing linearly reaching val_at_end_dist.
               end_dist         Distance up to which the attribute value is changing linearly and reaches a value of val_at_end_dist.
               
            d) Sigmoidal:
               ('sigm', refseg, val_at_refseg, val_at_end, midpoint, steepness)
               refseg           Reference segment used for path-distance measurement. Can be string such as 'soma[0](0.5)' or nrn.nrn.Segment object.
               val_at_refseg    Attribute value at the position of refseg.
               val_at_end       Attribute value as the distance from refseg grows to infinity.
               midpoint         Sigmoid midpoint distance in [um].
               steepness        Sigmoid steepness parameter in [um].

    sec_list : dict
        Section lists e.g. {'soma': [sec1], 'trunk': [sec1, sec2, ...]} with sec1, sec2,...,
        nrn.nrn.Section or str name of section

    subtree_sec_list : dict
        List of list of sections representing subtrees, e.g. {'tuft': [[sec1, sec2], [sec3, sec4]], ...}

    path_mrkr : str
        Character to mark mechanism level parameter path, e.g. '.' for 'pas.g'.
    """
    for p_name, p_val in par.items():    
        if isinstance(p_val, (int, float, np.integer, np.float, Iterable)) and not isinstance(p_val, dict):
            # set a segment mechanism or segment parameter e.g. 'pas.g' or 'cm' to be the same in all HOC sections
            for sec in h.allsec():
                _setpar(h = nrnh, sec = sec, p_name = p_name, p_val = p_val, path_mrkr = path_mrkr)
        elif isinstance(p_val, dict):
            # set a segment mechanism or segment parameter e.g. 'pas.g' or 'cm' to be the same in defined HOC sections
            for sec_name, sec_p_val in p_val.items():
                # use section list
                if sec_name[0] == '~':
                    for sec in sec_list[sec_name[1:]]:
                        _setpar(h = nrnh, sec = sec, p_name = p_name, p_val = sec_p_val, path_mrkr = path_mrkr)
                # use subtree section lists, i.e. these are lists of lists, with the inner most list being a subtree, with first section being a root section
                elif sec_name[0] == '@':
                    # iterate over each subtree list
                    for sec_list in subtree_sec_list[sec_name[1:]]:
                        # iterate over each section in the subtree list
                        for sec in sec_list:
                            # if no reference segment was specified, set this to be the parent of the first section in the list
                            # note: - this is to be able to make use of the fact that a subtree has as its first section a parent section
                            #       - this will take the value of the parameter from the parent section so distance dependent changes can be added
                            sec_p_val_cpy = copy.deepcopy(sec_p_val)
                            if "refseg" not in sec_p_val:    
                                sec_p_val_cpy["refseg"] = get_secseg(h = nrnh, secseg = sec_list[0]).parentseg()
                            _setpar(h = nrnh, sec = sec, p_name = p_name, p_val = sec_p_val_cpy, path_mrkr = path_mrkr)
                else:
                    # use section from hoc
                    try:
                        sec = get_secseg(h, sec_name)
                    except AttributeError:
                        raise Exception("Section '{}' is not defined in HOC.".format(sec_name))
                    _setpar(h = nrnh, sec = sec, p_name = p_name, p_val = sec_p_val, path_mrkr = path_mrkr)

# checked 8/05/21
def _setpar(h, sec, p_name, p_val, path_mrkr = '.'):
    """
    Sets a segment mechanism RANGE or mechanism GLOBAL parameter.

    Parameters
    ----------
    h : hoc.HocObject
        NEURON hoc interpreter object.

    sec : Neuron hoc section
    
    p_name : str
        Segment or segment mechanism parameter e.g. capacitance 'cm' or passive conductance 'pas.g'.
    
    p_val : float, int, dict
        Value to assign to segment parameter.
    
    path_mrkr : str
        Path marker e.g. '.'.
    """
    sec = get_secseg(h = h, secseg = sec)
    
    # split parameter
    p_split_name = p_name.split(path_mrkr)
    if len(p_split_name) == 1:
        seg_attr = p_split_name[0]
        mech_attr = ''
    elif len(p_split_name) == 2:
        seg_attr = p_split_name[0]
        mech_attr = p_split_name[1]
    else:
        raise Exception("Segment attribute/mechanism '{}' could not be formatted.".format(p_name))
    
    if p_name in ['L', 'nseg', 'Ra']:
        # set section level parameter directly
        # type checks
        if isinstance(p_val, (int, float, np.integer, np.float)):
            # nseg must be int
            if p_name == 'nseg' and isinstance(p_val, (float, np.float)):
                raise TypeError("Parameter 'nseg' for section '{}' must be integer.".format(sec.name()))
            setattr(sec, p_name, p_val)
        else:
            raise TypeError("Parameter '{}' type for section '{}' must be int or float.".format(p_name, sec.name()))
    else:
        # insert membrane mechanism if not inserted already
        if not sec.has_membrane(seg_attr) and seg_attr not in ['cm', 'diam']:
            try:
                sec.insert(seg_attr)
            except:
                raise Exception("'{}' membrane mechanism could not be inserted in section '{}'. Ensure mechanism is first loaded into hoc.".format(seg_attr, sec.name()))
        # check if GLOBAL by trying to set in hoc, otherwise it is RANGE and mechanism may need to be added to the section
        if is_mechpar_global(h = h, par = p_name, path_mrkr = path_mrkr):
            setattr(h, mech_attr+'_'+seg_attr, p_val)
        else:
            if hasattr(getattr(sec(0.5), seg_attr), mech_attr) or mech_attr == '': # either segment mechanism RANGE parameter e.g. 'pas.g' or segment RANGE parameter e.g. 'cm'
                # segment RANGE parameter
                if isinstance(p_val, (int, float, np.integer, np.float)):
                    # set segment level parameter to be the same in all segments within the section
                    # assume segment RANGE parameter, if not present then maybe it's GLOBAL and kept in the 'h' hoc object
                    for seg in sec:
                        util.set_attr_from_str(obj = seg, attr_str = p_name, val = p_val, path_marker = path_mrkr)
                elif isinstance(p_val, dict):
                    # ensure that reference segment is an inner section segment, i.e. not at the very ends that are used to connect sections
                    # note: connecting segments at section ends do not contain inserted mechanisms
                    refseg = get_secseg(h = h, secseg = p_val["refseg"])
                    if refseg.x == 1:
                        refseg = list(refseg.sec)[-1]
                    elif refseg.x == 0:
                        refseg = list(refseg.sec)[0]

                    # get parameter value at reference segment
                    if "val_at_refseg" in p_val:
                        refseg_val = p_val["val_at_refseg"]
                    else:
                        refseg_val = get_segdata(h = h, obj = refseg, attr = p_name)[0][0]
                    # constant value
                    if p_val["type"] == "const":
                        for seg in sec:
                            util.set_attr_from_str(obj = seg, attr_str = p_name, val = refseg_val, path_marker = path_mrkr)
                    # distance dependent models
                    elif p_val["type"] == "lin1":
                        for seg in sec:
                            val = refseg_val + p_val["slope"] * meas_seg_distance(h = h, ref_seg = refseg, to_seg = seg, use_path = True)
                            if "cutoff" in p_val:
                                if p_val["slope"] >= 0:
                                    if val > p_val["cutoff"]:
                                        attr_val = p_val["cutoff"]
                                    else:
                                        attr_val = val
                                else:
                                    if val < p_val["cutoff"]:
                                        attr_val = p_val["cutoff"]
                                    else:
                                        attr_val = val
                            else:
                                attr_val = val
                            
                            util.set_attr_from_str(obj = seg, attr_str = p_name, val = attr_val, path_marker = path_mrkr)
                    
                    elif p_val["type"] == "lin2":
                        for seg in sec:
                            attr_val = refseg_val + (p_val["val_at_dist"] - refseg_val)/ \
                            p_val["dist"] * meas_seg_distance(h = h, ref_seg = refseg, to_seg = seg, use_path = True)
                            if "cutoff" in p_val:
                                if p_val["val_at_dist"] > refseg_val and attr_val > p_val["cutoff"] or \
                                    p_val["val_at_dist"] < refseg_val and attr_val < p_val["cutoff"]:
                                    attr_val = p_val["cutoff"]

                            util.set_attr_from_str(obj = seg, attr_str = p_name, val = attr_val, path_marker = path_mrkr)


                    elif p_val["type"] == "loclin":
                        for seg in sec:
                            dist = meas_seg_distance(h = h, ref_seg = refseg, to_seg = seg, use_path = True)
                            if dist < p_val["start_dist"]:
                                attr_val = refseg_val
                            elif dist >= p_val["end_dist"]:
                                attr_val = p_val["val_at_end_dist"]
                            else:
                                attr_val = refseg_val + (p_val["val_at_end_dist"] - refseg_val)/ \
                                (p_val["end_dist"]-p_val["start_dist"]) * (dist-p_val["start_dist"])
                            util.set_attr_from_str(obj = seg, attr_str = p_name, val = attr_val, path_marker = path_mrkr)

                    elif p_val["type"] == "sigm":
                        attr_val = refseg_val + (p_val["val_at_end"] - refseg_val)/ \
                        (1.0 + np.exp((p_val["midpoint"] - meas_seg_distance(h = h, ref_seg = refseg, to_seg = seg, use_path = True))/p_val["steepness"]))
                        util.set_attr_from_str(obj = seg, attr_str = p_name, val = attr_val, path_marker = path_mrkr)

                    else:
                        raise ValueError("Distance dependent model '{}' is not implemented.".format(p_val["type"]))

                elif isinstance(p_val, Iterable):
                    # if number of segments is not the same as the number of values to be assigned, then interpolate values
                    p_val = np.array(p_val)
                    if len(p_val.shape) != 1:
                        raise ValueError("Parameter '{}' in section '{}' must be a 1D array.".format(p_name, sec.name()))
                    if len(p_val) != sec.nseg:
                        p_val = np.interp(np.linspace(0, 1.0, sec.nseg), np.linspace(0, 1.0, len(p_val)), p_val)                    
                    # assign segment attribute values
                    for idx, seg in enumerate(sec):
                            util.set_attr_from_str(obj = seg, attr_str = p_name, val = p_val[idx], path_marker = path_mrkr)
                else:
                    raise TypeError("Segment mechanism attribute '{}' type must be int, float or iterable.".format(p_name))
            
            else:
                raise ValueError("Parameter '{}' in section '{}' is neither a segment or mechanism RANGE/GLOBAL parameter.".format(p_name, sec.name()))

# checked 8/12/21
def load_mech(h, fpath):
    """
    Load external NEURON mechanisms and set global mech_loaded flag to True to suppress hoc scripts from loading library again (which will crash the kernel).

    Parameters
    ----------
    h : hoc.HocObject
        NEURON hoc interpreter object.

    fpath : str
        Absolute path to .hoc file

    Returns
    -------
    None

    Mechanisms are loaded from a platform independent folder path in fpath. Actual library file will be selected
    based on the platform architecture. If mechanism are loaded from HOC first, set HOC variable 'mech_loaded'
    from 0 to 1 to prevent loading mechanisms again here.
    """
    global mech_loaded

    # load custom defined mechanisms
    if mech_loaded or hasattr(h, 'mech_loaded') and h.mech_loaded:
        return

    # platform specific mechanism paths
    machine = platform.machine()
    if machine == 'AMD64':
        mechpath_machine = os.path.join(fpath, 'AMD64/nrnmech.dll')
    elif machine == 'x86_64':
        mechpath_machine = os.path.join(fpath, 'x86_64/.libs/libnrnmech.so')
    else:
        raise Exception("There are no membrane mechanisms compiled for platform '{}'".format(machine))
            
    # warning: loading of dll will hang if same mechanism name is defined in more than one file!
    if not int(h.nrn_load_dll(mechpath_machine)):
        raise Exception("Could not load custom defined mechanisms from '{}'.".format(mechpath_machine))
    else:
        h('mech_loaded = 1')
        mech_loaded = True

# checked 8/21/21
def get_hoc_name(h, obj):
    """
    Get the name of a NEURON section, segment, or point process object.

    Parameters
    ----------
    h : hoc.HocObject
        NEURON hoc interpreter object.
    obj : str, nrn.nrn.Section, nrn.nrn.Segment, hoc.HocObject
        NEURON section or segment name or objects.
    """
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, nrn.nrn.Segment):
        return obj.sec.name() + '({})'.format(obj.x)
    elif isinstance(obj, nrn.nrn.Section):        
        return obj.hname()
    elif isinstance(obj, nrn.hoc.HocObject):
        return obj.hname()
    else:
        raise ValueError()

def run(self):
        """
        Runs the currently defined simulation. Before running this function, ensure that all changes to models have been applied by either applying them
        manually using self._build_model or by setting the self._build_model flag to True, which will force their application.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
            Simulation measurement parameters defined in self.ps['g_recvar'] will be stored in self.rec 
        """
        # apply models if needed
        if self._build_model:
            self.build_model()

        # run model and record parameters
        self.rec = record(h = self.h, obj = self.sec['test'], params = self.ps['g_recvar'])
        # initialize simulation control object
        self.sim_ctrl = SimulationControl(self.h, self.ps['g_vinit'], self.ps['g_temp'], self.ps['g_sim_dur'], self.ps['g_sim_dt'])
        # use cache memory efficiently
        #cvode = h.CVode()
        #cvode.cache_efficient(1)
        # run simulation
        run_model(self.sim_ctrl)
        # convert NEURON vectors to numpy arrays        
        self.rec = apply_fn(np.array, self.rec)
        
        # correct variation in end point time that seems to be a rounding issue in NEURON.
        # the simulation time vector starts at t = 0 and must end at t = self.ps['g_sim_dur'] in a reproducible way regardless of the value of self.ps['g_sim_dt'].
        # instead what happens is that sometimes the simulation ends at t = self.ps['g_sim_dur'] or at t = self.ps['g_sim_dur'] + 
        # self.ps['g_sim_dt']

        def drop_last_sample(sig):
            """
            Drops last sample.
            """
            sig_new = sig[:-1]
            return sig_new

        if np.abs(self.rec['t'][-1] - self.ps['g_sim_dur'] - self.ps['g_sim_dt']) < 1e-6:
            self.rec = apply_fn(drop_last_sample, self.rec)
        elif np.abs(self.rec['t'][-1] - self.ps['g_sim_dur']) > 1e-6:
            raise Exception('Simulation end timestamp ({}) does not match simulation end time ({})'.format(self.rec['t'][-1], self.ps['g_sim_dur']))

def run_model(sim_ctrl):
    """
    Runs a simulation.
    
    IN
    sim_ctrl    SimulationControl object.
    """
    # Temperature in [Celsius]
    sim_ctrl.h.celsius = sim_ctrl.temp
    # Simulation time step in [ms]
    sim_ctrl.h.dt = sim_ctrl.dt
    # Initialize the simulation
    sim_ctrl.h.finitialize(sim_ctrl.v_init)
    sim_ctrl.h.fcurrent()
    # Run the simulation
    nrn_run(sim_ctrl.tstop)

def meas_Zin_freq_domain(h, freq, v_init = None, obj = None):
    """
    Measures input impedance magnitude and phase in the frequency domain using the built-in nrn.Impedance class. Note,
    if input impedances must be calculated for multiple segments, it is more efficient to pass these at once instead of
    calling this function repeatedly. Impedance calculation takes into account the effect of differential gating states.
    From the Impedance class documentation:
    
    "an extended impedance calculation is performed which takes into account the effect of differential gating states. ie. 
    the linearized cy' = f(y) system is used where y is all the membrane potentials plus all the states in KINETIC and DERIVATIVE 
    blocks of membrane mechanisms. Currently, the system must be computable with the Cvode method, i.e. extracellular and 
    LinearMechanism are not allowed"
    
    Parameters
    ----------
    h : nrn.hoc.HocObject
        NEURON hoc object.
    obj : nrn.nrn.Segment, nrn.nrn.Section, list (or tuple) of nrn.nrn.Segment, list (or tuple) of nrn.nrn.Section, dict of previous objects.
        If None, the impedance is measured for all section and segments in hoc.
    freq : float
        Frequency at which to measure the impedance in [Hz].
    
    Returns
    -------
    Output format depends on the input format:
    1) nrn.nrn.Segment -> numpy ndarray of shape (1,2), |Zin| = out[0,0], <Zin = out[0,1]
    2) nrn.nrn.Section -> numpy ndarray of shape (n,2), where n is the number of segments within the section and |Zin| = out[:,0], <Zin = out[:,1]
    3) list of nrn.nrn.Segment -> numpy ndarray of shape (n,2), where n is the number of segments within the list and |Zin| = out[:,0], <Zin = out[:,1]
    4) list of nrn.nrn.Section -> numpy ndarray of shape (n,2), where n is the total number of segments within the list and |Zin| = out[:,0], <Zin = out[:,1]
    5) dict of objects 1-4 -> dict of numpy ndarray with same labels as the original dict and entries processed similar to 1-4. 
    6) None -> dict with keys matching hoc section names and values according to 2)
    
    Units:
    |Zin| in [MOhm]
    <Zin in [deg]
    """
    # if required, initialize membrane potential in all compartments to compute 
    # the input impedance accurately
    if v_init is not None:
        h.finitialize(v_init)
        
    imp = h.Impedance()
    
    if type(obj) is nrn.nrn.Segment:
        imp.loc(0, sec = obj.sec)
        imp.compute(freq, 1)
        impedance = imp.input(obj.x, sec = obj.sec)
        phase = 180.0/np.pi*imp.input_phase(obj.x, sec = obj.sec)
        return np.array([[impedance, phase]])    
        
    elif type(obj) is nrn.nrn.Section:
        imp.loc(0, sec = obj)
        imp.compute(freq, 1)
        Zin = np.empty((obj.nseg, 2))
        for idx, seg in enumerate(obj):
            Zin[idx, 0] = imp.input(seg.x, sec = obj)
            Zin[idx, 1] = 180.0/np.pi*imp.input_phase(seg.x, sec = obj)
        return Zin

    elif type(obj) in [list, tuple]:
        segments = []
        for item in obj:
            if type(item) is nrn.nrn.Segment:
                segments.append(item)
            elif type(item) is nrn.nrn.Section:
                segments.extend(list(item))
        imp.loc(0, sec = segments[0].sec)
        imp.compute(freq, 1)
        Zin = np.empty((len(segments), 2))
        for idx, seg in enumerate(segments):
            Zin[idx, 0] = imp.input(seg.x, sec = seg.sec)
            Zin[idx, 1] = 180.0/np.pi*imp.input_phase(seg.x, sec = seg.sec)
        return Zin
        
    elif type(obj) is dict:
        Zin = {}
        for key in obj: 
            if type(obj[key]) is nrn.nrn.Segment:
                imp.loc(0, sec = obj[key].sec)
                imp.compute(freq, 1)
                impedance = imp.input(obj[key].x, sec = obj[key].sec)
                phase = 180.0/np.pi*imp.input_phase(obj[key].x, sec = obj[key].sec)
                Zin[key] = np.array([[impedance, phase]])    
            
            elif type(obj[key]) is nrn.nrn.Section:
                imp.loc(0, sec = obj[key])
                imp.compute(freq, 1)
                Zin[key] = np.empty((obj[key].nseg, 2))
                for idx, seg in enumerate(obj[key]):
                    Zin[key][idx, 0] = imp.input(seg.x, sec = obj[key])
                    Zin[key][idx, 1] = 180.0/np.pi*imp.input_phase(seg.x, sec = obj[key])

            elif type(obj[key]) in [list, tuple]:
                segments = []
                for item in obj[key]:
                    if type(item) is nrn.nrn.Segment:
                        segments.append(item)
                    elif type(item) is nrn.nrn.Section:
                        segments.extend(list(item))
                imp.loc(0, sec = segments[0].sec)
                imp.compute(freq, 1)
                Zin[key] = np.empty((len(segments), 2))
                for idx, seg in enumerate(segments):
                    Zin[key][idx, 0] = imp.input(seg.x, sec = seg.sec)
                    Zin[key][idx, 1] = 180.0/np.pi*imp.input_phase(seg.x, sec = seg.sec)
            
        return Zin    
                
    elif obj is None:
        Zin = {}
        # calculate impedance for all sections and segments in hoc
        for sec in h.allsec():
            imp.loc(0, sec = sec)
            imp.compute(freq, 1)
            Zin[sec.name()] = np.empty((sec.nseg, 2))
            for idx, seg in enumerate(sec):
                Zin[sec.name()][idx, 0] = imp.input(seg.x, sec = sec)
                Zin[sec.name()][idx, 1] = 180.0/np.pi*imp.input_phase(seg.x, sec = sec)
                
    else:
        raise Exception("Object type '{}' not supported".format(type(obj)))

def meas_Vm(sim_ctrl, sections, refloc = None):
    """
    Given a current stimulation waveform, the function measures the membrane potential response in
    given sections (or section segments).
    
    IN
    
    sim_ctrl                SimulationControl object.
    sections                Iterable of nrn.nrn.Section objects.
    refloc                  Normalized distance within each section at which Vm will be measured.
                            If None, then Vm will be measured from all segments from the provided sections list.
    
    OUT
    
    rec                     Dictionary with keys 'v' and 't' for membrane potential in [mV] and time in [ms] as numpy arrays.
    """ 
    # record membrane potential response
    rec = record(sim_ctrl.h, sections, refloc=refloc)
    run_model(sim_ctrl)
    nrn_vect_to_np_array(rec)
    
    return rec

def get_segdata(h, obj, attr = None, sec_loc = None, ref_seg = None, use_path_dist = True, flatten = True):
    """
    Transforms an object containing sections and segments into another object with similar structure containing segments or 
    their attributes as well as their relative distances from a reference segment if needed.
    
    IN
    h               NEURON hoc object.
    obj             Can be any of the following or combinations thereof:
                    - nrn.nrn.Segment object.
                    - nrn.nrn.Section object.
                    - iterable.
                    - dictionary.
    attr            If None the function transforms sections and segments into segments, otherwise it returns segment attributes such as 'diam' or 'pas.e'.
    sec_loc         Normalized location within each section in the range [0, 1] from where to consider a segment from a section.
                    If None, then all segments within each section are considered.
    ref_seg         Reference segment used for distance measurements. 
                    If such a reference segment is provided then the output data will be:
                        - if attr is None: tuple of (<segment object>, <distance measurement>).
                        - if attr is not None: 2D numpy array where the 0-index is the attribute of the segment and 
                          the 1-index  is the distance between the ref_seg and the segment in question.
                    If ref_seg is None:
                        - if attr is None: segment object.
                        - if attr is not None: 2D numpy array where the 0-index is the attribute of the segment and 
                          the 1-index is the segment index within the section it belongs to (0-index means the end node of the section)
    use_path_dist   If True, uses path distance measurement between segments, otherwise it uses a direct distance measurement between the coordinates of the segments.
    flatten         If flatten is True, then numpy arrays and iterables in a list are concatenated.
     
    OUT
    1. obj = nrn.nrn.Segment
        1.2 attr = None -> [(nrn.nrn.Segment, distance_to_refseg)]
        1.2 attr = e.g. 'diam' or 'pas.e' -> np.ndarray of shape (1,2) with first dimension indexing the segment number (one segment in this case) and
                   the second dimension 0-idx = nrn.nrn.Segment.diam or nrn.nrn.Segment.pas.e and 1-idx = distance_to_refseg or segment index within its section. 
                  
    2. obj = nrn.nrn.Section
        2.1 sec_loc = None
            2.1.1 attr = None -> [(nrn.nrn.Segment_1, distance<ref_seg,nrn.nrn.Segment_1>), (nrn.nrn.Segment_2, distance<ref_seg, nrn.nrn.Segment_2>),...], i.e. list of tuples 
                         with distance between reference segment and a particular segments within the section and the segment object for all segments within the section. 
            2.1.1 attr = e.g. 'diam' or 'pas.e' -> np.ndarray of shape (obj.nseg,2) with first dimension indexing the segment number and the second 
                         dimension 0-idx = nrn.nrn.Segment.diam or nrn.nrn.Segment.pas.e and 1-idx = distance_to_refseg or segment index within its section.
        2.2 sec_loc = float or int in the interval [0, 1] -> function is called recursively on obj(sec_loc), i.e. on the nrn.nrn.Segment located at sec_loc within the obj nrn.nrn.Section.        
        
    3. obj = list or tuple -> the function is called recursively on each element and if flatten is True, numpy arrays and iterables in the list are concatenated.
       
    4. obj = dictionary -> function is called recursively on each key and the output is also a dictionary with the same keys as the original dictionary.         
    """
    if type(obj) is nrn.nrn.Segment:
        if ref_seg is None:
            # there is no reference segment that can be used for distance measurement, so just return a list with segments data
            if attr is None:
                out = [obj]
            else:
                out = np.empty((1,2))
                out[0,0] = get_attr_from_str(obj, attr)
                out[0,1] = obj.x
        else:
            # reference segment for distance measurement provided, pack distance together with requested segment attribute
            if attr is None:
                # form a tuple of segment distance to reference segment and nrn.nrn.Segment object
                out = [(obj, meas_seg_distance(h, ref_seg, obj, use_path_dist))]
            else:
                # pack segment distance to reference segment and requested value of segment attribute into a 2D numpy array with 0-index distance and 1-index attribute value
                out = np.empty((1,2))
                out[0,0] = get_attr_from_str(obj, attr)
                out[0,1] = meas_seg_distance(h, ref_seg, obj, use_path_dist)
        
    elif type(obj) is nrn.nrn.Section:
        if sec_loc is None:
            # get attribute from all segments
            if ref_seg is None:
                # there is no reference segment that can be used for distance measurement, so just return a list with segments data
                if attr is None:
                    out = [seg for seg in obj]
                else:
                    out = np.empty((obj.nseg, 2))
                    for idx, seg in enumerate(obj):
                        out[idx, 0] = get_attr_from_str(seg, attr)
                        out[idx, 1] = seg.x
            else:
                # reference segment for distance measurement provided, pack distance together with requested segment attribute
                if attr is None:
                    # form a list of tuples of segment distance to reference segment and nrn.nrn.Segment object
                    out = [(seg, meas_seg_distance(h, ref_seg, seg, use_path_dist)) for seg in obj]
                else:
                    # pack segment distance to reference segment and requested value of segment attribute into a 2D numpy array with 0-index distance and 1-index attribute value
                    out = np.empty((obj.nseg, 2))
                    for idx, seg in enumerate(obj):
                        out[idx, 0] = get_attr_from_str(seg, attr)
                        out[idx, 1] = meas_seg_distance(h, ref_seg, seg, use_path_dist)
        else:
            # get attribute only from segment at sec_loc within section
            out = get_segdata(h, obj(sec_loc), attr = attr, sec_loc = sec_loc, ref_seg = ref_seg, use_path_dist = use_path_dist, flatten = flatten)
            
    elif type(obj) is list or type(obj) is tuple:
        # apply function recursively to each list or tuple element
        objects = []
        for item in obj:
            objects.append(get_segdata(h, item, attr = attr, sec_loc = sec_loc, ref_seg = ref_seg, use_path_dist = use_path_dist, flatten = flatten))
        
        # combine segment data from all list elements if needed
        if flatten:
            if all(isinstance(x, np.ndarray) for x in objects):
                out = np.concatenate(objects)
            elif all(isinstance(x, collections.Iterable) for x in objects):
                out = []
                for x in objects:
                    out.extend(x)
        else:
            out = objects
        
        # convert output to tuple or keep as list
        if type(obj) is tuple:
            out = tuple(out)
                    
    elif isinstance(obj, dict):
        # call recursively on each key
        out = {}
        for key, val in obj.items():
            out[key] = get_segdata(h, val, attr = attr, sec_loc = sec_loc, ref_seg = ref_seg, use_path_dist = use_path_dist, flatten = flatten)
            
    else:
        out = None
        
    return out    

def meas_electrotonic_length(h, freq, seg1, seg2):
    """
    Measures the electrotonic length starting from seg1 to seg2 at a certain frequency between two segments.
    Note that L(seg1->seg2) != L(seg2->seg1) .
    This function uses the nrn.Impedance class.
    
    Parameters
    ----------
    h : neuron.hoc.HocObject
        NEURON hoc object.
    freq : float
        Test frequency in [Hz].
    seg1, seg2 : nrn.nrn.Segment
        Segments between which to measure the electrotonic distance.      
    
    Returns
    -------
    float
        Electrotonic length.
    """
    imp = h.Impedance()
    imp.loc(seg2.x, sec = seg2.sec)
    imp.compute(freq, 1)
    
    return np.abs(np.log(imp.ratio(seg1.x, sec = seg2.sec)))