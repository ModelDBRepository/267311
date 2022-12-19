# Morphology analysis and plotting
#
# Copyright (c) 2022 Adrian Negrean
# negreanadrian@gmail.com
#
# Software released under MIT license, see license.txt for conditions
from __future__ import division
import sys, os, re, numpy as np, pandas as pd, io

import matplotlib as mpl
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
from matplotlib import collections as mcol
from collections import OrderedDict, namedtuple

# create a point class named tuple
# use as:
# point = Point(2,4)
# point.x
# point.y
Point = namedtuple("Point", "x y")

# package for converting .hoc Neuron morphology to .SWC file format
from hoc2swc import hoc2swc # use pip install hoc2swc

global_yz_mark = []

from Bio import Phylo

import plotly.graph_objs as go
from plotly.offline import iplot, plot, init_notebook_mode

# manually add neuron to PYTHONPATH for this platform
nrn_python_path = '/usr/local/nrn/lib/python/'
if os.path.exists(nrn_python_path):
    sys.path.append(nrn_python_path)
    # note: if NEURON cannot open the display, it crashes the kernel
    # therefore, it's safe to remove the display key alltogether
    if os.environ.has_key('DISPLAY'):
        del os.environ['DISPLAY']
    from neuron import h as nrn_h, nrn
    nrn_imported = True
else:
    nrn_imported = False

class AnchoredHScaleBar(mpl.offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """
    def __init__(self, size = 1, extent = 0.03, label = "", loc = 2, ax = None,
                 pad = 0.4, borderpad = 0.5, ppad = 0, sep = 2, prop = None, 
                 frameon = True, linekw = {}, label_fontsize = 8, **kwargs):
        if not ax:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()
        size_bar = mpl.offsetbox.AuxTransformBox(trans)
        line = Line2D([0,size],[0,0], **linekw)
        vline1 = Line2D([0,0],[-extent/2.,extent/2.], **linekw)
        vline2 = Line2D([size,size],[-extent/2.,extent/2.], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = mpl.offsetbox.TextArea(label, minimumdescent = False, textprops = dict(fontsize = label_fontsize))
        self.vpac = mpl.offsetbox.VPacker(children = [size_bar,txt],  
                                 align = "center", pad = ppad, sep = sep) 
        mpl.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad = pad, 
                 borderpad = borderpad, child = self.vpac, prop = prop, frameon = frameon,
                 **kwargs)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def sort_legend_labels(ax):
    """
    Sorts order of legend entries by label name.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Plot axis.

    Returns
    -------
    tuple
        (handles, labels)
        Pass on to ax.legend()
    """
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    if handles and labels:
        labels, handles = zip(*sorted(zip(labels, handles), key = lambda t: t[0]))
    return handles, labels
    
def _phylo_tree(parent_sec = "soma", cb_func = None):
    """
    Traverses the morphology and applies a custom function to each section.
    This output style is to be used for phylogram plotting.
    
    Parameters
    ----------
    parent_sec : str
        Section name.
    cb_func : callable
        Callback function of the form cb_func(sec) with sec being a Neuron section object.
    
    Returns
    -------
    dict with section name and 0-end segment location tuple keys (<section name>, 0) and tuple values
    (cb_func output, dict) with callback function output applied to parent section and dict with child
    sections and cb_func results applied to each section.
    """
    nrn_parent_sec = get_secseg(h = nrn_h, secseg = parent_sec)
    sl = nrn_h.SectionList()
    sl.children(nrn_parent_sec)
    out = {}
    for sec in sl:
        out[(sec.name(), 0)] = _phylo_tree(sec.name(), cb_func = cb_func)
    
    return cb_func(nrn_parent_sec), out

def _get_tree(h, nrn_parent_sec, cb_func = None):
    """
    Traverses a given morphology tree and applies a custom function to each section.
    
    Parameters
    ----------
    nrn_parent_sec : hoc.Section
        Root section name.
    cb_func : callable
        Callback function of the form cb_func(sec) with sec being a Neuron section object.
    
    Returns
    -------
    dict
        Section name keys and tuple values (cb_func output, dict) with callback function output applied
        to parent section and dict with child sections and cb_func results applied to each section in a recursive way.
    """
    sl = h.SectionList()
    sl.children(nrn_parent_sec)
    out = {}
    for sec in sl:
        out[sec.name()] = _get_tree(h = h, nrn_parent_sec = sec, cb_func = cb_func)
    
    return cb_func(nrn_parent_sec), out

def get_tree(h, cb_func = None):
    """
    Traverses all morphology trees and applies a custom function to each section.
    
    Parameters
    ----------
    cb_func : callable
        Callback function of the form cb_func(sec) with sec being a Neuron section object.
    
    Returns
    -------
    dict
        Section name keys and tuple values (cb_func output, dict) with callback function output applied
        to parent section and dict with child sections and cb_func results applied to each section in a recursive way.
    """
    sl = h.SectionList()
    sl.allroots()
    out = {}
    for parent_sec in sl:
        out[parent_sec.name()] =_get_tree(h = h, nrn_parent_sec = parent_sec, cb_func = cb_func)
    return out

def convert_hoc_to_swc(h, fpath):
    """
    Convert .hoc morphology specification to .swc for compatibility with other software. Converted morphology file is saved in the same folder
    with same name as the original file.

    Parameters
    ----------
    fpath : str
      Absolute path to .hoc file

    Returns
    -------
    None
    """
    # delete all existing sections (including mechanisms and point processes) from hoc.
    # note: this seems to fail when creating sections with h.Section()
    assert h('forall delete_section()')
    hoc2swc(fpath, os.path.splitext(fpath)[0]+'.swc')

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
    # delete all existing sections (including mechanisms and point processes) from hoc.
    # note: this seems to fail when creating sections with h.Section()
    assert h('forall delete_section()')
    # execute hoc commands from file
    print("Opening HOC file '{}'".format(fpath))
    if not h('xopen("' + fpath + '")'):
        raise Exception("Execution of " + fpath + " failed.")
    print('done.')

# almost done
# taken from Alex Williams' PyNeuron-Toolbox package
def add_pre(h, sec_list, sec, order_list = None, branch_order = None):
    """
    Helper function that traverses a neuron's morphology (or a sub-tree)
    of the morphology in pre-order.

    Parameters
    ----------
    h : hoc.HocObject
        Neuron interpreter.
    sec_list : list
        List to use to populate with hoc.Section
    sec : nrn.Section
        Neuron section.

    Returns
    -------
    None
        Sections are added to sec_list
    """

    sec_list.append(sec)
    sref = h.SectionRef(sec = sec)

    if branch_order is not None:
        order_list.append(branch_order)
        if len(sref.child) > 1:
            branch_order += 1

    for next_node in sref.child:
        add_pre(h, sec_list, next_node, order_list, branch_order)

# taken from Alex Williams' PyNeuron-Toolbox package
def root_sections(h):
    """
    Returns a list of all sections that have no parent.

    Parameters
    ----------
    h : hoc.HocObject
        Neuron interpreter.

    Returns
    -------
    list of nrn.Section objects
    """
    roots = h.SectionList()
    roots.allroots()
    return list(roots)

def allsec_preorder(h):
    """
    Alternative to using h.allsec(). This returns all sections in order from
    the root. Traverses the topology each neuron in "pre-order".

    Parameters
    ----------
    h : hoc.HocObject
        Neuron interpreter.

    Returns
    -------
    list of nrn.Section objects
    """
    #Iterate over all sections, find roots
    roots = root_sections(h)

    # Build list of all sections
    sec_list = []
    for r in roots:
        add_pre(h,sec_list,r)
    return sec_list

def get_secseg(h, secseg):
    """
    Converts a string section or segment name to nrn.Section or nrn.Segment objects.

    Parameters
    ----------
    h : hoc.HocObject
        Neuron interpreter.
    secseg : nrn.Section, nrn.Segment, str
        Section or segment object or name such as 'soma[0](0.5)'.

    Returns
    -------
    nrn.Section or nrn.Segment object if found, otherwise None.
    """
    if type(secseg) in [str, unicode]:
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

    elif type(secseg) in [nrn.Segment, nrn.Section]:
        out = secseg
    else:
        out = None

    return out

# taken from Alex Williams' PyNeuron-Toolbox package
def get_section_path(h, sec):
    """
    Obtains cartesian coordinates along a Neuron section.

    Parameters
    ----------
    h : hoc.HocObject
        Neuron interpreter.
    sec : nrn.Section
        Neuron section.

    Returns
    -------
    numpy.ndarray
        Section coordinates of shape (n,3) where n is the number of coordinates.
    """
    n3d = int(h.n3d(sec = sec))
    xyz = []
    for i in range(0,n3d):
        xyz.append([h.x3d(i, sec = sec), h.y3d(i, sec = sec), h.z3d(i, sec = sec)])
    xyz = np.array(xyz)
    return xyz

# taken from Alex Williams' PyNeuron-Toolbox package
def cartesian_to_spherical(xyz):
    """
    Converts cartesian coordinates into spherical coordinates.

    Parameters
    ----------
    xyz : 2D numpy.ndarray
        Cartesian coordinates (x,y,z) on each row.

    Returns
    -------
    tuple
        (r, theta, phi) where:
            r : 1D numpy.array
                Spherical distance.
            theta : 1D numpy.array
                Angle in XY plane.
            phi : 1D numpy.array
                Angle from Z axis.
    """
    d_xyz = np.diff(xyz, axis = 0)

    r = np.linalg.norm(d_xyz, axis = 1)
    theta = np.arctan2(d_xyz[:,1], d_xyz[:,0])
    hyp = d_xyz[:,0]**2 + d_xyz[:,1]**2
    phi = np.arctan2(np.sqrt(hyp), d_xyz[:,2])

    return (r,theta,phi)

# taken from Alex Williams' PyNeuron-Toolbox package
def spherical_to_cartesian(r, theta, phi):
    """
    Converts spherical to cartesian coordinates.

    Parameters
    ----------
    r, theta, phi : float
        Scalar spherical coordinates.

    Returns
    -------
    tuple
        (x,y,z) cartesian coordinates.
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return (x,y,z)

# taken from Alex Williams' PyNeuron-Toolbox package
def interpolate_jagged(xyz, nseg):
    """
    Interpolates along a jagged path in 3D.

    Parameters
    ----------
        xyz : numpy.ndarray
            XYZ coordinates along a path, of shape (n,3), where n is the number of points.
        nseg : int
            Number of segments along path.

    Returns
    -------
        list of numpy.array
            Segment paths.
    """

    # Spherical coordinates specifying the angles of all line
    # segments that make up the section path
    (r,theta,phi) = cartesian_to_spherical(xyz)

    # cumulative length of section path at each coordinate
    rcum = np.append(0,np.cumsum(r))

    # breakpoints for segment paths along section path
    breakpoints = np.linspace(0,rcum[-1],nseg+1)
    np.delete(breakpoints,0)

    # Find segment paths
    seg_paths = []
    for a in range(nseg):
        path = []

        # find (x,y,z) starting coordinate of path
        if a == 0:
            start_coord = xyz[0,:]
        else:
            start_coord = end_coord # start at end of last path
        path.append(start_coord)

        # find all coordinates between the start and end points
        start_length = breakpoints[a]
        end_length = breakpoints[a+1]
        mid_boolean = (rcum > start_length) & (rcum < end_length)
        mid_indices = np.nonzero(mid_boolean)[0]
        for mi in mid_indices:
            path.append(xyz[mi,:])

        # find (x,y,z) ending coordinate of path
        end_coord = find_coord(end_length,xyz,rcum,theta,phi)
        path.append(end_coord)

        # Append path to list of segment paths
        seg_paths.append(np.array(path))

    # Return all segment paths
    return seg_paths

# taken from Alex Williams' PyNeuron-Toolbox package
def find_coord(targ_length, xyz, rcum, theta, phi):
    """
    Find (x,y,z) ending coordinate of segment path along section path.

    Parameters
    ----------
    targ_length : float
        Length of segment path starting from the begining of the section path.
    xyz : numpy.ndarray
        XYZ coordinates describing the section path of shape (n,3), where n is the number of points.
    rcum : float
        Cumulative sum of section path length at each node in xyz.
    theta, phi : float
        Angles for each XYZ point.

    Returns
    -------
    numpy.array
        XYZ coordinates.
    """
    #   [1] Find spherical coordinates for the line segment containing
    #           the endpoint.
    #   [2] Find endpoint in spherical coords and convert to cartesian
    i = np.nonzero(rcum <= targ_length)[0][-1]
    if i == len(theta):
        return xyz[-1,:]
    else:
        r_lcl = targ_length-rcum[i] # remaining length along line segment
        (dx,dy,dz) = spherical_to_cartesian(r_lcl,theta[i],phi[i])
    return xyz[i,:] + [dx,dy,dz]

def add_annotation(h, sec, locs, ax, annotation_text = "", annotation_text_size = 12, markspec = 'or',
    annotation_color = (1,0,0,0.75), marker_color = (1,0,0,0.75), legend_text = "", **kwargs):
    """
    Marks a segment location.

    Parameters
    ----------
    h : hoc.HocObject
        Neuron interpreter.
    
    sec : nrn.Section, str
        Neuron section object or section name.
    
    locs : float, array_like of float
        Segment locations, between 0 and 1.
    
    ax : matplotlib.axes._subplots.AxesSubplot
        Plotting axes.
    
    annotation_text : str
        Annotation text.

    annotation_text_size : float
                Annotation font size in points.
    
    markspec : str
        Marker specifier.
    
    annotation_color : tuple, str
        Annotation color.
            If tuple:
                - RGB (r,g,b)
                - RGB+alpha (r,g,b,a)
            If str:
                - Named color.
                - Hex color
            Or any other valid matplotlib color specifier.

    marker_color : tuple, str
        Marker color, similar to annotation_color specification.

    legend_text : str
        Adds a legend label to the marker.
    
    kwargs : other optional pyplot parameters
    """
    # get list of cartesian coordinates specifying section path
    xyz_global = []

    sections = allsec_preorder(h) # Get sections in "pre-order"

    for _sec in sections:
        xyz_list = get_section_path(nrn_h,_sec).tolist()
        xyz_global.extend(xyz_list)
    xyz_global = np.array(xyz_global)
    global_x = xyz_global[:, 0]
    global_y = xyz_global[:, 1]
    global_z = xyz_global[:, 2]

    xyz = get_section_path(h, get_secseg(h,sec))
    (r,theta,phi) = cartesian_to_spherical(xyz)
    rcum = np.append(0,np.cumsum(r))

    # convert locs into lengths from the beginning of the path
    if type(locs) is float or type(locs) is np.float64:
        locs = np.array([locs])
    if type(locs) is list:
        locs = np.array(locs)
    lengths = locs*rcum[-1]

    # find cartesian coordinates for markers
    xyz_marks = []
    for targ_length in lengths:
        xyz_marks.append(find_coord(targ_length, xyz, rcum, theta, phi))
    xyz_marks = np.array(xyz_marks)

    # plot markers
    line, = ax.plot(xyz_marks[:,0], xyz_marks[:,1], xyz_marks[:,2], markspec, color = marker_color, label = legend_text, **kwargs)

    x_mark = xyz_marks[0,0]
    if xyz_marks[:, 1] > (global_y.min() + global_y.max()) / 2:
        y_mark = global_y.max()
    if xyz_marks[:, 1] <= (global_y.min() + global_y.max()) / 2:
        y_mark = global_y.min() - 0.3 * (global_y.max()-global_y.min())/2
    z_mark = xyz_marks[0, 2]

    if global_yz_mark != []:
        i = 0
        while True:
            mark1 = 0
            mark2 = 0
            for idx in global_yz_mark:
                if y_mark==idx[0] and z_mark-idx[1] <= 10 and z_mark-idx[1] >= 0:
                    mark1 = 1
                    break
                if y_mark==idx[0] and idx[1]-z_mark <= 10 and idx[1]-z_mark >= 0:
                    mark2 = 1
                    break
            if mark1 == 0 and mark2 == 0:
                break
            if i < 5 and mark1 == 1 and mark2 == 0:
                z_mark = z_mark + 12
            if i < 5 and mark1 == 0 and mark2 == 1:
                z_mark = z_mark - 12
            if i >=5 or (mark1 == 1 and mark2 == 1):
                y_mark = y_mark + 0.35 * (global_y.max()-global_y.min())/2
            i = i + 1

    global_yz_mark.append([y_mark, z_mark])
    if y_mark > (global_y.min() + global_y.max()) / 2:
        y_mark_arr = y_mark
    if y_mark <= (global_y.min() + global_y.max()) / 2:
        y_mark_arr = global_y.min()

    if annotation_text:
        ax.text3D(x_mark, y_mark, z_mark+3, s = annotation_text, color = annotation_color, fontsize = annotation_text_size)
        arw = Arrow3D([xyz_marks[0,0],x_mark],[xyz_marks[0,1],y_mark_arr],[xyz_marks[0,2],z_mark],
                      arrowstyle = "-|>", color = annotation_color, lw = 0.5,
                      mutation_scale = 15)
        ax.add_artist(arw)

def perpendicular_vector(orientation,radius,number):
    res = []
    for i in range(number):
        x = np.random.randn(3)
        x = np.cross(orientation, x)
        x = x / np.linalg.norm(x)
        x2 = np.multiply(x,radius)
        res.append(x2)
    return res

# adapted from Alex Williams' PyNeuron-Toolbox package
def weighed_shapeplot(h, ax, sections = None, order = 'pre', cvals = None, median_lw = 1, clim = None, cmap = cm.YlOrBr_r, **kwargs):
    """
    Plots a 3D shapeplot.

    Parameters
    ----------
        h : hoc.HocObject
            Neuron interpreter.
        ax : matplotlib.axes._subplots.AxesSubplot
            Plotting axes.
        sections : list of h.Section()
            Sections to plot.
        order : None, str
            choose:
            None : use h.allsec() to get sections
            'pre' : pre-order traversal of morphology
        cvals : iterable
            Values mapped to color by cmap; useful for displaying voltage, calcium or some other state
            variable across the shapeplot.

        median_lw : float
            Median line weight.

        kwargs passed on to matplotlib (e.g. color='r' for red lines)

    Returns
    -------
        line : matplotlib.lines.Line2D
            Plotted line.
    """

    median_diameter = np.median([sec.diam for sec in h.allsec()])

    # Default is to plot all sections.
    if sections is None:
        if order == 'pre':
            sections = allsec_preorder(h) # Get sections in "pre-order"
        else:
            sections = list(h.allsec())

    # Determine color limits
    if cvals is not None and clim is None:
        cn = [ isinstance(cv, numbers.Number) for cv in cvals ]
        if any(cn):
            clim = [np.min(cvals[cn]), np.max(cvals[cn])]

    # Plot each segment as a line
    lines = []
    i = 0

    for sec in sections:
        xyz = get_section_path(h,sec)
        xyz_list = get_section_path(h,sec).tolist()
        seg_paths = interpolate_jagged(xyz, sec.nseg)


        # plot all sections except the soma
        if sec != sections[0]:
            for (j, path) in enumerate(seg_paths):
                line, = ax.plot(path[:,0], path[:,1], path[:,2], '-k', lw = sec.diam*median_lw/median_diameter, solid_capstyle = 'round', alpha = 0.75, **kwargs)
                if cvals is not None:
                    if isinstance(cvals[i], numbers.Number):
                        # map number to colormap
                        col = cmap(int((cvals[i]-clim[0])*255/(clim[1]-clim[0])))
                    else:
                        # use input directly. E.g. if user specified color with a string.
                        col = cvals[i]
                    line.set_color(col)
                lines.append(line)


        # plot soma section
        if sec == sections[0]:
            points_vert = []

            for j in range(len(xyz_list)-1):
                if j == 0:
                    orientation = [xyz_list[j + 1][0] - xyz_list[j][0], xyz_list[j + 1][1] - xyz_list[j][1],
                                   xyz_list[j + 1][2] - xyz_list[j][2]]
                    point_x = xyz_list[j][0]
                    point_y = xyz_list[j][1]
                    point_z = xyz_list[j][2]
                    center = [point_x, point_y, point_z]
                    for per in range(0,100,10):
                        res = perpendicular_vector(orientation = orientation,
                                                   radius = per*sec.diam * median_lw / median_diameter/100, number = 20)
                        for idx2 in res:
                            points_vert.append(center + idx2)
                if j == len(xyz_list) - 2:
                    orientation = [xyz_list[j + 1][0] - xyz_list[j][0], xyz_list[j + 1][1] - xyz_list[j][1],
                                   xyz_list[j + 1][2] - xyz_list[j][2]]
                    point_x = xyz_list[j+1][0]
                    point_y = xyz_list[j+1][1]
                    point_z = xyz_list[j+1][2]
                    center = [point_x, point_y, point_z]
                    for per in range(0,100,10):
                        res = perpendicular_vector(orientation = orientation,
                                                   radius = per*sec.diam * median_lw / median_diameter/100, number = 20)
                        for idx2 in res:
                            points_vert.append(center + idx2)

                orientation = [xyz_list[j + 1][0] - xyz_list[j][0], xyz_list[j + 1][1] - xyz_list[j][1],
                               xyz_list[j + 1][2] - xyz_list[j][2]]
                for idx in range(10):
                    point_x = xyz_list[j+1][0] - idx * ((xyz_list[j + 1][0] - xyz_list[j][0]) / 10)
                    point_y = xyz_list[j+1][1] - idx * ((xyz_list[j + 1][1] - xyz_list[j][1]) / 10)
                    point_z = xyz_list[j+1][2] - idx * ((xyz_list[j + 1][2] - xyz_list[j][2]) / 10)
                    center = [point_x,point_y,point_z]
                    res = perpendicular_vector(orientation = orientation, radius = sec.diam*median_lw/median_diameter, number = 10)
                    for idx2 in res:
                        points_vert.append(center + idx2)

            hull = ConvexHull(points_vert)
            faces = hull.simplices

            # Apply color to face
            vert_list = []
            for arr in points_vert:
                arr_list = arr.tolist()
                vert_list.append(arr_list)
            vert_arr = np.asarray(vert_list, dtype = np.float32)


            ls = LightSource(azdeg = 225.0, altdeg = 45.0)

            # First change - normals are per vertex, so I made it per face.
            normalsarray = np.array([np.array((np.sum(vert_arr[face[:], 0] / 3), np.sum(vert_arr[face[:], 1] / 3),
                                               np.sum(vert_arr[face[:], 2] / 3)) / np.sqrt(
                np.sum(vert_arr[face[:], 0] / 3) ** 2 + np.sum(vert_arr[face[:], 1] / 3) ** 2 + np.sum(
                    vert_arr[face[:], 2] / 3) ** 2)) for face in faces])

            # Next this is more asthetic, but it prevents the shadows of the image being too dark. (linear interpolation to correct)
            shade_min = np.min(ls.shade_normals(normalsarray, fraction = 1.0))  # min shade value
            shade_max = np.max(ls.shade_normals(normalsarray, fraction = 1.0))  # max shade value
            diff = shade_max - shade_min
            newMin = 0.3
            newMax = 0.95
            newdiff = newMax - newMin

            # Using a constant color, put in desired RGB values here.
            colourRGB = np.array((169 / 255.0, 169 / 255.0, 169 / 255.0))

            # The correct shading for shadows are now applied. Use the face normals and light orientation to generate a shading value and apply to the RGB colors for each face.
            rgbNew = np.array([colourRGB * (newMin + newdiff * ((shade - shade_min) / diff)) for shade in
                               ls.shade_normals(normalsarray, fraction = 1.0)])


            for i in range(len(faces)):
                sq = [
                    [points_vert[faces[i][0]][0], points_vert[faces[i][0]][1], points_vert[faces[i][0]][2]],
                    [points_vert[faces[i][1]][0], points_vert[faces[i][1]][1], points_vert[faces[i][1]][2]],
                    [points_vert[faces[i][2]][0], points_vert[faces[i][2]][1], points_vert[faces[i][2]][2]]
                ]
                f = a3.art3d.Poly3DCollection([sq], linewidths = 0)
                f.set_facecolor(colors.rgb2hex(rgbNew[i]))
                f.set_edgecolor('k')
                f.set_alpha(0.75)
                ax.add_collection3d(f)

        i += 1

    return lines

def plot_morphology(hoc_fpath, annotation_df = None, ax = None, median_lw = 1, legend_fontsize = 12,
    legend_ncol = 3, legend_zoffset = -50, figsize = (10,10), default_marker_size = 5,
    default_marker_color = (1,0,0,0.5), open_hoc_file = True):
    """
    Plot morphology from .hoc file

    Parameters
    ----------
    hoc_fpath : str
        NEURON .hoc morphology file.

    annotation_df : None, pandas.DataFrame
        Annotation information as dataframe with columns:
            'section' : str
                Section name, e.g. 'soma' or 'dend[10]'.

            'location' : float
                Normalized segment location within the section [0,1].

            'marker_color' : tuple, str, default (1,0,0,0.5)
                Segment marker color.
                If tuple:
                    - RGB (r,g,b)
                    - RGB+alpha (r,g,b,a)
                If str:
                    - Named color.
                    - Hex color
                Or any other valid matplotlib color specifier.

            'annotation_color' : tuple, str, default (1,0,0,0.75)
                Segment annotation color. Same description as for 'marker_color'.

            'marker_size' : float, default 5
                Annotation marker size in points.

            'annotation_text' : str
                Annotation label.

            'annotation_text_size' : float, default 12
                Annotation font size in points.

            'legend_text' : str
                Legend text for this marker.

    ax : None, matplotlib.axes._subplots.AxesSubplot
        Plotting axes with '3d' projection.

    median_lw : float
        Median line weight used for plotting segments. Adjusting this parameter scales the thickness of segments.

    legend_fontsize : float
        Legend font size in points.

    legend_ncol : int
        Number of legend columns.

    legend_zoffset : float
        Legend z-axis offset in data corrdinates. More negative values will shift the legend towards the top.

    default_marker_size : float
        Marker size in pt.

    default_marker_color : tuple, str, default (1,0,0,0.75)
        Segment annotation color. Same description as for 'marker_color'.

    open_hoc_file : bool
        If True, opens morphology again. Set to False if this is not needed and the same morphology can be used for different
        sequential plots.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Plotting axes.
    """
    # clear hoc of all sections and mechanisms and load .hoc file
    if open_hoc_file:
        load_hoc_file(nrn_h, hoc_fpath)

    # shapeplot
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize = figsize, subplot_kw = {'projection': '3d'}, constrained_layout = True)
    else:
        fig = None

    ax.view_init(elev = 0, azim = 0)

    weighed_shapeplot(nrn_h, ax, median_lw = median_lw)

    # remove grid lines
    ax.grid(False)
    
    # remove panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # transparent spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # remove ticks
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])

    # remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.invert_zaxis()

    section_lookup = {str(sec): sec for sec in nrn_h.allsec()}

    if annotation_df is not None:
        for _, loc in annotation_df.iterrows():
            add_annotation(h = nrn_h, sec = section_lookup[loc.section], locs = loc.location,
                annotation_text = "", # annotation not working properly # loc.annotation_text if 'annotation_text' in loc and not pd.isnull(loc.annotation_text) else "",
                ax = ax,
                annotation_color = loc.annotation_color if 'annotation_color' in loc and not pd.isnull(loc.annotation_color) else (1,0,0,0.75),
                annotation_text_size = loc.annotation_text_size if 'annotation_text_size' in loc and not pd.isnull(loc.annotation_text_size) else 12,
                marker_color = loc.marker_color if 'marker_color' in loc and not pd.isnull(loc.marker_color) else default_marker_color,
                markersize = loc.marker_size if 'marker_size' in loc and not pd.isnull(loc.marker_size) else default_marker_size,
                legend_text = loc.legend_text if 'legend_text' in loc and not pd.isnull(loc.legend_text) else ""
                )
    # add legend if there are any legend labels
    if 'legend_text' in annotation_df and not annotation_df['legend_text'].isnull().all():
        handles, labels = ax.get_legend_handles_labels()

        if 'marker_color' in annotation_df:
            marker_colors = [mc if not pd.isnull(mc) else default_marker_color for mc in annotation_df['marker_color']]
        else:
            marker_colors = [default_marker_color]*len(labels)
        by_label = OrderedDict(zip(zip(labels, marker_colors), handles))
        if by_label:
            # transform object corrdinates to projection coordinates
            f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
            labels, handles = zip(*sorted(zip([k[0] for k in by_label.keys()], by_label.values()), key = lambda t: t[0]))
            
            ax.legend(handles, labels, loc = 'lower left', fontsize = legend_fontsize,
                ncol = legend_ncol, bbox_to_anchor = f(ax.get_xlim()[0],ax.get_ylim()[0],ax.get_zlim()[1]+legend_zoffset), bbox_transform = ax.transData)

    return fig, ax

def _to_newick(morph_tree, root_name = 'ROOT'):
    """
    Converts a hierarchical tree into Newick format.

    Parameters
    ----------
    morph_tree : dict
        Sections tree.
    """
    dist, tree_dict = morph_tree
    level = [_to_newick(mtree, root_name = _rename_braces(name[0])) 
                 for name, mtree in tree_dict.iteritems()]
    rest = "(" + ','.join(level) + ")" if level else ""
    return "{}{}:{}".format(rest, _rename_braces(root_name), dist)

def _get_circular_tree_data(hoc_fpath, annotation_df = None, dist = 1, start_angle = 0, end_angle = 360,
    start_leaf = 'last', plotter = 'pyplot', angular_nseg = 20, default_marker_size = 5, default_marker_color = (1,0,0,0.5),
    open_hoc_file = True, trunk_linewidth = 1, nontrunk_linewidth = 0.35):
    """
    Generates data needed to get the Plotly plot of a circular tree.

    Parameters
    ----------
    hoc_fpath : str
        NEURON .hoc morphology file.

    annotation_df : None, pandas.DataFrame
        Annotation information as dataframe with columns:
            'section' : str
                Section name, e.g. 'soma' or 'dend[10]'.

            'location' : float
                Normalized segment location within the section [0,1].

            'marker_color' : tuple, str, default (1,0,0,0.5)
                Segment marker color.
                If tuple:
                    - RGB (r,g,b)
                    - RGB+alpha (r,g,b,a)
                If str:
                    - Named color.
                    - Hex color
                Or any other valid matplotlib color specifier.

            'annotation_color' : tuple, str, default (1,0,0,0.75)
                Segment annotation color. Same description as for 'marker_color'.

            'marker_size' : float, default 5
                Annotation marker size in points.

            'annotation_text' : str
                Annotation label.

            'annotation_text_size' : float, default 12
                Annotation font size in points.

            'legend_text' : str
                Legend text for this marker.

    dist : float
        The vertical distance between two consecutive leafs in the associated rectangular tree layout.
    start_angle : float
        Angle in [deg] representing the angle of the first leaf mapped to a circle.
    end_angle: float
        Angle in [deg] representing the angle of the last leaf.
    start_leaf: str
        Choose between:
            'first': to map leafs in a counter-clockwise direction.
            'last': to map leafs in a clockwise direction.
    plotter : str
        Choose between plotting engines to format line styles and colors appropriately: 'pyplot', 'plotly'.
    angular_nseg : int
        Number of angular segment to use for tree arcs.
    open_hoc_file : bool
        If True, opens morphology again. Set to False if this is not needed and the same morphology can be used for different
        sequential plots.
    trunk_linewidth : float
        Line width for trunk sections in pt.
    nontrunk_linewidth : float
        Line width for non-trunk section in pt.

    Returns
    -------
    lines, line_widths, arcs, arc_widths, annotation

    lines : Point class
        XY coordinates of radial sections.
        lines.x and lines.y are of type list
    line_widths : list of float
        Radial line widths in pt.
    arcs : Point class
        XY coordinates of circular arcs connecting the sections.
        arcs.x and arcs.y are of type list
    arc_widths : list of float
        Angular line widths in pt.
    annotation : pandas.DataFrame
        Clade information with columns:
        'x' : float
            X coordinate.
        'y' : float
            Y coordinate.
        'segment' : str
            Segment name.
        'marker_symbol' : str
            Marker symbol.
        'marker_color' : str
            Marker color.
        'marker_size' : str
            Marker size.
        'legend_text' : str
            Add legend text to this marker.
    """
    def get_radius(tree):
        """
        Associates to each clade root its radius, equal to the distance from that clade to the tree root
        returns dict {clade: node_radius}
        """
        node_radius = tree.depths()
        
        #  If the tree did not record the branch lengths assign the unit branch length
        #  (ex: the case of a newick tree "(A, (B, C), (D, E))")
        if not np.count_nonzero(node_radius.values()):
            node_radius = tree.depths(unit_branch_lengths = True)
        return node_radius
   
    def get_vertical_position(tree):
        """
        returns a dict {clade: ycoord}, where y-coord is the cartesian y-coordinate 
        of a  lade root in a rectangular phylogram
        
        """
        n_leafs = tree.count_terminals() # Counts the number of tree leafs.
        
        # Assign y-coordinates to the tree leafs
        # note: the list of leafs mapped in anticlockwise direction onto circles can be tree.get_terminals() 
        #       or its reversed version tree.get_terminals()[::-1]. 
        if start_leaf == 'first':
            node_ycoord = dict((leaf, k) for k, leaf in enumerate(tree.get_terminals()))
        elif start_leaf == 'last':
            node_ycoord = dict((leaf, k) for k, leaf in enumerate(reversed(tree.get_terminals())))
        else:
            raise ValueError("start leaf can be only 'first' or 'last'")
            
        def assign_ycoord(clade):#compute the y-coord for the root of this clade
            for subclade in clade:
                if subclade not in node_ycoord: # if the subclade root hasn't a y-coord yet
                    assign_ycoord(subclade)
            node_ycoord[clade] = 0.5 * (node_ycoord[clade.clades[0]] + node_ycoord[clade.clades[-1]])

        if tree.root.clades:
            assign_ycoord(tree.root)
        return node_ycoord

    def ycoord2theta(y):
        # maps an y in the interval [ymin-dist, ymax] to the interval [radian(start_angle), radian(end_angle)]
        return start_angle + (end_angle - start_angle) * (y-ymin) / float(ymax-ymin) 

    def get_points_on_lines(linetype = 'radial', linestyle = 'pyplot', angular_nseg = 20, x_left = 0, x_right = 0, y_right = 0, y_bot = 0, y_top = 0):
        """
        - define the points that generate a radial branch and the circular arcs, perpendicular to that branch
         
        - a circular arc (angular linetype) is defined by 10 points on the segment of ends
        (x_bot, y_bot), (x_top, y_top) in the rectangular layout,
         mapped by the polar transformation into 10 points that are spline interpolated
        - returns for each linetype the lists X, Y, containing the x-coords, resp y-coords of the
        line representative points

        Parameters
        ----------
        linetype : str
            Choose between 'radial' and 'angular'.
        linestyle : str
            Choose between pyplot or plotly compatibility: 'pyplot' or 'plotly'
        angular_nseg : int
            Number of angular segment per arc.

        Returns
        -------
        tuple
        (x,y)
            x : list
                x-axis coordinates of point on the line/arc
            y : list
                y-axis coordinates of point on the line/arc
        """
       
        if linetype == 'radial':
            theta = ycoord2theta(y_right) 
            if linestyle == 'pyplot':
                x = [x_left*np.cos(theta), x_right*np.cos(theta)]
                y = [x_left*np.sin(theta), x_right*np.sin(theta)]
            elif linestyle == 'plotly':
                x = [x_left*np.cos(theta), x_right*np.cos(theta), None]
                y = [x_left*np.sin(theta), x_right*np.sin(theta), None]
            else:
                raise ValueError("linestyle can be 'pyplot' or 'plotly'.")
        
        elif linetype == 'angular':
            theta_b = ycoord2theta(y_bot)
            theta_t = ycoord2theta(y_top)
            t = np.linspace(0, 1, angular_nseg)
            theta = (1-t) * theta_b + t * theta_t
            if linestyle == 'pyplot':
                x = list(x_right * np.cos(theta))
                y = list(x_right * np.sin(theta))
            elif linestyle == 'plotly': 
                x = list(x_right * np.cos(theta)) + [None]
                y = list(x_right * np.sin(theta)) + [None]
            else:
                raise ValueError("linestyle can be 'pyplot' or 'plotly'.")
        
        else:
            raise ValueError("linetype can be only 'radial' or 'angular'.")
       
        return x,y   
        
    def get_line_lists(clade, nrn_trunk_section_names_set, x_left, xlines, ylines, xarcs, yarcs,
        line_widths, arc_widths, linestyle = 'pyplot', angular_nseg = 20, trunk_linewidth = 1,
        nontrunk_linewidth = 0.35):
        """
        Recursively compute the lists of points that span the tree branches.

        Parameters
        ----------
        clade : Bio.Phylo.Newick.Clade
            Clade to start from.
        nrn_trunk_section_names_set : set
            Set containing NEURON section names belonging to the trunk section list.
        xlines, ylines : list
            The lists of x-coords, resp y-coords of radial edge ends.
        xarcs, yarcs : list
            The lists of points generating arc segments for tree branches.
        line_widths, arc_widths : list
            Line widths assigned for plotting radial and agular lines.
        linestyle : str
            Choose between 'pyplot' and 'plotly' for rendering.
        angular_nseg : int
            Number of angular line segments.
        trunk_linewidth : float
            Trunk clade line thickness in pt.
        nontrunk_linewidth : float
            Non-trunk clade line thickness in pt.
        """
        x_right = node_radius[clade]
        y_right = node_ycoord[clade]

        if nrn_trunk_section_names_set:
            if clade.name == 'ROOT':
                clade_linewidth = trunk_linewidth
            else:
                # get NEURON section name
                nrn_sec_name = clade.name.split('<')[0].replace('{', '[').replace('}', ']')
                if nrn_sec_name in nrn_trunk_section_names_set:
                    clade_linewidth = trunk_linewidth
                else:
                    clade_linewidth = nontrunk_linewidth
        else:
            clade_linewidth = nontrunk_linewidth

        line_widths.append(clade_linewidth)
   
        x,y = get_points_on_lines(linetype = 'radial', linestyle = linestyle,
            x_left = x_left, x_right = x_right, y_right = y_right)
   
        if linestyle == 'pyplot':
            xlines.append(x)
            ylines.append(y)
        elif linestyle == 'plotly':
            xlines.extend(x)
            ylines.extend(y)
        else:
            raise ValueError("linestyle can be 'pyplot' or 'plotly'.")
   
        if clade.clades:
           
            arc_widths.append(clade_linewidth)

            y_top = node_ycoord[clade.clades[0]]
            y_bot = node_ycoord[clade.clades[-1]]
       
            x,y = get_points_on_lines(linetype = 'angular', linestyle = linestyle, angular_nseg = angular_nseg,
                x_right = x_right, y_bot = y_bot, y_top = y_top)
            if linestyle == 'pyplot':
                xarcs.append(x)
                yarcs.append(y)
            elif linestyle == 'plotly':    
                xarcs.extend(x)
                yarcs.extend(y)
            else:
                raise ValueError("linestyle can be 'pyplot' or 'plotly'.")
       
            # get and append the lists of points representing the branches of the descedants
            for child in clade:
                get_line_lists(clade = child, nrn_trunk_section_names_set = nrn_trunk_section_names_set,
                    x_left = x_right, xlines = xlines, ylines = ylines,
                    xarcs = xarcs, yarcs = yarcs, line_widths = line_widths, arc_widths = arc_widths,
                    linestyle = linestyle, angular_nseg = angular_nseg, trunk_linewidth = trunk_linewidth,
                    nontrunk_linewidth = nontrunk_linewidth)

    def get_parent(tree, child_clade):
        node_path = tree.get_path(child_clade)
        return node_path[-2]

    # clear hoc of all sections and mechanisms and load .hoc file
    if open_hoc_file:
        load_hoc_file(nrn_h, hoc_fpath)
    # construct morphology tree with section lengths in [um] starting from 'soma' named section
    morph_tree = _phylo_tree("soma", lambda sec: sec.L)
    # insert location marks as extra nodes in the tree keyed using segment notation convention "<section name>(<segment location>)"
    
    if annotation_df is not None:
        # annotate by segment
        annotation_df["segment"] = ["{}({})".format(sec,loc) for sec, loc in zip(annotation_df["section"], annotation_df["location"])]
        annotation_df = annotation_df.set_index("segment")
        morph_tree = _morph_tree_insert(morph_tree, zip(annotation_df["section"], annotation_df["location"]))
    
    # convert morphology tree to Newick format
    newick_fmt = _to_newick(morph_tree)
    # create phylo tree
    tree = Phylo.read(io.StringIO(newick_fmt), "newick")

    start_angle *= np.pi/180 # conversion to radians
    end_angle *= np.pi/180
    
    node_radius = get_radius(tree)
    node_ycoord = get_vertical_position(tree)
    y_vals = node_ycoord.values()
    ymin, ymax = min(y_vals), max(y_vals)
    ymin -= dist # this dist subtraction is necessary to avoid coincidence of the first and last leaf angle
                 # when the interval  [ymin, ymax] is mapped onto [0, 2pi],
                
    if hasattr(nrn_h, 'trunk'):
        # note: turning this into a set speeds up lookup
        nrn_trunk_section_names = set([sec.name() for sec in nrn_h.trunk])
    else:
        nrn_trunk_section_names = set()

    xlines = []
    ylines = []
    xarcs = []
    yarcs = []
    line_widths = []
    arc_widths = []
    get_line_lists(clade = tree.root, nrn_trunk_section_names_set = nrn_trunk_section_names, x_left = 0,
        xlines = xlines, ylines = ylines, xarcs = xarcs, yarcs = yarcs, line_widths = line_widths,
        arc_widths = arc_widths, linestyle = plotter, angular_nseg = angular_nseg, trunk_linewidth = trunk_linewidth,
        nontrunk_linewidth = nontrunk_linewidth)

    annotation = []
    unique_markings = {}
    for clade in tree.find_clades(order = 'preorder'): #it was 'level'
        # annotate only segments
        if clade.name[-1] != '>':
            continue
        parent_clade = get_parent(tree, clade)
        theta = ycoord2theta(node_ycoord[parent_clade])
        renamed_clade = clade.name.replace('>',')').replace('<','(').replace('{','[').replace('}',']') 
        marking_df = annotation_df.loc[[renamed_clade]]
        
        for m_idx, m in marking_df.iterrows():
            legend_text = m.legend_text if 'legend_text' in m and not pd.isnull(m.legend_text) else ""
            if legend_text and (m_idx, legend_text) in unique_markings:
                continue
            else:
                if legend_text:
                    unique_markings[(m_idx, legend_text)] = None       
                annotation.append({
                    "x": node_radius[parent_clade]*np.cos(theta), 
                    "y": node_radius[parent_clade]*np.sin(theta),
                    "segment": renamed_clade,
                    "marker_color": m.marker_color if 'marker_color' in m and not pd.isnull(m.marker_color) else default_marker_color,
                    "marker_symbol": 'circle', # need to take from annotation df
                    "marker_size": m.marker_size if 'marker_size' in m and not pd.isnull(m.marker_size) else default_marker_size,
                    "legend_text": legend_text,
                    "annotation_text": m.annotation_text
                })

    return Point(xlines, ylines), line_widths, Point(xarcs, yarcs), arc_widths, pd.DataFrame(annotation)

def _rename_braces(name):
    """
    Replaces [, ] with with {, } and (, ) with <, >
    """ 
    return name.replace("[", "{").replace("(", "<").replace(")", ">").replace("]", "}")

def _parse_segment(segment):
    """
    Turns 'dend[40](0.1)' into
        {'node': 'dend[40]', 'rel_dist': 0.1}
    """
    node_name, dist_info = segment.split("(")
    rel_dist, _ = dist_info.split(")")
    
    return {
        'node': node_name,
        'rel_dist': float(rel_dist),
        'segment': segment
    }
    
def _morph_tree_insert(morph_tree, segments):
    """
    Insert segment nodes in the morphology tree.

    Parameters
    ----------
    morph_tree : dict
        Morphology tree keyed by section name with values equal to section lengths
    segments : list of 2 element iterable
        2 element iterable of section name and normalized segment location ["<section name>", "<segment location>"].

    Returns
    -------
    dict
        Adjusted morphology tree keyed by section names, e.g. "dend[10]".
    """
    inserted_morph_tree = morph_tree
    for seg in segments:
        inserted_morph_tree = _morph_tree_insert_single(morph_tree = inserted_morph_tree, sec_name = seg[0], loc = seg[1])
        
    return inserted_morph_tree
    
def _morph_tree_insert_single(morph_tree, sec_name, loc):
    """
    Insert a single segment in the hieraqrchical morphology.

    Parameters
    ----------
    morph_tree : dict
        Hierarchical morphology.
    sec_name : str
        Section name.
    loc : float
        Normalized segment location within a section. Must be [0,1].
    """
    assert 0<=loc<=1
    dist, tree_dict = morph_tree
    inserted_tree_dict = {}
    # tree is a tuple
    for tree_node, tree in tree_dict.items():
        tree_sec_name, tree_seg_loc = tree_node
        # if name of section to be inserted does not match the name of the current tree node section name, go deeper
        if tree_sec_name != sec_name:
            inserted_tree_dict[tree_node] = _morph_tree_insert_single(morph_tree = tree, sec_name = sec_name, loc = loc)
        else:
            # name of section to be inserted matches the name of the current tree node section
            # check if there is a child node that has a normalized location smaller or equal than the one currently to be inserted
            # if so, then go to that subtree
            subtree_dist, subtree = tree
            child_keys = subtree.keys()
            if child_keys and child_keys[0][0] == sec_name and child_keys[0][1]<=loc:
                inserted_tree_dict[tree_node] = _morph_tree_insert_single(morph_tree = tree, sec_name = sec_name, loc = loc)
            else:
                # there is no child section with same name that has a smaller segment location
                dist_from_current_node_start = (loc-tree_seg_loc)*get_secseg(nrn_h, sec_name).L
                dist_till_end_of_current_node = subtree_dist - dist_from_current_node_start

                inserted_tree_dict[tree_node] = (dist_from_current_node_start, {("{}({})".format(sec_name, loc), loc): 
                                                (dist_till_end_of_current_node, subtree)})
                
    return (dist, inserted_tree_dict)

def plot_phylogram(hoc_fpath, annotation_df = None, ax = None, figsize = (10,10), lw = 0.35, trunk_lw = 0.7,
    plotter = 'pyplot', legend_fontsize = 12, legend_ncol = 3, default_marker_size = 5,
    default_marker_color = (1,0,0,0.5), open_hoc_file = True, add_roi_target_labels = False,
    roi_target_label_xy_offset = (20, 0)):
    """
    Plots a dendritic tree phylogram.

    Parameters
    ----------
    hoc_fpath : str
        NEURON .hoc morphology file.
    annotation_df : None, pandas.DataFrame

        Annotation information as dataframe with columns:
            'section' : str
                Section name, e.g. 'soma' or 'dend[10]'.

            'location' : float
                Normalized segment location within the section [0,1].

            'marker_color' : tuple, str, default (1,0,0,0.5)
                Segment marker color.
                If tuple:
                    - RGB (r,g,b)
                    - RGB+alpha (r,g,b,a)
                If str:
                    - Named color.
                    - Hex color
                Or any other valid matplotlib color specifier.

            'annotation_color' : tuple, str, default (1,0,0,0.75)
                Segment annotation color. Same description as for 'marker_color'.

            'marker_size' : float, default 5
                Annotation marker size in points.

            'annotation_text' : str
                Annotation label.

            'annotation_text_size' : float, default 12
                Annotation font size in points.

            'legend_text' : str
                Legend text for this marker.

    ax : matplotlib.axes._subplots.AxesSubplot
        Matplotlib axis to use for plotting.

    figsize : tuple
        Figure width and height in inches.

    lw : float
        Line weight in points for non-trunk sections (or all sections if 'trunk' section list is not specified in NEURON).

    trunk_lw : float
        Line weight in points for trunk sections.

    plotter : str
        Choice of plotting engine. Choose between 'pyplot' and 'plotly'.

    legend_fontsize : float
        Legend font size in pt.

    legend_ncol : int
        Number of legend columns.

    open_hoc_file : bool
        If True, opens morphology again. Set to False if this is not needed and the same morphology can be used for different
        sequential plots.

    add_roi_target_labels : bool
        If True, adds ROI scan target labels next to the marked scan locations that have an associated legend label.

    roi_target_label_xy_offset : tuple
        X & Y axis offset in data coordinates between the marked location and ROI scan target label lower left corner.
    """
    lines, line_widths, arcs, arc_widths, adjusted_annotation_df = _get_circular_tree_data(hoc_fpath = hoc_fpath, annotation_df = annotation_df,
        plotter = plotter, default_marker_size = default_marker_size, default_marker_color = default_marker_color,
        open_hoc_file = open_hoc_file, trunk_linewidth = trunk_lw, nontrunk_linewidth = lw)

    if plotter == 'pyplot': 
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize = figsize)
        else:
            fig = None

        # radial lines
        lines_coord = [[(x[0][idx], x[1][idx]) for idx in range(len(x[0]))] for x in zip(lines.x,lines.y)]
        lc = mcol.LineCollection(lines_coord, color = 'k', linewidths = line_widths, zorder = 1,
            joinstyle = 'round', capstyle = 'round')
        ax.add_collection(lc)

        # arcs
        arcs_coord = [[(x[0][idx], x[1][idx]) for idx in range(len(x[0]))] for x in zip(arcs.x,arcs.y)]
        lc = mcol.LineCollection(arcs_coord, color = (0.7, 0.7, 0.7), linewidths = arc_widths, zorder = 0,
            joinstyle = 'round', capstyle = 'round')
        ax.add_collection(lc)

        unique_markings = set()
        if not adjusted_annotation_df.empty:
            for i, loc in adjusted_annotation_df.iterrows():
                # mark location
                if loc.legend_text and (loc.legend_text, loc.marker_color) in unique_markings:
                    ax.scatter(loc.x, loc.y, c = loc.marker_color, s = loc.marker_size, zorder = 2)
                else:
                    if loc.legend_text:
                        unique_markings.add((loc.legend_text, loc.marker_color)) 
                    ax.scatter(loc.x, loc.y, c = loc.marker_color, s = loc.marker_size, label = loc.legend_text, zorder = 2)
                # add roi target name next to the marked location
                if add_roi_target_labels:
                    if loc.legend_text:
                        ax.text(x = loc.x+roi_target_label_xy_offset[0], y = loc.y+roi_target_label_xy_offset[1], s = loc.annotation_text,
                            color = loc.marker_color, zorder = 2)
        
        # autoscale and equalize aspect ratio
        ax.autoscale()
        ax.set_aspect('equal')
        # remove spines and ticks
        ax.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
        ax.tick_params(axis = 'y', which = 'both', right = False, left = False, labelleft = False)
        for pos in ['right','top','bottom','left']:
            ax.spines[pos].set_visible(False)

        # add scalebar
        scalebar = AnchoredHScaleBar(size = 100, label = "100 $\mu$m", loc = 4, frameon = False, 
            pad = 0.6, sep = 4, linekw = dict(color = "k", linewidth = 1.5), extent = 0.012, label_fontsize = 7)
        ax.add_artist(scalebar)
        
        # add legend if there are any legend labels
        default_marker_color = (1,0,0,0.5)
        if annotation_df is not None and 'legend_text' in annotation_df and not annotation_df['legend_text'].isnull().all():
            # sort by legend labels
            handles, labels = sort_legend_labels(ax)
            if handles and labels:
                ax.legend(handles, labels, loc = 'lower left', bbox_to_anchor= (0.0, 1.01), fontsize = legend_fontsize, ncol = legend_ncol)

        return fig, ax

    if plotter == 'plotly':

        trace_radial_lines = dict(
            type = 'scatter',
            x = lines.x,
            y = lines.y,
            mode = 'lines',
            line = dict(color = 'rgb(0,0,0)', width = 1),
            hoverinfo = 'none'
        )

        trace_arcs = dict(
            type = 'scatter',
            x = arcs.x,
            y = arcs.y,
            mode = 'lines',
            line = dict(color='rgb(180,180,180)', width = 1, shape = 'spline'),
            hoverinfo = 'none'
        )

        if not adjusted_annotation_df.empty:
            trace_nodes = dict(
                type = 'scatter',
                x = adjusted_annotation_df.x,
                y = adjusted_annotation_df.y, 
                mode = 'markers',
                marker = dict(
                    color = adjusted_annotation_df.marker_color,
                    size = adjusted_annotation_df.marker_size,
                ),
                text = adjusted_annotation_df.segment, 
                hoverinfo = "text"
            )
        else:
            trace_nodes = {}

        layout = dict(
            font = dict(family = 'Balto', size = 14),
            width = 700,
            height = 750,
            autosize = False,
            showlegend = False,
            xaxis = dict(visible = False),
            yaxis = dict(visible = False, scaleanchor = "x", scaleratio = 1), 
            hovermode = 'closest',
            plot_bgcolor = 'rgb(245,245,245)',
            margin = dict(t = 75)
        )

        fig = go.FigureWidget(data = [trace_radial_lines, trace_arcs, trace_nodes], layout = layout)
        fig.update_layout(
            {
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            }
        )

        init_notebook_mode(connected = True)
        iplot(fig)

        return fig

    else:
        raise ValueError("plotter parameter must be 'pyplot' or 'plotly'.")