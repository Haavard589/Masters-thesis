# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:58:17 2024

@author: hfyhn
"""

# -*- coding: utf-8 -*-
import os, subprocess

import file_support
import template_matching
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from skimage import measure
from tqdm import tqdm
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import maximum_filter
from scipy.fft import fft, fftfreq, ifft
from orix.quaternion import Orientation, symmetry
from pathlib import Path
import sys

TM = None

def test_lib_param():
    global TM

    elements = ["muscovite"]
    TM = template_matching.Template_matching(elements = elements)
    
    TM.create_lib(16.0, deny_new = False, minimum_intensity=1E-20, max_excitation_error=78E-4, force_new = True, camera_length=120, half_radius = 128, reciprocal_radius=1.40752296788822, accelerating_voltage = 200, diffraction_calibration = 0.010996273186626718,precession_angle=0.0)
    TM.scatterplot([0,0,90], "")
 

def Rotation_matrix(t, deg = False):
    if deg:
        t = t/180*np.pi
    return np.array([[np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)]]) 

def create_circular_mask(h, w, center=None, radius=None, mirror = False):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]

    dist_from_center = np.sqrt((X - (256 - center[1 if mirror else 0]))**2 + (Y - (256 - center[0 if mirror else 1]))**2)

    mask = dist_from_center <= radius
    return mask


def res_to_xmap(folder, plot = False):
    global TM
    

    res = np.load(folder + r"\res.npy", allow_pickle=True)    
    phasedict = np.load(folder + r"\phasedict.npy", allow_pickle=True)    
    elements = ["muscovite", "quartz"]
    signal = file_support.read_file(folder + r"\signal.hspy", lazy=True)
    TM = template_matching.Template_matching(signal = signal, elements = elements)
    TM.create_lib(1.0, deny_new = False, minimum_intensity=1E-20, max_excitation_error=78E-4, force_new = False)
    TM.result = res.all()
    TM.phasedict = phasedict.all()
    TM.phasedict[len(TM.elements)] = 'vacuum'
    
    TM.result['phase_index'] = np.where(TM.result['correlation'] < 0.001, len(TM.elements), TM.result['phase_index'])
    map = np.load(folder + r"\E3\map_00.npy")
    print(TM.result['phase_index'][:,:,0].shape)
    TM.result['phase_index'][:,:,0] = np.where(map == 0, len(TM.elements), TM.result['phase_index'][:,:,0])

    TM.init_xmap()
    if plot:
        TM.plot_correlation()
        
        TM.plot_orientations_mapping()
        
        
        TM.plot_phase_map()

def save_as_svg(figure, **kwargs):
    inkscape_path = kwargs.get('inkscape', "C://Program Files//Inkscape//bin//inkscape.exe")
    filepath = kwargs.get('filename', None)

    if filepath is not None:
        path, filename = os.path.split(filepath)
        filename, extension = os.path.splitext(filename)

        svg_filepath = os.path.join(path, filename+'.svg')
        figure.savefig(svg_filepath, format='svg')

    
    
def plot_ipf_radius(euler_angle, radius): 
    N = 300
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection="ipf", symmetry=symmetry.D3d)

    orientations = Orientation.from_euler([euler_angle], symmetry=symmetry.D3d, degrees=True)
    ax.scatter(orientations)
    d = radius * np.pi/180
  

    
    euler_angle1 = euler_angle[1]*np.pi/180
    euler_angle2 = euler_angle[2]*np.pi/180
    
    if euler_angle[1] == 0:
        e1 = np.linspace(-np.pi, np.pi, N)
        e0 = np.ones(N) * d

    else:
        e0 = np.linspace(-d+euler_angle1, d+euler_angle1, N)
        e1 = euler_angle2 + np.arccos((np.cos(d)-np.cos(euler_angle1)*np.cos(e0))/(np.sin(euler_angle1)*np.sin(e0)))


    orientations = Orientation.from_euler(np.column_stack((np.zeros(len(e0)), e0, e1)), symmetry=symmetry.D3d, degrees=False)
    ax.scatter(orientations, s = 1, color = "black")

    orientations = Orientation.from_euler(np.column_stack((np.zeros(len(e0)), e0, 2*euler_angle2-e1)), symmetry=symmetry.D3d, degrees=False)
    ax.scatter(orientations, s = 1, color = "black")  
   
    save_as_svg(fig,filename = ...)


    
def compare_cnn_peak_finding(x , y):
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]

    signal_c = file_support.read_file(..., lazy=True)
    signal_pre = file_support.read_file(..., lazy=True)
    signal_pro = file_support.read_file(..., lazy=True)
    

    signals = [signal_c, signal_pro, signal_pre]
    for i in range(len(signals)):
        signals[i] = signals[i].inav[x,y].data
    signals[-1] *= signals[0]
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(bottom=0.4)
    plt.rcParams['font.size'] = 12

    ax = [fig.add_subplot(1,len(signals)+1, i+1) for i in range(len(signals))]
    
    for i,a in enumerate(ax):
        a.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 

        a.set_title("(" + alphabet[i] + ")", y = -0.2)
        

    for i,pred in enumerate(signals):
        ax[i].imshow(signals[i], cmap = "Greys_r", norm = "symlog")
    
def misorientation_plot(x,y):
    o = TM.result["orientation"][y,x]
    o = Orientation.from_euler(o, degrees  = True)
    others = Orientation.from_euler(TM.result["orientation"][:,128:], degrees  = True)
    mis = o.angle_with(others, degrees=True)
    fig, ax = plt.subplots(figsize=(4,3),ncols = 1, nrows = 1)
    pos = ax.imshow(mis, cmap = "Reds")
    fig.colorbar(pos, ax=ax)
    ax.axis("off")
    
    

def analyse_pseudo():
    global TM
    res_to_xmap(...)
    elements = ["quartz"]

    template = np.argmin(np.array([np.linalg.norm(x - [0,80.2913055, -44.0]) for x in TM.library[elements[0]]['orientations']]))
    print( TM.library[elements[0]]['orientations'][template])
    signal = TM.library[elements[0]]['pixel_coords'][template]
    
    angle = 167.4+85
            
    R = Rotation_matrix(-angle, True)
    
    signal = np.array([np.matmul(R, x - [128, 128]) + [128, 128] for x in signal])

    
    image = np.zeros((256,256))

    for point in signal:
        image += create_circular_mask(256, 256, center=point, radius=5, mirror = True)

    template2 = np.argmin(np.array([np.linalg.norm(x - [0,80.2913055, -76.0]) for x in TM.library[elements[0]]['orientations']]))
    print( TM.library[elements[0]]['orientations'][template2])
    signal2 = TM.library[elements[0]]['pixel_coords'][template2]
    
    angle2 = 167.4+120
            
    signal2[:,0] = signal2[:,0]
    R = Rotation_matrix(-angle2, True)
    
    signal2 = np.array([np.matmul(R, x - [128, 128]) + [128, 128] for x in signal2])
    
    
    image2 = np.zeros((256,256))

    for point in signal2:
        image2 += create_circular_mask(256, 256, center=point, radius=5, mirror = True)

    fig = plt.figure(figsize=(10, 10), constrained_layout = True)

    ax1 = fig.add_subplot(122)

    ax1.set_title('(b)', y = -0.2)
    
    ax1.imshow(image, cmap="Greys_r", norm ="symlog")
    plt.axis("off")

    ax2 = fig.add_subplot(121)

    ax2.set_title('(a)', y = -0.2)
    
    ax2.imshow(image2, cmap="Greys_r", norm ="symlog")

    plt.axis("off")


def analyse_sim():
    global TM

    elements = ["quartz"]
    res_to_xmap(...)
    template2 = np.argmin(np.array([np.linalg.norm(x - [0,80.2913055, -76.0]) for x in TM.library[elements[0]]['orientations']]))
    print( TM.library[elements[0]]['orientations'][template2])
    for template in tqdm([template2]):
        signal = TM.library[elements[0]]['pixel_coords'][template]
        angle = 167.4+120
                
        R = Rotation_matrix(-angle, True)
        signal2 = np.array([np.matmul(R, x - [128, 128]) + [128, 128] for x in signal])

        image = np.zeros((256,256))
        image2 = np.zeros((256,256))

        for point in signal:
            image[point[0], point[1]] = 1
        for point in signal2:
            image2 += create_circular_mask(256, 256, center=point, radius=5, mirror = True)


        TM.entire_dataset_single_signal(element = elements[0], image = image, plot = True, intensity_transform_function = lambda x: x**0.4, nbest =1, template = template, max_r = 128) 
        
        best = np.argmax(TM.correlations)
        coords = TM.library[elements[0]]['pixel_coords'][best]
        

        
        angle = TM.angles[best] + 287.4
        
        
        coords[:,0] = 256 - coords[:,0]
        R = Rotation_matrix(angle + 180, True)
        
        coords = np.array([np.matmul(R, x - [128, 128]) + [128, 128] for x in coords])
        TM.IPF_maping(0, [TM.library[elements[0]]['orientations'][template], 
                          TM.library[elements[0]]['orientations'][np.argmax(TM.correlations)] + np.array([angle,0,0])],
                      image = image2, scatter = coords)


def Template_match(folder, file, orientation):
    global TM
    signal = file_support.read_file(folder + file, lazy=False)
   
    elements = ["muscovite", "quartz"]
    TM = template_matching.Template_matching(signal = signal, elements = elements)
    TM.create_lib(1.0, deny_new = False, minimum_intensity=1E-20, max_excitation_error=78E-4, force_new = False)
    

    TM.template_match(intensity_transform_function=lambda x:x**0.4)
    np.save(folder + r"\res_"+orientation+".npy",TM.result)

    TM.plot_correlation()
    
    TM.plot_orientations_mapping()
    
    TM.signal.plot()
    
    TM.plot_phase_map()
    del TM
    
def compare_phase():
    res_to_xmap(...)
    TM.phasedict[len(TM.elements)] = 'vacuum'
    TM.result['phase_index'] = np.where(TM.result['correlation'] == 0.001, len(TM.elements), TM.result['phase_index'])
    phasemap_cnn = TM.result["phase_index"] + 1
    phasemap_cnn = np.where(phasemap_cnn == 3, 0 , phasemap_cnn)
    res_to_xmap(...)
    phasemap = TM.result["phase_index"] + 1
    phasemap = np.where(phasemap == 3, 0 , phasemap)
    
    plt.figure()
    plt.imshow(phasemap_cnn )
    
    
    plt.figure()
    phase_diff = np.where(phasemap_cnn == 1, 2, 0) - np.where(phasemap == 1, 2, 0)
    phase_diff = np.where(phasemap_cnn == 0, 0, phase_diff)
    phase_diff = np.where(phasemap == 0, 0, phase_diff)

    plt.imshow(phase_diff[:,:128], cmap="coolwarm_r", interpolation='nearest')
    print("False negative PF",np.sum(np.where(phase_diff[:,:128] == -2,1,0)))
    print("False negative PF",np.sum(np.where(phase_diff[:,:128] == -1,1,0)))

    print("Equal phase",np.sum(np.where(phase_diff[:,:128] == 0,1,0)))
    print("False negative PF",np.sum(np.where(phase_diff[:,:128] == 1,1,0)))

    print("False negative CNN",np.sum(np.where(phase_diff[:,:128] == 2,1,0)))
    plt.axis("off")

    plt.figure()

    plt.axis("off")
    
    plt.figure()
    phase_diff = np.where(phasemap_cnn == 2, 2, 0) - np.where(phasemap == 2, 2, 0)
    phase_diff = np.where(phasemap_cnn == 0, 0, phase_diff)
    phase_diff = np.where(phasemap == 0, 0, phase_diff)

    
    plt.figure()
    
    plt.imshow(phase_diff[:,128:],  cmap="coolwarm_r", interpolation='nearest')
    print("False negative PF",np.sum(np.where(phase_diff[:,128:] == -2,1,0)))
    print("Qual phase",np.sum(np.where(phase_diff[:,128:] == 0,1,0)))
    print("False negative CNN",np.sum(np.where(phase_diff[:,128:] == 2,1,0)))

    plt.axis("off")
    
def fix_pseudo(res):
    shape = res["mirrored_template"].shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k, e in enumerate(res["mirrored_template"][i,j]):

                if e:
                    res["template_index"][i,j][0] = res["template_index"][i,j][k]
                    res["mirrored_template"][i,j][0] = res["mirrored_template"][i,j][k]
                    TM.result['orientation'][i,j][0] = TM.result['orientation'][i,j][k]
                    TM.result['correlation'][i,j][0] = TM.result['correlation'][i,j][k]
                    break

    return res