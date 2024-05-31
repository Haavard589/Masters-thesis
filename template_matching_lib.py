# -*- coding: utf-8 -*-
from pyxem.utils import indexation_utils as iutls
from pyxem.utils import plotting_utils as putls
import pyxem as pxm #Electron diffraction tools based on hyperspy


from orix.quaternion import Orientation, symmetry
from orix.vector.vector3d import Vector3d
from orix.projections import StereographicProjection
from orix import plot
from orix.crystal_map.crystal_map import CrystalMap


import numpy as np 

from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors #Some plotting color tools
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.font_manager as fm
from matplotlib_scalebar.dimension import _Dimension

from Utils.GetDiffLibrary import GetDiffLibrary, GetDiffLibrary_by_filepath

import hyperspy.api as hs #General hyperspy package
from spyder.api.widgets.mixins import SpyderWidgetMixin



class Template_matching:
    def __init__(self, signal = None, elements = None):
        self.signal = signal
        self.image = None
        self.orientations = None
        self.library = None
        self.template_indices = None
        self.angles = None
        self.correlations = None
        self.mirrored_angles = None 
        self.mirrored_correlations = None
        
        self.symmetries = None
        self.space_groups = None
        self.strucures = None
        
        self.orientations = None
        
        self.result = None  
        self.phasedict = None
        
        self.xmap = None 
        
        self.cif_files = None
        self.elements = elements
        self.set_symmetry()
        
       
        
    def set_symmetry(self, elements = None, cif_file = None):  
        if elements is not None:
            self.elements = elements
        print(self.elements)
        self.symmetries   = [""]*len(self.elements)
        self.space_groups = [""]*len(self.elements)
        self.strucures    = [""]*len(self.elements)
        
        for i, element in enumerate(self.elements):
            if element == "muscovite" or element == "muscovite2":
                self.symmetries[i] = symmetry.C2h # C2h is monoclinic
                self.space_groups[i] = 15
                self.strucures[i] = "monoclinic"
    
            if element == "muscoviteT":
                self.symmetries[i] = symmetry.D3
                self.space_groups[i] = 151
                self.strucures[i] = "trigonal"
    
            if element == "illite":
                self.symmetries[i] = symmetry.C2h
                self.space_groups[i] = 12
                self.strucures[i] = "monoclinic"
                
            if element == "aluminium":
                self.symmetries[i] = symmetry.Oh
                self.space_groups[i] = 225
                self.strucures[i] = "cubic"
                
            if element == "LNMO":
                self.symmetries[i] = symmetry.C1
                self.space_groups[i] = 227
                self.strucures[i] = "cubic"
                
            if element == "CeAlO3":
                self.symmetries[i] = symmetry.D4h
                self.space_groups[i] = 123
                self.strucures[i] = "cubic"    
                
            if element == "quartz":
                self.symmetries[i] = symmetry.D3d
                self.space_groups[i] = 162
                self.strucures[i] = "trigonal"    
          
        
        
        if cif_file is None:
            self.cif_files = [r"C:\Users\hfyhn\Documents\Skole\Fordypningsoppgave\Data\cif" + "\\" + element + ".cif" for element in self.elements]
        else:
            self.cif_files = cif_file
    
    def correct_shifts_COM(self, com_mask, nav_mask=None, plot_results=False, inplace=False):
        com = self.signal.center_of_mass(mask=com_mask)
        if plot_results:
            com.get_bivariate_histogram().plot()
            
        beam_shift = pxm.signals.BeamShift(com.T)
        beam_shift.make_linear_plane(mask=nav_mask)
        print(beam_shift)
        print(beam_shift.isig[0] - self.signal.axes_manager.signal_shape[0]/2.0)
        print(beam_shift.isig[1], self.signal.axes_manager.signal_shape[1]/2.0)
    
        x_shift, y_shift = [beam_shift.isig[ax] - self.signal.axes_manager.signal_shape[ax]/2.0 for ax in (0, 1)]
        
        print(f'Estimated beam shift X min/max = ({x_shift.min().data}, {x_shift.max().data})\nEstimated beam shift Y min/max = ({y_shift.min().data}, {y_shift.max().data})')
        
        dp_max_before = self.signal.max(axis=[0, 1])
        
        #A trick to make sure that the shifted signal contains the same metadata etc as the original signal. Might not be needed...
        if not inplace:
            shifted_signal = self.signal.deepcopy()
        else:
            shifted_signal = self.signal
        
        shifted_signal.shift_diffraction(x_shift, y_shift, inplace=True)
        
        dp_max_after = shifted_signal.max(axis=[0, 1])
            
        if plot_results:
            hs.plot.plot_images([dp_max_before, dp_max_after], overlay=True, colors=['w', 'r'], axes_decor='off', alphas=[1, 0.75])
        
        return shifted_signal, x_shift, y_shift
        
    
    
    def init_lib(self, filepath):
        self.library = GetDiffLibrary_by_filepath(filepath)

    def create_lib(self, resolution = 2.0, minimum_intensity=1E-20, max_excitation_error=1.7E-2, force_new = False, deny_new = False, half_radius = None, diffraction_calibration = None, camera_length = None, precession_angle = None,reciprocal_radius = None, accelerating_voltage = None):

        if self.signal is not None:
            diffraction_calibration = self.signal.axes_manager[-1].scale
            camera_length = self.signal.metadata.Acquisition_instrument.TEM.Detector.Diffraction.camera_length
            precession_angle = self.signal.metadata.Acquisition_instrument.TEM.rocking_angle
            accelerating_voltage = self.signal.metadata.Acquisition_instrument.TEM.beam_energy
            half_radius = np.min(self.signal.axes_manager.signal_shape)//2
            reciprocal_radius = np.max(np.abs(self.signal.axes_manager[-1].axis))
        print(diffraction_calibration, camera_length,
        half_radius,
        reciprocal_radius,
        resolution,
        minimum_intensity,
        max_excitation_error,
        precession_angle,
        accelerating_voltage)

        self.library = GetDiffLibrary(diffraction_calibration = diffraction_calibration, 
                                camera_length = camera_length,
                                half_radius = half_radius,
                                reciprocal_radius = reciprocal_radius,
                                resolution = resolution,
                                make_new=force_new,
                                grid = None, 
                                minimum_intensity=minimum_intensity,
                                max_excitation_error=max_excitation_error,
                                precession_angle = precession_angle,
                                cif_files = self.cif_files,
                                elements = self.elements,
                                structures = self.strucures, 
                                accelerating_voltage = accelerating_voltage,
                                deny_new_create = deny_new
                                )
        
    
    def test_lib_param(self, minimum_intensitys=[1E-20], max_excitation_errors=[1.7E-2]):
        diffraction_calibration = self.signal.axes_manager[-1].scale
        half_radius = np.min(self.signal.axes_manager.signal_shape)//2
        reciprocal_radius = np.max(np.abs(self.signal.axes_manager[-1].axis))
        print(half_radius, diffraction_calibration, reciprocal_radius)
        
        camera_length = self.signal.metadata.Acquisition_instrument.TEM.Detector.Diffraction.camera_length
        precession_angle = self.signal.metadata.Acquisition_instrument.TEM.rocking_angle
        

        for minimum_intensity in minimum_intensitys:
            for max_excitation_error in max_excitation_errors:
                self.library = GetDiffLibrary(diffraction_calibration = diffraction_calibration, 
                                        camera_length = camera_length,
                                        half_radius = half_radius,
                                        reciprocal_radius = reciprocal_radius,
                                        resolution = 16,
                                        make_new = True,
                                        grid = None, 
                                        minimum_intensity=minimum_intensity,
                                        max_excitation_error=max_excitation_error,
                                        #precession_angle = precession_angle,
                                        cif_files = self.cif_files,
                                        elements = self.elements,
                                        structures = self.strucures, 
                                        accelerating_voltage = self.signal.metadata.Acquisition_instrument.TEM.beam_energy
                                        )
                template = np.argmin(np.array([np.linalg.norm(x - np.array([0,0,90])) for x in self.library[self.elements[0]]['orientations']]))

                self.scatterplot(template, str(minimum_intensity) + " | " + str(max_excitation_error))

    def plot_random_diff_lib(self,x = 0, y = 0, i = 0, element = "muscovite"):
        
        image = self.signal.inav[x, y].data

            
        simulation_test = self.library[element]['simulations'][i]
        # for completeness in the illustration, all keyword arguments are given and explained
        # an array of angles and corresponding correlation values are returned
        angle, correlation = iutls.get_in_plane_rotation_correlation(
            image,
            simulation_test,
            intensity_transform_function=None,  # a function applied both to the image and template intensities before calculating the correlation
            delta_r = 1,                        # sampling in the radial direction
            delta_theta = 0.1,                  # sampling in the azimuthal direction
            max_r = None,                       # maximum radius to consider, by default the distance from the center to the corner
            find_direct_beam = False,            # convenience, if the pattern was not centered, this will perform a rough centering
            direct_beam_position = False,        # manually provide the coordinates of the direct beam
            normalize_image=True,               # divide the correlation by the norm of the image
            normalize_template=True,            # divide the correlation by the norm of the template
        )
        
        fig, ax = plt.subplots()
        ax.plot(angle, correlation)
        ax.set_xlim(0, 360)
        ax.set_xlabel("Angular shift (Degrees)")
        ax.set_ylabel("Correlation")
                
        putls.plot_template_over_pattern(image,
                                 simulation_test,
                                 in_plane_angle=angle[np.argmax(correlation)],
                                 coordinate_system = "cartesian", 
                                 size_factor = 10,
                                 max_r = 200,
                                 find_direct_beam=True,
                                 cmap = "Greens_r"
                                )
        
    def scatterplot(self, euler_angles,title = ""):
        plt.figure()
            
        for element in self.elements:
            i = np.argmin(np.linalg.norm(self.library[element]["orientations"] - euler_angles, axis = 1))
            coords = self.library[element]["pixel_coords"][i]
            plt.scatter(coords[:,0], coords[:,1], marker = "x", label = element)
        plt.legend(loc='upper right')
        plt.axis('equal')
        plt.title("Euler angle " + str(self.library[element]["orientations"][i]) + " | " + title)

        
    def entire_dataset_single_signal(self, x = -1, y = -1, element = "muscovite", plot = True, image = None, nbest = 5, intensity_transform_function = lambda x:x**0.15, template = -1, euler_angle = -1, max_r = None):
        frac_keep = 1  # if frac_keep < 1 or 1 < n_keep < number of templates then indexation will be performed on the
        n_keep = None
        if image is None: 
            s = self.signal.inav[x,y]
       
            try:
                s.compute()
            except Exception:
                pass
            self.image = s.data
        else:
            self.image = image

        self.orientations = None
        self.template_indices = None
        self.angles = None
        self.correlations = None
        self.mirrored_angles = None
        self.mirrored_correlations = None
        
       
        self.template_indices, self.angles, self.correlations, self.mirrored_angles\
            , self.mirrored_correlations = iutls.correlate_library_to_pattern(
            self.image, 
            self.library[element]['simulations'], 
            frac_keep=frac_keep, 
            n_keep=n_keep, 

            delta_r = 1,
            delta_theta = 0.1,
            max_r = max_r if max_r is not None else np.min(self.signal.axes_manager.signal_shape)//2,
            intensity_transform_function = intensity_transform_function,
            normalize_image = False,
            normalize_templates = True
        )
            
            
    
        self.orientations = Orientation.from_euler(self.library[element]['orientations'][self.template_indices], symmetry=self.symmetries[0], degrees=True)
        argmax = np.argmax(self.correlations)
        if not plot:
            return argmax, self.angles[argmax]
        
        self.plot_simulation(element = element, nbest = nbest, template = template, euler_angle = euler_angle)
        return argmax, self.angles[argmax]
    def IPF_map(self):
        for i, element in enumerate(self.elements):
            fig = plt.figure(figsize=(8, 8))
            
            max_correlation = np.max(np.stack((self.correlations[i], self.mirrored_correlations[i])))
            max_angle = np.max(np.stack((self.angles, self.mirrored_angles)))
                    
            
            ax0 = fig.add_subplot(221, projection="ipf", symmetry=self.symmetries[i])
            ax0.scatter(self.orientations[i], c=self.correlations[i]/max_correlation, cmap='inferno')
            ax0.set_title('Correlation for element '+ element)
            
            ax1 = fig.add_subplot(222, projection="ipf", symmetry=self.symmetries[i])
            ax1.scatter(self.orientations[i], c=self.mirrored_correlations[i]/max_correlation, cmap='inferno')
            ax1.set_title('Correlation (m) for element '+ element)
            
            ax2 = fig.add_subplot(223, projection="ipf", symmetry=self.symmetries[i])
            ax2.scatter(self.orientations[i], c=self.angles[i]/max_angle, cmap='inferno')
            ax2.set_title('Angle for element '+ element)
            
            ax3 = fig.add_subplot(224, projection="ipf", symmetry=self.symmetries[i])
            ax3.scatter(self.orientations[i], c=self.mirrored_angles[i]/max_angle, cmap='inferno')
            ax3.set_title('Angle (m) for element '+ element)
            
            plt.colorbar(ScalarMappable(norm=mcolors.Normalize(0, max_angle), cmap='inferno'), ax=ax2)
            plt.colorbar(ScalarMappable(norm=mcolors.Normalize(0, max_angle), cmap='inferno'), ax=ax3)
        
    def IPF_maping(self, nbest, orientation = None, image = None, scatter = None, symmetry_index = 0):

        correlations_list = np.argsort(self.correlations)[-nbest:]
        mirrored_correlations_list = np.argsort(self.mirrored_correlations)[-nbest:]
        orientations = self.orientations[correlations_list]
        orientations_m = self.orientations[mirrored_correlations_list]

        correlations = self.correlations[correlations_list] - np.min(self.correlations[correlations_list])

        mirrored_correlations = self.mirrored_correlations[mirrored_correlations_list]

        fig = plt.figure(figsize=(10, 10), constrained_layout = True)

        
        plt.rcParams['font.size'] = 26
        max_correlation = np.max(np.stack((correlations, mirrored_correlations)))
        
        if orientation is None: 
            ax0 = fig.add_subplot(111, projection="ipf", symmetry=self.symmetries[symmetry_index])
            
        else:
            if image is None:
                    
                ax0 = fig.add_subplot(121, projection="ipf", symmetry=self.symmetries[symmetry_index])
                ax1 = fig.add_subplot(122, projection="ipf", symmetry=self.symmetries[symmetry_index])
                ax1.set_title('Correlation')
                ax1.scatter(Orientation.from_euler([orientation], symmetry=self.symmetries[symmetry_index], degrees=True), c=np.linspace(0,1,num=len(orientation)), cmap='coolwarm')
    
                ax1.set_title('Correlation')
            else:
               ax0 = fig.add_subplot(221, projection="ipf", symmetry=self.symmetries[symmetry_index])
               ax1 = fig.add_subplot(222, projection="ipf", symmetry=self.symmetries[symmetry_index])
               o = Orientation.from_euler(orientation, symmetry=self.symmetries[symmetry_index], degrees=True)
               #o[0].
               #ax1.set_title("Misorientation " + str(o[0].angle_with(o[1], degrees=True)[0]))
               ax1.set_title("(b)", y = -0.8)

               ax1.scatter(o[0], c = "blue",s = 100, label = "Background")
               ax1.scatter(o[1], c = "red", s = 100, label = "Best fit")
               
               handles, labels = ax1.get_legend_handles_labels()
               ax1.legend(handles[-2:], labels[-2:], loc='upper right',bbox_to_anchor=(1, -0.15), ncol=1)
               #ax1.legend(["Background", "Template match"], loc = "upper right")
               
               ax2 = fig.add_subplot(212)
               ax2.set_position([0.392,0,0.33,0.33])
               #ax2.set_title('Euler angle \n Background' + str(orientation[0]) + "\n Template match" + str(orientation[1]))

               ax2.set_title('(c)', y = -0.2)
               
               ax2.imshow(image, cmap="Greys_r", norm ="symlog")
               ax2.scatter(scatter[:,0], scatter[:,1], marker = "x", color = "red")
               ax2.tick_params(left = False, right = False , labelleft = False , 
                       labelbottom = False, bottom = False) 
               ax3 = fig.add_subplot(224)
               ax3.set_title('(d)', y = -0.2)
               ax3.tick_params(left = False, right = False , labelleft = False , 
                       labelbottom = False, bottom = False) 
               ax3.imshow(image, cmap="Greys_r", norm ="symlog")

               ax3.spines['top'].set_visible(False)
               ax3.spines['right'].set_visible(False)
               ax3.spines['bottom'].set_visible(False)
               ax3.spines['left'].set_visible(False)
               oris = Orientation.stack(Orientation.from_euler(orientation, degrees=True)).squeeze()
               oris.symmetry = orientations[0].symmetry    
               oris.scatter(ec="k", s=100, c=[[0,0,1],[1,0,0]], figure = fig, position = (2,2,4))
               
               
        ax0.set_title('(a)', y = -0.8)
        ax0.scatter(orientations, c=correlations/max_correlation, cmap='inferno')

    def IPF_mapping_given_orientation(self, orientations, correlations, symmetry = 0,invert = False):
    
        fig = plt.figure(figsize=(8, 8))

        max_correlation = np.max(correlations)
    

        ax0 = fig.add_subplot(111, projection="ipf", symmetry=self.symmetries[symmetry])

        orientations = Orientation.from_euler(orientations, symmetry=self.symmetries[symmetry], degrees=True)
        ax0.scatter(orientations, c=invert - (correlations/max_correlation), cmap='coolwarm')
        

   
    def order_best(self, nbest = 5, find_mirror = True):
        best_templates = [0]*nbest
        best_angles = [0]*nbest
        best_correlation = [0]*nbest
        mirrored = [0]*nbest
        correlation_list = np.argsort(self.correlations)[-nbest:]
        mirrored_correlations_list = np.argsort(self.mirrored_correlations)[-nbest:]
        counter = 1
        m_counter = 1
        for i in range(nbest):
            
            ci = correlation_list[-counter]
            
            mci = mirrored_correlations_list[-m_counter]
            #print(ci,mci, counter, m_counter)
            if not find_mirror or self.correlations[ci] > self.mirrored_correlations[mci]:
                best_angles[i] = self.angles[ci]
                best_templates[i] = self.template_indices[ci]
                best_correlation[i] = self.correlations[ci]
                mirrored[i] = False
                counter += 1 
            
            else:
                best_angles[i] = self.mirrored_angles[mci]
                best_templates[i] = self.template_indices[mci]
                best_correlation[i] = self.mirrored_correlations[mci]
                mirrored[i] = True
                m_counter += 1 
                
        return best_templates, best_angles, best_correlation, mirrored
        
                   
    def plot_simulation(self, nbest = 1, element = "muscovite", template = -1, euler_angle = -1):
        plt.axis('off')
        best_templates, best_angles, best_correlation, mirrored = self.order_best(nbest, False)
    
        if template == -1 and euler_angle != -1:
            template = np.argmin(np.array([np.linalg.norm(x - euler_angle) for x in self.library[element]['orientations']]))

        if template != -1:
            putls.plot_template_over_pattern(self.image,
                                             self.library[element]['simulations'][template],
                                             in_plane_angle=self.angles[template],
                                             coordinate_system = "cartesian", 
                                             size_factor = 1,
                                             max_r = 200,
                                             mirrored_template=False,
                                             find_direct_beam=False,
                                             marker_color='r',
                                             cmap = "Blues",
                                             norm='symlog'
                                            )
            print("Actual pattern:", self.correlations[template])
        for i, template in enumerate(best_templates):
            plt.axis('off')
            print("\t\n",str(self.library[element]['orientations'][template] + np.array([best_angles[i],0,0])), mirrored[i],best_correlation[i])
            
            putls.plot_template_over_pattern(self.image,
                                             self.library[element]['simulations'][template],
                                             in_plane_angle=best_angles[i],
                                             coordinate_system = "cartesian", 
                                             size_factor = 1,
                                             max_r = 200,
                                             mirrored_template=False,
                                             find_direct_beam=False,
                                             marker_color='r',
                                             cmap = "viridis",
                                             norm='symlog'
                                            )
            plt.axis('off')
            
    def template_match(self, intensity_transform_function = lambda x:x**0.4):
        # let's not throw away any templates prematurely and perform a full calculation on all
        frac_keep = 1
        #self.signal.data = np.where(self.signal.data == 0.0,  -50.0, self.signal.data)

        self.result, self.phasedict = iutls.index_dataset_with_template_rotation(self.signal,
                                                                        self.library,
                                                                        phases = self.elements,
                                                                        n_best = 20,
                                                                        frac_keep = frac_keep,
                                                                        n_keep = None,
                                                                        delta_r = 1,
                                                                        delta_theta = 1,
                                                                        max_r = np.min(self.signal.axes_manager.signal_shape)//2,
                                                                        intensity_transform_function = intensity_transform_function,
                                                                        normalize_images = True,
                                                                        normalize_templates = True,
                                                                        )
        
        self.phasedict[len(self.elements)] = 'vacuum'
        self.result['phase_index'][np.isnan(self.result['correlation'])] = len(self.elements)
        self.phasedict[len(self.elements)] = 'vacuum'
        self.result['phase_index'] = np.where(self.result['correlation']==0, len(self.elements), self.result['phase_index'])
        
        
        print("Mean correlation", np.nanmean(self.result["correlation"]))
        
    def plot_template_matching(self, element = "muscovite"):
        if self.result is None:
            print("You must run templete match before you can plot!")
            return 
        px, py = [self.signal.axes_manager[ax].index for ax in (0, 1)] #Get the x-y coordinates to check
        n_sol = 0 # Select which solution to plot, should be an integer between 0 and `n_best-1`
        
        solution = self.result["orientation"] #Get the orientations of the  result
        correlations = np.nan_to_num(self.result["correlation"][:, :, n_sol].ravel()) #Get the correlation for
        
        #Plot IPF maps of the results
        orientations = Orientation.from_euler(solution, symmetry=self.symmetry, degrees=True)
        sols = orientations.shape[-1]
        fig = plt.figure(figsize=(sols*1.2, 2*1.2))
        max_correlation = np.max(np.nan_to_num(self.result["correlation"]))
        for n in range(sols):
            ax = fig.add_subplot(1, sols, n+1, projection='ipf', symmetry=self.symmetry)
            ax.scatter(orientations[:, :, n], c=correlations/max_correlation, cmap='inferno')
        
            
        # Get the results from the selected scan pixel and solution
        sim_sol_index = self.result["template_index"][py, px, n_sol]
        mirrored_sol = self.result["mirrored_template"][py, px, n_sol]
        in_plane_angle = self.result["orientation"][py, px, n_sol, 0] #! NOTE: the first angle should be the in plane angle! But the template in the resulting figure does not look correct - must check!
        # Get the template for the selected solution and pixel
        sim_sol = self.library[element]['simulations'][sim_sol_index]
        
        #Plot an IPF map and the diffraction pattern with template overlay in a single figure
        fig = plt.figure(figsize=(8, 4))
        
        ax0 = fig.add_subplot(121, projection="ipf", symmetry=self.symmetry)
        ax0.scatter(orientations[:, :, n_sol], c=correlations/np.max(correlations), cmap='inferno')
        ax0.scatter(orientations[py, px], c=np.arange(sols), cmap='Greys')
        ax0.set_title('Correlation')
        
        ax1 = fig.add_subplot(122)
        
        # plotting the diffraction pattern and template
        putls.plot_template_over_pattern(self.signal.get_current_signal().data,
                                         sim_sol,
                                         ax=ax1,
                                         in_plane_angle=in_plane_angle,
                                         coordinate_system = "cartesian", 
                                         size_factor = 10,
                                         norm=mcolors.SymLogNorm(0.03),
                                         max_r = 200,
                                         mirrored_template=mirrored_sol,
                                         find_direct_beam=True,
                                         cmap = "inferno",
                                         marker_color = "green"
                                        )
        for i in [ax0, ax1]:
            i.axis("off")
            
            
    
    def init_xmap(self):
        self.xmap = iutls.results_dict_to_crystal_map(self.result, self.phasedict, diffraction_library=self.library, index=0)
        for i, space_group in enumerate(self.space_groups):
            if np.size(self.xmap[self.elements[i]]) == 0:
                continue 
            self.xmap.phases[i].space_group = space_group
    
        self.set_xmap_step_size()
        
            
    def set_xmap_step_size(self):
        """Change the step size of an orix CrystalMap
        
        """
        x = self.xmap.x * self.signal.axes_manager[0].scale
        y = self.xmap.y * self.signal.axes_manager[1].scale
        rot = self.xmap.rotations
        phaseid = self.xmap.phase_id
        prop = self.xmap.prop
        is_in_data = self.xmap.is_in_data
        phaselist = self.xmap.phases
        new_xmap = CrystalMap(rotations = rot,
                            phase_id = phaseid,
                            x = x,
                            y = y,
                            prop = prop,
                            scan_unit = "nm",
                            is_in_data = is_in_data,
                            phase_list=phaselist)
        self.xmap = new_xmap
        
    def plot_orientations_mapping(self, no_correlation = True, correlation = True):
        vectors = [Vector3d.xvector(),
                   Vector3d.yvector(),
                   Vector3d.zvector()]

       
        #if self.xmap is None:
        self.init_xmap()

        nx, ny = 256,256 #self.xmap.shape
        aspect_ratio = nx/ny
        figure_width = 4
    
        for i, element in enumerate(self.elements):
            if np.size(self.xmap[element]) == 0:
                continue 
            ckey = plot.IPFColorKeyTSL(self.symmetries[i])# Plot the key once
            fig = ckey.plot(return_figure=True)
            fig.set_size_inches(3, 2)

            for j, ploting in enumerate([no_correlation, correlation]):
                if not ploting:
                    continue

                for v in vectors:
                    ckey = plot.IPFColorKeyTSL(self.symmetries[i], direction=v)
                    overlay = None
                    if j == 1:
                        overlay = "correlation"
                        
                    if np.size(self.xmap[element]) == 0:
                        continue 
                    

                    fig = self.xmap[element].plot(value = ckey.orientation2color(self.xmap[element].orientations), 
                                                  overlay = overlay, 
                                                  figure_kwargs={'figsize': (figure_width, figure_width*aspect_ratio)}, 
                                                  return_figure=True)
                    ax = fig.get_axes()[0]
                    ax.axis('off')
                    print(self.elements[i], element)
                    o_map = np.where(self.result["phase_index"][:,:,0].flatten() == i)
                    colors = ckey.orientation2color(self.xmap[element].orientations)

                    c_map = np.zeros((256*256,3))
                    c_map[o_map] = colors
                    overlay = c_map.reshape(256,256,3)*np.expand_dims(self.result["correlation"][:,:,0], axis=-1)

                    image = np.where(np.isnan(overlay), np.zeros((256,256,3)), overlay)
                    image = image / np.max(image)
                    plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    
    def plot_correlation(self, update = False):
       if update or self.xmap is None:
           self.init_xmap()
           
       self.xmap.plot('correlation', colorbar=True)
        
       
    def plot_phase_map(self, update = False):
        if update or self.xmap is None:
            self.init_xmap()
        nx, ny = self.xmap.shape
        aspect_ratio = nx/ny
        figure_width = 4  
        fig = self.xmap.plot(figure_kwargs={'figsize': (figure_width, figure_width*aspect_ratio)}, return_figure=True)
        ax = fig.get_axes()[0]
        ax.axis('off')
        
    def plot_orientations_overlay_correlation(self, element = "muscovite"):
        if self.result is None:
            print("You must run templete match before you can plot!")
            return 
        
        if self.xmap is None:
            self.init_xmap()
            
        vectors = [Vector3d.xvector(),
                   Vector3d.yvector(),
                   Vector3d.zvector()]
        
        ckey = plot.IPFColorKeyTSL(self.symmetry)# Plot the key once
        fig = ckey.plot(return_figure=True)
        fig.set_size_inches(2, 2)
        
        nx, ny = self.xmap.shape
        aspect_ratio = nx/ny
        figure_width = 4
        
        for v in vectors:
            ckey = plot.IPFColorKeyTSL(self.symmetry, direction=v)
            fig = self.xmap[element].plot(ckey.orientation2color(self.xmap[element].orientations), overlay="correlation", figure_kwargs={'figsize': (figure_width, figure_width*aspect_ratio)}, return_figure=True)
            ax = fig.get_axes()[0]
            ax.axis('off')
 