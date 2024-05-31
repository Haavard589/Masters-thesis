# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:16:57 2024

@author: hfyhn
"""

import tkinter as tk                     
import template_matching
import numpy as np
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from itertools import product, combinations, cycle
from orix.quaternion import Orientation, symmetry
from scipy.spatial.transform import Rotation as R

elements = ["muscovite", "quartz"]
TM = template_matching.Template_matching(elements = elements)


TM.create_lib(1.0, deny_new = False, minimum_intensity=1E-20, max_excitation_error=78E-4, force_new = False, camera_length=129, half_radius = 128, reciprocal_radius=1.3072443077127025, accelerating_voltage = 80, diffraction_calibration = 0.010212846154005488,precession_angle=1.0)
#TM.create_lib(1.0, deny_new = False, minimum_intensity=0.0001, max_excitation_error=0.01, force_new = False, camera_length=200, half_radius = 128, reciprocal_radius=1.3072443077127025, accelerating_voltage = 200, diffraction_calibration = 0.010212846154005488,precession_angle=1.0)

#TM.symmetries[0] = symmetry.C1
#TM.symmetries[1] = symmetry.Oh

orientations1 = Orientation.from_euler(TM.library[elements[0]]['orientations'], symmetry=TM.symmetries[0], degrees=True)
orientations2 = Orientation.from_euler(TM.library[elements[1]]['orientations'], symmetry=TM.symmetries[1], degrees=True)


window = tk.Tk()
window.geometry("1750x1000")
canvas_musc = tk.Canvas(window)
canvas_illite = tk.Canvas(window)
canvas_both = tk.Canvas(window)
#canvas_ipf = tk.Canvas(window)
fig = Figure(figsize = (4,2), 
             dpi = 100) 

ax = fig.add_subplot(111, projection="ipf", symmetry=TM.symmetries[0])

canvas_ipf = FigureCanvasTkAgg(fig, 
                           master = window)   

fig3 = Figure(figsize = (4,2), 
             dpi = 100) 

canvas_ipf2 = FigureCanvasTkAgg(fig3, 
                           master = window)   


ax3 = fig3.add_subplot(111, projection="ipf", symmetry=TM.symmetries[1])

fig2 = Figure(figsize = (4,2), 
             dpi = 100) 

ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_aspect("auto")
ax2.set_autoscale_on(True)


canvas_box = FigureCanvasTkAgg(fig2, 
                           master = window)   
def callback(P):
    if P.replace(".", "").replace("-","").isnumeric() or P == "":
        return True
    else:
        return False
vcmd = (window.register(callback))



euler = [tk.Entry(window, validate = "all", validatecommand=(vcmd, '%P')) for _ in range(3)]

euler2 = [tk.Entry(window, validate = "all", validatecommand=(vcmd, '%P')) for _ in range(3)]

template_txt = tk.Text(window, height = 1, 
                width = 5, 
                bg = "light yellow")

distance_txt = tk.Entry(window, validate = "all", validatecommand=(vcmd, '%P'))
distance = 45
distance_txt.insert(0, distance)
template = 0

def miss_orientation(tilt1):
    tilt1 = np.array(tilt1)*np.pi/180
    return np.arccos(np.cos(tilt1[1])*np.cos(tilt1[2]))

def new_angle(tilt1, mis_ori, a):
    return tilt1[1] + np.arccos((np.cos(mis_ori) - np.sin(tilt1[2])*np.sin(a))/np.cos(tilt1[2])*np.cos(a))

def coordinate_from_euler(e1,e2):
    e1 %= 180
    e2 %= 180
    a = np.where(e1 <= 90, e1, 180 - e1) * np.pi / 180
    b = np.where(e2 <= 90, e2, 180 - e2) * np.pi / 180

    r = np.sin(a)/(1 + np.cos(a))
    x = r * np.sin(b) *  np.where(e2 > 90, -1, 1)
    y = r * np.cos(b)
    return x,y

def rotation_matrix_x(theta):
    return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])

def rotation_matrix_y(theta):
    return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])

def rotation_matrix_z(theta):
    return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])


def plotbox(euler_angle):
    d = [-1, 1]
    #a *= np.pi/180
    #b *= np.pi/180
    colors = cycle(["r","g","g","g","g","g","g","g","r","g","g","g"])
    ax2.clear()

    r = R.from_euler('zxy', np.array(euler_angle) + [0,90,0], degrees=True).as_matrix()
    for s, e in combinations(np.array(list(product(d,d,d))), 2):
        if np.sum(np.abs(s-e)) == d[1]-d[0]:
            s_rotated = np.matmul(s,r)
            e_rotated = np.matmul(e,r)

            
            
            #s_rotated = np.matmul(np.matmul(s,rotation_matrix_z(b)), rotation_matrix_y(a))
            #e_rotated = np.matmul(np.matmul(e,rotation_matrix_z(b)), rotation_matrix_y(a))
            ax2.plot3D(*zip(s_rotated,e_rotated), color=next(colors))
    
    canvas_box.draw() 

def plot(euler_angle, euler_angle2, ax, box): 
    #x, y = coordinate_from_euler(euler_angle[1],euler_angle[2])

    
    N = 300
    [c.remove() for c in ax.collections]
    #[c.remove() for c in ax3.collections]

    #theta1, theta2 = 0, 180
    #radius = 1
    #center = (0, 0)
    #w = Wedge(center, radius, theta1, theta2, fc='none', edgecolor='black')
    #ax.add_patch(w)
    

    orientations = Orientation.from_euler([euler_angle,euler_angle2], symmetry=TM.symmetries[0], degrees=True)
    ax.scatter(orientations, c = [1,2], cmap='coolwarm')
    #ax3.scatter(orientations, cmap='coolwarm')

    #ax.scatter(x, y)
    d = distance * np.pi/180
  
    #e1 = np.arccos(np.cos(d)/np.cos(e0))*180/np.pi

    
    euler_angle1 = euler_angle[1]*np.pi/180
    euler_angle2 = euler_angle[2]*np.pi/180
    
    if euler_angle[1] == 0:
        e1 = np.linspace(-np.pi, np.pi, N)
        e0 = np.ones(N) * d
        #e0 = np.append(e0, np.pi/2)
        #e1 = np.append(e1, euler_angle[2] + d)
    else:
        #e0 = np.linspace(-np.pi, np.pi, N)
        #delta1 = np.arctan((np.cos(d)-np.cos(euler_angle1))/np.sin(euler_angle2))
        #delta2 = np.arctan((np.cos(d)+np.cos(euler_angle1))/np.sin(euler_angle2))
        e0 = np.linspace(-d+euler_angle1, d+euler_angle1, N)
        e1 = euler_angle2 + np.arccos((np.cos(d)-np.cos(euler_angle1)*np.cos(e0))/(np.sin(euler_angle1)*np.sin(e0)))
        #e1 = np.append(e1, np.pi/2)
        #e0 = np.append(e0, np.arcsin(np.cos(d)))


    orientations = Orientation.from_euler(np.column_stack((np.zeros(len(e0)), e0, e1)), symmetry=symmetry.C1, degrees=False)
    ax.scatter(orientations, s = 1, color = "black")
    #ax3.scatter(orientations, s = 1, color = "black")

    orientations = Orientation.from_euler(np.column_stack((np.zeros(len(e0)), e0, 2*euler_angle2-e1)), symmetry=TM.symmetries[0], degrees=False)
    ax.scatter(orientations, s = 1, color = "black")  
   
    
    #ax3.scatter(orientations, s = 1, color = "black")      

    #x0,y0 = coordinate_from_euler(e0, e1)
    
    
    #ax.scatter(x0,y0, s = 1, color = "black")
    #x0,y0 = coordinate_from_euler(e0, 2*euler_angle[2]-e1)

    #ax.scatter(x0,y0, s = 1, color = "black")
    
    if box:
        plotbox(euler_angle)
    
        
        canvas_ipf.draw() 
    else:
        canvas_ipf2.draw() 


def draw_disc(x,y,r,color,canvas):
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return canvas.create_oval(x0, y0, x1, y1, outline = color, fill = color)
      

def Rotation_matrix(t, deg = False):
    if deg:
        t = t/180*np.pi
    return np.array([[np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)]]) 

def draw(euler_angle1, euler_angles2):
        
    #euler_angle = [0,42,0]
    #template = np.argmin(np.array([np.linalg.norm(x - euler_angle) for x in TM.library[elements[0]]['orientations']]))
    canvas_musc.delete("all")
    canvas_illite.delete("all")
    canvas_both.delete("all")
    
    e = Orientation.from_euler([0] + euler_angle1[1:], symmetry=TM.symmetries[0],degrees=True)
    
    template = np.argmin(orientations1.angle_with(e))
    signal = TM.library[elements[0]]['pixel_coords'][template]
    if euler_angle1[2] < 0:
        euler_angle1[2] = 90 + euler_angle1[2]
    
    R = Rotation_matrix(euler_angle1[0], True)
    intensity = np.log(TM.library[elements[0]]['intensities'][template])

    r = np.where(intensity < 1, 1, intensity)

    for i,point in enumerate(signal):
        point = np.matmul(R, np.array(point) - [128, 128]) + [128, 128]

        draw_disc(point[0] + 10,point[1] + 10,int(r[i]), "black", canvas_musc)   
        draw_disc(point[0] + 10,point[1] + 10,int(r[i]), "red", canvas_both)   
        
    e = Orientation.from_euler([0] + euler_angles2[1:], symmetry=TM.symmetries[1],degrees=True)
    
    template = np.argmin(orientations2.angle_with(e))    
    signal = TM.library[elements[1]]['pixel_coords'][template]
    R = Rotation_matrix(euler_angles2[0], True)
    intensity = np.log(TM.library[elements[1]]['intensities'][template])
    r = np.where(intensity < 1, 1, intensity)
    for i,point in enumerate(signal):
        point = np.matmul(R, np.array(point) - [128, 128]) + [128, 128]
        

        draw_disc(point[0] + 10,point[1] + 10, int(r[i]), "black", canvas_illite)   
        draw_disc(256 - point[0] + 10,point[1] + 10, int(r[i]), "blue", canvas_both)   


def update(euler_angles1, euler_angles2):
    for i,e in enumerate(euler):  
       
        e.delete(0, 'end')
        e.insert(0, str(np.round(euler_angles1[i],3)))
        
    for i,e in enumerate(euler2):  
        e.delete(0, 'end')
        e.insert(0, str(np.round(euler_angles2[i],3)))


    draw(euler_angles1, euler_angles2)
    plot(euler_angles1, euler_angles2, ax, True)
    plot(euler_angles2, euler_angles1, ax3, False)


def confirm():
    global distance 
    euler_angles1 = [0.0]*3
    for i, euler_angle in enumerate(euler):
        euler_angles1[i] = float(euler_angle.get())

    euler_angles2 = [0.0]*3
    for i, euler_angle in enumerate(euler2):
        euler_angles2[i] = float(euler_angle.get())


      
    distance = float(distance_txt.get())

    update(euler_angles1, euler_angles2)
    
def increase():
    update(template + 1)
    
def decrease():
    update(template - 1)

def on_click(event):

    #draw_disc(event.x, event.y, 10, "red", canvas_ipf)
    h = canvas_ipf.get_width_height()[1]
    #y = (event.y - h/my)/(2*h-2*h/my)
    y = (event.y)/(h/(1.44126*1.018/0.9842)) - 0.25
    w = canvas_ipf.get_width_height()[0]
    x = ((event.x)/(w/(1.44126*1.018*0.980690)) - 0.75115 + 0.0164)*2
    
    print(h,w,x,y,event.x,event.y)
    if y > 0 and x**2 + y**2 <= 1:
        euler_angle = np.array([0.0,np.pi/2-np.arctan((1-(x**2 + y**2))/(2*(x**2+y**2)**0.5)),np.pi/2 - np.arctan(y/x)])*180/np.pi
        template = np.argmin(np.linalg.norm(TM.library[elements[0]]['orientations'] - euler_angle, axis = 1))
        #plot(x = x, y = y)
        update(template)
    
    
def angle_modulo(angle, euler_fundamental_region):
    for i, fundamental_region in enumerate(euler_fundamental_region):
        angle %= fundamental_region
    
def keydown(event):
    if event.keycode == 13:
        confirm()
        return

    euler_angles1 = [0.0]*3
    euler_angles2 = [0.0]*3

    for i, euler_angle in enumerate(euler):
        euler_angles1[i] = float(euler_angle.get())
       
    for i, euler_angle in enumerate(euler2):
        euler_angles2[i] = float(euler_angle.get())
        
        
    z = euler_angles1[0]
    x = euler_angles1[1]
    y = euler_angles1[2]
    arrows = True
    if event.keycode == 32 or event.keycode == 37 or event.keycode == 38 or event.keycode == 39 or event.keycode == 40:
        z = euler_angles2[0]
        x = euler_angles2[1]
        y = euler_angles2[2]
        arrows = False
    
    d = 1
    #print(event)

    if event.keycode == 37 or event.keycode == 65:
        
        euler_angle = [z, x, y - d]
    elif event.keycode == 39 or event.keycode == 68:
        euler_angle = [z, x, y + d]
    elif event.keycode == 38 or event.keycode == 87:
        euler_angle = [z, x + d, y]
    elif event.keycode == 40 or event.keycode == 83:
        euler_angle = [z, x - d, y]
    elif event.keycode == 32 or event.keycode == 82:
        euler_angle = [z + d, x, y]
    else:
        return
    
    
    
    if arrows:
        for i, fundamental_region in enumerate(TM.symmetries[0].euler_fundamental_region):
            euler_angle[i] %= fundamental_region
        update(euler_angle, euler_angles2)
    else:
        for i, fundamental_region in enumerate(TM.symmetries[1].euler_fundamental_region):
            euler_angle[i] %= fundamental_region
        update(euler_angles1, euler_angle)


        
def main():
    global lbl_value, windom, canvas_musc, canvas_illite, canvas_ipf
    
    
    window.rowconfigure([0,1,2,3], minsize=50, weight=1)
    window.columnconfigure([0, 1, 2,3,4,5,6,7,8], minsize=50, weight=1)
    window.bind("<KeyPress>", keydown)
    canvas_musc.grid(row = 2, column = 0, columnspan = 3, sticky="nsew")
    canvas_illite.grid(row = 2, column = 6, columnspan = 3, sticky="nsew")
    canvas_both.grid(row = 2, column = 3, columnspan = 3, sticky="nsew")
    canvas_ipf2.get_tk_widget().grid(row = 3, column = 3, columnspan = 2, ipadx=40, ipady=20)
    canvas_ipf.get_tk_widget().grid(row=3, column=0, columnspan = 2, ipadx=40, ipady=20)
    #canvas_ipf.get_tk_widget().grid(row=3, column=0, columnspan = 2, ipadx=40, ipady=20)

    canvas_ipf.mpl_connect('button_press_event', on_click)
    canvas_box.get_tk_widget().grid(row=3, column=6, columnspan = 6, ipadx=40, ipady=20)

    #canvas_ipf.bind("<Button-1>", on_click)
    

    #btn_decrease = tk.Button(master=window, text="-", command=decrease)
    #btn_decrease.grid(row=0, column=0, rowspan = 1,columnspan = 3,  sticky="nsew")
    
    
    #btn_increase = tk.Button(master=window, text="+", command=increase)
    #btn_increase.grid(row=0, column=6, rowspan = 1, columnspan = 3,sticky="nsew")

    btn_confirm = tk.Button(master=window, text="Update", command=confirm)
    btn_confirm.grid(row=0, column=5, rowspan = 1,columnspan = 1,  sticky="nsew")


    for i,e in enumerate(euler):
        e.grid(row = 1, column = i, columnspan = 1, sticky="nsew")

    for i,e in enumerate(euler2):
        e.grid(row = 1, column = 3 + i, columnspan = 1, sticky="nsew")


    template_txt.grid(row = 0, column = 4, columnspan = 1, sticky="nsew")

    distance_txt.grid(row = 0, column = 3, columnspan = 1, sticky="nsew")

    update([0,0,0], [0,0,0])
    
    window.mainloop()
    Orientation
if __name__ == "__main__":
    main()