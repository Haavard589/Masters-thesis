# method for line detection
#import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt 
from sklearn.decomposition import NMF, PCA
import hyperspy.api as hs
from scipy.ndimage import rotate

stick_man = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
])


def plot(H, W):
    for i, component in enumerate(H):
    
        plt.figure()
    
        plt.imshow(component.reshape((M,N)), norm = "symlog")
        
        #plt.plot(np.linspace(65*0.005, 2000*0.005, 2000 - 65), component)
        plt.axis('off')      


        plt.figure()
        
        plt.imshow(W[:,:,i])
        plt.axis('off')      
        
        
    plt.figure()
    plt.imshow(stick_man)
    plt.axis('off')

img = np.load(r'C:\Users\hfyhn\Downloads\testingNMF.npy')
img = rotate(img, angle=14.8)
img = img[500:2000,500:2000]
M = img.shape[0]//50
N = img.shape[1]//50
def stuff():
    plt.imshow(img)
    print(np.shape(img))
    tiles = np.array([[img[x:x+M,y:y+N] for x in range(0,img.shape[0],M)] for y in range(0,img.shape[1],N)])
    print(np.shape(tiles))
    
    for i in range(16):
        #for j in range(64):
            plt.figure()
            plt.imshow(tiles[i,0])
        
    shape = np.shape(tiles)
    reshaped_array = tiles.reshape((shape[0]*shape[1], shape[2]*shape[3]))
    
    # Specify the number of components (features) for factorization
    n_components = 10
    
    # Create the NMF model
    model = NMF(n_components=n_components, init='random', random_state=42, max_iter = 200)
    
    # Fit the model to the flattened array
    W = model.fit_transform(reshaped_array)
    w = W.reshape((shape[0],shape[1],n_components))
    
    plot(model.components_, w)
    
    return
    #img = rotate(img, angle=14.8)
    
    #location = r"C:\Users\hfyhn\Documents\Skole\Fordypningsoppgave\Data\excite cambridge\D5 muscovite\20230711_124805\hyperspy\crystal_3_blob_2.hspy"
    
    #signal = hs.load(location, lazy=False)
    #signal = signal.inav[13:37,106:118]
    
    #signal = signal.inav[:,:50]
    #img = signal.T.sum().data
    plt.imshow(img)
    
    pca = PCA(n_components=20)
    pca.fit(img)
    plt.figure()
    plt.plot(pca.explained_variance_ratio_, marker = "o")
    plt.figure()
    plt.plot(pca.singular_values_, marker = "x")
    
    n_components = 2
    
    model = NMF(n_components=n_components, init='random', random_state=42, max_iter = 200)
    
    #reshaped_array = data_matrix.reshape((shape[0]*shape[1], shape[2]*shape[3]))
    
    # Fit the model to the flattened array
    W = model.fit_transform(img)
    #w = W.reshape((shape[0],shape[1],n_components))
    
    plot(model.components_, W)
    