# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit
from IPython.display import display
from qiskit import transpile
from qiskit_aer import Aer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from PIL import Image
#from skimage.transform import resize
import cv2
import os
import time
style.use('bmh')
import resource

input_images_path = r"masksEspecial/"
files_names = os.listdir(input_images_path)
#print(files_names)

tams = [128]
ti=time.time()
media = 0
numImages = 0
for tam in tams:
    numImages = 0
    media = 0
    for file_name in files_names:
        st = time.time()

        img = cv2.imread('masksEspecial/'+file_name)
        imgRes = cv2.resize(img, dsize=(tam, tam), interpolation=cv2.INTER_CUBIC)

        # Convert the RBG component of the image to B&W image, as a numpy (uint8) array
        image_size = tam
        image = []
        for i in range(image_size):
            image.append([])
            for j in range(image_size):
                image[i].append(imgRes[i][j][0] / 255)

        image = np.array(image)
        #print('Image shape (numpy array):', image.shape)

        # Function for plotting the image using matplotlib
        def plot_image(img, title: str):
            plt.axis('off')
            plt.title(title)
            plt.xticks(range(img.shape[0]))
            plt.yticks(range(img.shape[1]))
            plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='gist_gray')
            plt.savefig('result'+str(tam)+'/'+title+file_name)
            #plt.show()
            
        #plot_image(image, 'Original Image')

        # Convert the raw pixel values to probability amplitudes
        def amplitude_encode(img_data):
            
            # Calculate the RMS value
            rms = np.sqrt(np.sum(np.sum(img_data**2, axis=1)))
            
            # Create normalized image
            image_norm = []
            for arr in img_data:
                for ele in arr:
                    image_norm.append(ele / rms)
                
            # Return the normalized image as a numpy array
            return np.array(image_norm)

        # Get the amplitude ancoded pixel values
        # Horizontal: Original image
        image_norm_h = amplitude_encode(image)

        # Vertical: Transpose of Original image
        image_norm_v = amplitude_encode(image.T)

        # Initialize some global variable for number of qubits
        if(tam == 8):
            data_qb = 6
        elif(tam == 16):
            data_qb = 8
        elif(tam == 32):
            data_qb = 10
        elif(tam == 64):
            data_qb = 12
        elif(tam == 128):
            data_qb = 14
        elif(tam == 256):
            data_qb = 16
        elif(tam == 512):
            data_qb = 18
        anc_qb = 1
        total_qb = data_qb + anc_qb

        # Initialize the amplitude permutation unitary
        D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)

        # Create the circuit for horizontal scan
        qc_h = QuantumCircuit(total_qb)
        qc_h.initialize(image_norm_h, range(1, total_qb))
        qc_h.h(0)
        qc_h.unitary(D2n_1, range(total_qb))
        qc_h.h(0)
        #display(qc_h.draw('mpl', fold=-1))

        # Create the circuit for vertical scan
        qc_v = QuantumCircuit(total_qb)
        qc_v.initialize(image_norm_v, range(1, total_qb))
        qc_v.h(0)
        qc_v.unitary(D2n_1, range(total_qb))
        qc_v.h(0)
        #display(qc_v.draw('mpl', fold=-1))

        # Combine both circuits into a single list
        circ_list = [qc_h, qc_v]

        # Simulating the cirucits
        back = Aer.get_backend('statevector_simulator')
        compiled = transpile(circ_list, backend=back)
        job = back.run(compiled)
        results = job.result()        
        sv_h = results.get_statevector(qc_h)
        sv_v = results.get_statevector(qc_v)

        from qiskit.visualization import array_to_latex
        #print('Horizontal scan statevector:')
        #display(array_to_latex(sv_h[:30], max_size=30))
        #print()
        #print('Vertical scan statevector:')
        #display(array_to_latex(sv_v[:30], max_size=30))

        # Classical postprocessing for plotting the output

        # Defining a lambda function for
        # thresholding to binary values
        threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)

        # Selecting odd states from the raw statevector and
        # reshaping column vector of size 64 to an 8x8 matrix
        edge_scan_h = np.abs(np.array([1 if threshold(sv_h[2*i+1].real) else 0 for i in range(2**data_qb)])).reshape(tam, tam)
        edge_scan_v = np.abs(np.array([1 if threshold(sv_v[2*i+1].real) else 0 for i in range(2**data_qb)])).reshape(tam, tam).T

        # Plotting the Horizontal and vertical scans
        plot_image(edge_scan_h, 'Horizontal scan output')
        plot_image(edge_scan_v, 'Vertical scan output')

        # Combining the horizontal and vertical component of the result
        edge_scan_sim = edge_scan_h | edge_scan_v

        # Plotting the original and edge-detected images
        #plot_image(image, 'Original image')
        plot_image(edge_scan_sim, '')
        #cv2.imwrite("result.png", edge_scan_sim)
        numImages = numImages+1
        
        et = time.time()
        temp=et-st
        print("Ha tardado",round(temp,3),"segundos en calular la imagen")
        media = media + (et - st)

    print("Ha tardado",round(media,3),"segundos en calcular las",round(numImages,3),"imágenes del tamaño",round(tam,3))    
    media = media / numImages
    print("Tiempo medio - tamaño "+str(tam)+":") 
    print(round(media,3))
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Memoria máxima usada: {mem} KB")
tf=time.time()
tt=tf-ti
print("Tiempo total: ", round(tt,3))
