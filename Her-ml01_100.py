import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.colors import LogNorm
import  matplotlib.transforms as transforms
#from collections import defaultdict
import time 
import os
#from PIL import Image

start_time = time.time()  # Time to execute the program.
path=''
path1 = '/home/k1629656/Jets/Herwig/herwig_gg2gg/'
path2='/home/k1629656/Jets/Herwig/herwig_qqbar2gg/'  # Path of the program
impath='/home/k1629656/Jets/gluons'#'/usr/Herwig/gluon_jets/images_100-110GeV'# _100-110GeV/' # Folder to which the image files are saving 
imageFile1='herwig_gg2gg_60000_100-110_jet_image.dat'  # The generated data from Herwig simulation
imageFile2 ='herwig_qqbar2gg_60000_100-110_jet_image.dat'
df_image1 = pd.read_csv(path1+imageFile1,dtype=None,delimiter=",")
df_image2 = pd.read_csv(path2+imageFile2,dtype=None,delimiter=",")

df_image = pd.concat([df_image1,df_image2],ignore_index=True,axis=0)

df_image.to_csv('herwig_gloun_jet_120000_100-110_jet_image.dat',index = False)
#imageFile = 'herwig_gloun_jet_120000_100-110_jet_image.dat'
#df_image=pd.read_csv(path+imageFile,dtype= None, delimiter=",")   # Data=> Dataframe
df_image= df_image.iloc[:134533,:] # index of the last line in 125000th Jet

g= df_image.loc[df_image['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
g_i = g.index # Indices of '@@@'
#print g_i[5000]
a = np.size(g_i)# Size of the index array

delta = 0.8/33 # length of each pixel


boxmax=0.4
r = 0.00001
total_pT = 0.00



mean = np.zeros((33,33),dtype=float)
image_new=np.zeros((33,33))
image= np.zeros((33,33,a),dtype=float)
x,y,z=np.shape(image)
#with open('img','w') as f:
for index,j in enumerate(g_i): # For all values and their corresponding indices in g_i      
	if g_i[index-1]+1<g_i[index]:
        	        
        	image_array= df_image.iloc[g_i[index-1]+1:g_i[index],:]
                image_array= np.array(image_array[:],dtype=float)
                
                  #PT
                #=======#                
                pT =np.array(image_array[:,[2]])# Call the third column in all image file as 'pT' 
                pT =pT.reshape((np.size(pT),))# Reshape it for plotting otherwise errors                
                        
                        #f.write(str(pT[0])+'\n')
                  #ETA
                #=======#
                eta =np.array(image_array[:,[0]])# Call the first column in all image file as 'eta' 
                eta =eta.reshape((np.size(eta),)) # Reshape it for plotting otherwise errors 
                mean_eta = np.average(eta,axis=0,weights=pT)# Weighted mean      
                eta =(eta- mean_eta) # Centering
                size = np.size(eta)# Number of particles in the jet size(eta)=size(phi)=size(pT)
                

                  #PHI
                #=======#
                phi =np.array(image_array[:,1]) # Call the second column in all image file as 'phi' 
                phi =phi.reshape((np.size(phi),)) # Reshape it for plotting otherwise errors        
                mean_phi = np.average(phi,axis=0,weights=pT)#Weighted mean
                phi =(phi-mean_phi)#Centering

                   #PIXEL 
                #=======#                            
                for i in range(0,size): 
                	if (abs(eta[i])<=boxmax): # if the value of -0.4<=eta<=0.4
                	       	image[16,16,index]=pT[0] # The center pixel is JET PT
                	        j = round (((eta[i])/delta)) # a number between -16 to 16
                                j = j+16 #j+16 = 16 if j =0 => The central pixel value
                                if (abs(phi[i])<=boxmax): # a number between -16 to 16
                                	k = round (((phi[i])/delta))
                                        k = k+16 #k+16 = 16 if k =0 => The central pixel value
                                        image[j,k,index]= image[j,k,index]+pT[i]#image is an 3D array, each layer gives an image
                                        #f.write(str(image[j,k,index])+'\n')
                                                
                
		total_pT=np.sum(image)# Total pT in an image
		image[:,:,index] = image[:,:,index]/total_pT #Normalisation


                        
#Plotting average image before zero centering and standardizing
#==============================================================#
mean= np.mean(image,axis=2)#Average of all images
std = np.std(image,axis=2)#Std of all images
minpt=np.amin(image)
maxpt=np.amax(image)
print minpt
print maxpt
plt.pcolormesh(mean,cmap =plt.cm.jet,)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
v = np.arange(minpt,maxpt)
cb = plt.colorbar()
plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
plt.grid(True)
av1_imfile ='av_image_before_step4-5_herwig1_2.jpeg'
plt.savefig(os.path.join(impath,av1_imfile)) # Save image files in the given folder
plt.close()


for index,j in enumerate(g_i): # For all values and their corresponding indices in g_i      
        if g_i[index-1]+1<g_i[index]:
                for k in range(0,x):
                        for j in range(0,y):
                                image[j,k,index]=image[j,k,index]-mean[j,k] #Step 4 : Zerocentering
                                image[j,k,index]=image[j,k,index]/(std[j,k]+r) #Step5: Standardize
minpt=np.amin(image)
maxpt=np.amax(image)
for index,j in enumerate(g_i): # For all values and their corresponding indices in g_i      
        if g_i[index-1]+1<g_i[index]:


                # Plotting
                #=========#
                image_new=image[:,:,index]
                #z_min, z_max = image_new.min(), np.abs(image_new).max()
                plt.rcParams['axes.facecolor'] = 'white'
                fig_size = plt.rcParams["figure.figsize"]
                #fig_size[0] = 0.4155844155
                #fig_size[1] = 0.4155844155
                plt.rcParams["figure.figsize"] = fig_size
                plt.axes().set_aspect('equal')
                plt.pcolormesh(image_new,cmap=plt.cm.jet,vmin=minpt,vmax=maxpt)
                plt.xlim([0,33])
                plt.ylim([0,33])
                plt.xticks([])
                plt.yticks([])
                #cb = plt.colorbar()
                #plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
                #plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
                #cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')#

                #plt.grid(True)

                imfile ='image_herwig1_2_%002d.jpeg'%index# Name image files as image_01.png
                plt.savefig(os.path.join(impath,imfile),bbox_inches='tight', pad_inches = 0) # Save image files in the given folder
                plt.close()

                


                                
# Plotting Average Image
#========================

mean=np.mean(image,axis=2)
plt.pcolormesh(mean,cmap =plt.cm.jet,)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
#cb = plt.colorbar()
#plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
#plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
#cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
plt.grid(True)
av2_imfile ='av_image_herwig1_2.jpeg'
plt.savefig(os.path.join(impath,av2_imfile)) # Save image files in the given folder
plt.close()


print (time.time()-start_time), "seconds" # Print the time taken for this program to execute this
