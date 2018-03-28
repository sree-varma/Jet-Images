import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
from matplotlib.colors import LogNorm
import  matplotlib.transforms as transforms
#from collections import defaultdict
import time
#import pp
import os
import sys
from multiprocessing import Process,Queue
from numba import guvectorize
import cv2
#guvectorize(['void(float32[:,:], float32[:,:], float32[:,:])'],
#             '(m,n),(n,p)->(m,p)', target='cuda')


delta = 0.8/33 # length of each pixel
boxmax=0.4
r = 0.00001
total_pT = 0.00
total_pT3=0.00
#q = Queue()
value=0.0

def image_array(index_array,channel_1,channel_2,channel_3,df_image,q):
    global temp1
    for index,j in enumerate(index_array): # For all values and their corresponding indices in g_i
	
        #print "for all values and their corresponding indices in g_i"
        if index_array[index-1]+1<index_array[index]:
                #print "if there are vales between index"
                image= df_image.iloc[index_array[index-1]+1:index_array[index],:]
                image= np.array(image[1:],dtype=float)
                #print image
                  #PT
                #=======#
                pT =np.array(image[:,[2]])# Call the third column in all image file as 'pT'
                
                pT =pT.reshape((np.size(pT),))# Reshape it for plotting otherwise errors
                #print np.sum(pT)
                        #f.write(str(pT[0])+'\n')
                  #ETA
                #=======#
                eta =np.array(image[:,[0]])# Call the first column in all image file as 'eta'
                eta =eta.reshape((np.size(eta),)) # Reshape it for plotting otherwise errors
                mean_eta = np.ma.average(eta,axis=0,weights=pT)# Weighted mean     \

                eta =(eta- mean_eta) # Centering
                size = np.size(eta)# Number of particles in the jet size(eta)=size(phi)=size(pT)

                  #PHI
                #=======#
                phi =np.array(image[:,1]) # Call the second column in all image file as 'phi'
                phi =phi.reshape((np.size(phi),)) # Reshape it for plotting otherwise errors
                mean_phi = np.ma.average(phi,axis=0,weights=pT)#Weighted mean
                phi =(phi-mean_phi)#Centering
		
		  #CHARGE
                #=========#
                charge = np.array(image[:,3])
		#charge = charge.astype(float)
	
                charge =charge.reshape((np.size(charge),))


                   #PIXEL
                  #=======#

		
	        for i in range(0,size):
        	        if (abs(eta[i])<=boxmax):#boxmax): # if the value of -0.4<=eta<=0.
          	                j = int(round(((eta[i])/delta))) # a number between -16 to 16
        	                j = j+16 #j+16 = 16 if j =0 => The central pixel value
				#j = j(dtype= 'int')
        	                if (abs(phi[i])<=boxmax): # a number between -16 to 16
           	                      	k = int(round(((phi[i])/delta)))
        	                        k = k+16 #k+16 = 16 if k =0 => The central pixel value.

					if charge[i]==0:#or charge[i]<0 :
						channel_2[j,k,index]= channel_2[j,k,index]+pT[i]#image is an 3D array, each layer gives an image
						
						
					else:
						channel_3[j,k,index]= channel_3[j,k,index]+pT[i]
						channel_1[j,k,index]=1.0
        	                        
        	                        
		total_pT3=np.sum(channel_3[:,:,index])
		if total_pT3==0 or total_pT3=='nan':
			channel_3[:,:,index]=channel_3[:,:,index]/(total_pT3+0.00001)
                #print total_pT
		else:
                	channel_3[:,:,index]=channel_3[:,:,index]/total_pT3 #Normalisation
		
			
		total_pT=np.sum(channel_2[:,:,index])
		if total_pT==0 or total_pT=='nan':
			channel_2[:,:,index]=channel_2[:,:,index]/(total_pT+0.00001)
                #print total_pT
		else:
                	channel_2[:,:,index]=channel_2[:,:,index]/total_pT #Normalisation

		total_multiplicity=np.sum(channel_1[:,:,index])
		if total_multiplicity==0 or total_multiplicity=='nan':
			channel_1[:,:,index]=channel_1[:,:,index]/(total_multiplicity+0.00001)
                #print total_pT
		else:
                	channel_1[:,:,index]=channel_1[:,:,index]/total_multiplicity #Normalisation
    l=[channel_1,channel_2,channel_3] 
    q.put(l)#channel_1,channel_2,channel_3)
    return l#channel_1,channel_2,channel_3 #image_array



def jet_images(index_array,x,y,image_array,mean,std):
    for index,j in enumerate(index_array): # For all values and their corresponding indices in g_i
        if index_array[index-1]+1<index_array[index]:
            for k in range(0,x):
                for j in range(0,y):
                    image_array[j,k,index]=(image_array[j,k,index]-mean[j,k]) #Step 4 & 5: Zerocentering and Standardizing
                    image_array[j,k,index]=image_array[j,k,index]/(std[j,k]+r) #Step5: Standardize

    return image_array


          

def shuffle(image_array):
    x,y,z = image_array.shape
    zero_array=np.zeros((x,y,z),dtype=float)
    a= np.arange(0,z)
    b=np.random.shuffle(a)
    for i in range(0,z):
        zero_array[:,:,a[i]]=zero_array[:,:,a[i]]+image_array[:,:,i]
    return zero_array




def plotting(index_array,image_array,image_name,train_path,test_path):
    for index,j in enumerate(index_array): # For all values and their corresponding indices in gi_quark
        if index_array[index-1]+1<index_array[index]:

                # Plotting
                #=========#
                image_array_new=image_array[:,:,:,index]
		image_array_new=image_array_new*(256.)#/np.amax(image_array_new)) 
               
                if index<100001:
                        imfile ='_%002d.jpeg'%index# Name image files as image_01.png
			img=cv2.imwrite(os.path.join(train_path,image_name+imfile), image_array_new)# Save image files in the given folder
                        
                else:
                        imfile ='_%002d.jpeg'%index# Name image files as image_01.png
                        img=cv2.imwrite(os.path.join(test_path,image_name+imfile), image_array_new) # Save image files in the given folder
 
                        


