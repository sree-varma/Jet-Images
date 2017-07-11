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
#pathq1='/home/k1629656/Jets/Herwig/herwig_gg2qqbar/'
#pathq2='/home/k1629656/Jets/Herwig/herwig_qq2qq/'  # Path of the program
#pathq3='/home/k1629656/Jets/Herwig/herwig_qqbar2qqbar/'
#imqpath='/home/k1629656/Jets/quarks/'#'/usr/Herwig/quark_jets/images_100-110GeV'#\
train_qimage_path='/home/k1629656/Jets/train/quarks/'#'/usr/Herwig/gluon_jets/images_100-110GeV'#\
test_qimage_path='/home/k1629656/Jets/test/quarks/'



#pathg1 = '/home/k1629656/Jets/Herwig/herwig_gg2gg/'
#pathg2='/home/k1629656/Jets/Herwig/herwig_qqbar2gg/'  # Path of the program
train_gimage_path='/home/k1629656/Jets/train/gluons/'#'/usr/Herwig/gluon_jets/images_100-110GeV'#\
test_gimage_path='/home/k1629656/Jets/test/gluons/'

#FILES

# QUARKS
#========#

#quarkFile1='herwig_gg2qqbar_40000_100-110_jet_image.dat'  # The generated data f\rom Herwig simulation
#quarkFile2='herwig_qq2qq_40000_100-110_jet_image.dat'
#quarkFile3='herwig_qqbar2qqbar_40000_100-110_jet_image.dat'
#df_quark1 = pd.read_csv(pathq1+quarkFile1,dtype=None,delimiter=",")
#df_quark2 = pd.read_csv(pathq2+quarkFile2,dtype=None,delimiter=",")
#df_quark3 = pd.read_csv(pathq3+quarkFile3,dtype=None,delimiter=",")
#df_quark = pd.concat([df_quark1,df_quark2,df_quark3],ignore_index=True,axis=0)
#df_quark.to_csv('herwig_quark_jet_120000_100-110_jet_image.dat',index = False)
#df_quark= df_quark.iloc[:113420,:] # index of the last line in 125000th Jet


pathq1= '/usr/Pythia/quark_jets/'
quarkFile1='pythia_quark_jet_30000_200-220_jet_image.dat'
df_quark = pd.read_csv(pathq1+quarkFile1,dtype=None,delimiter=",")
df_quark=df_quark.iloc[:129966,:]#208976,:]#62709,:]

# GLUONS
#========#


#gluonFile1='herwig_gg2gg_60000_100-110_jet_image.dat'  # The generated data from Herwig simulation
#gluonFile2 ='herwig_qqbar2gg_60000_100-110_jet_image.dat'
#df_gluon1 = pd.read_csv(pathg1+gluonFile1,dtype=None,delimiter=",")
#df_gluon2 = pd.read_csv(pathg2+gluonFile2,dtype=None,delimiter=",")
#df_gluon = pd.concat([df_gluon1,df_gluon2],ignore_index=True,axis=0)
#df_gluon.to_csv('herwig_gloun_jet_120000_100-110_jet_image.dat',index = False)
#df_gluon= df_gluon.iloc[:134533,:] # index of the last line in 125000th Jet


pathg1= '/usr/Pythia/gluon_jets/'
gluonFile1='pythia_gloun_jet_30000_200-220_jet_image.dat'
df_gluon = pd.read_csv(pathg1+gluonFile1,dtype=None,delimiter=",")
df_gluon=df_gluon.iloc[:196330,:]#298560,:]#89633,:]














gquark= df_quark.loc[df_quark['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
gi_quark = gquark.index # Indices of '@@@'#
print gi_quark[4000]#10000_100-110 #3000_100-110


ggluon= df_gluon.loc[df_gluon['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
gi_gluon = ggluon.index # Indices of '@@@'
print gi_gluon[4000]

delta = 0.8/33 # length of each pixel
boxmax=0.4
r = 0.00001
total_pT = 0.00


mean_quark = np.zeros((33,33),dtype=float)
mean_gluon = np.zeros((33,33),dtype=float)

quark_new=np.zeros((33,33))
gluon_new = np.zeros((33,33))

a = np.size(gi_quark)
b=np.size(gi_gluon)

quark= np.zeros((33,33,a),dtype=float)
gluon= np.zeros((33,33,b),dtype=float)

x,y,z=np.shape(quark)



#QUARKS#


for index,j in enumerate(gi_quark): # For all values and their corresponding indices in g_i
        if gi_quark[index-1]+1<gi_quark[index]:

                quark_array= df_quark.iloc[gi_quark[index-1]+1:gi_quark[index],:]
                quark_array= np.array(quark_array[1:],dtype=float)

                  #PT
                #=======#
                pT_quark =np.array(quark_array[:,[2]])# Call the third column in all image file as 'pT'
                pT_quark =pT_quark.reshape((np.size(pT_quark),))# Reshape it for plotting otherwise errors

                        #f.write(str(pT[0])+'\n')
                  #ETA
                #=======#
                eta_quark =np.array(quark_array[:,[0]])# Call the first column in all image file as 'eta'
                eta_quark =eta_quark.reshape((np.size(eta_quark),)) # Reshape it for plotting otherwise errors
                mean_eta_quark = np.average(eta_quark,axis=0,weights=pT_quark)# Weighted mean     \

                eta_quark =(eta_quark- mean_eta_quark) # Centering
                size = np.size(eta_quark)# Number of particles in the jet size(eta)=size(phi)=size(pT)

                  #PHI
                #=======#
                phi_quark =np.array(quark_array[:,1]) # Call the second column in all image file as 'phi'
                phi_quark =phi_quark.reshape((np.size(phi_quark),)) # Reshape it for plotting otherwise errors
                mean_phi_quark = np.average(phi_quark,axis=0,weights=pT_quark)#Weighted mean
                phi_quark =(phi_quark-mean_phi_quark)#Centering
                   #PIXEL
                #=======#
                for i in range(0,size):
                        if (abs(eta_quark[i])<=boxmax): # if the value of -0.4<=eta<=0.4
                               # quark[16,16,index]=pT_quark[0] # The center pixel is JET PT
                                j = round (((eta_quark[i])/delta)) # a number between -16 to 16
                                j = j+16 #j+16 = 16 if j =0 => The central pixel value
                                if (abs(phi_quark[i])<=boxmax): # a number between -16 to 16
                                        k = round (((phi_quark[i])/delta))
                                        k = k+16 #k+16 = 16 if k =0 => The central pixel value
                                        quark[j,k,index]= quark[j,k,index]+pT_quark[i]#image is an 3D array, each layer gives an image
                                        #f.write(str(image[j,k,index])+'\n')

                total_pT=np.sum(quark)# Total pT in an image
                quark[:,:,index] = quark[:,:,index]/total_pT #Normalisation



for index,j in enumerate(gi_gluon): # For all values and their corresponding indices in g_i
        if gi_gluon[index-1]+1<gi_gluon[index]:

                gluon_array= df_gluon.iloc[gi_gluon[index-1]+1:gi_gluon[index],:]
                gluon_array= np.array(gluon_array[1:],dtype=float)

                  #PT
                #=======#
                pT_gluon =np.array(gluon_array[:,[2]])# Call the third column in all image file as 'pT'
                pT_gluon =pT_gluon.reshape((np.size(pT_gluon),))# Reshape it for plotting otherwise errors

                        #f.write(str(pT[0])+'\n')
                  #ETA
                #=======#
                eta_gluon =np.array(gluon_array[:,[0]])# Call the first column in all image file as 'eta'
                eta_gluon =eta_gluon.reshape((np.size(eta_gluon),)) # Reshape it for plotting otherwise errors
                mean_eta_gluon = np.average(eta_gluon,axis=0,weights=pT_gluon)# Weighted mean     \

                eta_gluon =(eta_gluon- mean_eta_gluon) # Centering
                size = np.size(eta_gluon)# Number of particles in the jet size(eta)=size(phi)=size(pT)

                  #PHI
                #=======#
                phi_gluon =np.array(gluon_array[:,1]) # Call the second column in all image file as 'phi'
                phi_gluon =phi_gluon.reshape((np.size(phi_gluon),)) # Reshape it for plotting otherwise errors
                mean_phi_gluon = np.average(phi_gluon,axis=0,weights=pT_gluon)#Weighted mean
                phi_gluon =(phi_gluon-mean_phi_gluon)#Centering
                   #PIXEL
                #=======#
                for i in range(0,size):
                        if (abs(eta_gluon[i])<=boxmax): # if the value of -0.4<=eta<=0.4
                                #gluon[16,16,index]=pT_gluon[0] # The center pixel is JET PT
                                j = round (((eta_gluon[i])/delta)) # a number between -16 to 16
                                j = j+16 #j+16 = 16 if j =0 => The central pixel value
                                if (abs(phi_gluon[i])<=boxmax): # a number between -16 to 16
                                        k = round (((phi_gluon[i])/delta))
                                        k = k+16 #k+16 = 16 if k =0 => The central pixel value
                                        gluon[j,k,index]= gluon[j,k,index]+pT_gluon[i]#image is an 3D array, each layer gives an image
                                        #f.write(str(image[j,k,index])+'\n')

                total_pT=np.sum(gluon)# Total pT in an image
                gluon[:,:,index] = gluon[:,:,index]/total_pT #Normalisation









#Plotting average image before zero centering and standardizing
#==============================================================#
mean_quark= np.mean(quark[:,:,:4000],axis=2)#Average of all images
std_quark = np.std(quark[:,:,:4000],axis=2)#Std of all images
#f.write(str(image[j,k,index])+'\n')

#print std_quark


mean_gluon=np.mean(gluon,axis=2)
std_gluon=np.std(gluon,axis=2)
#print std_gluon
#f.write(str(image[j,k,index])+'\n')
mean=np.zeros((33,33,2),dtype=float)
mean[:,:,0]=mean_quark
mean[:,:,1]=mean_gluon

std=np.zeros((33,33,2),dtype=float)
std[:,:,0]=std_quark
std[:,:,1]=std_gluon


mean=np.mean(mean,axis=2)
std=np.std(std,axis=2)
#f.write(str(image[j,k,index])+'\n')




qminpt=np.amin(quark)
qmaxpt=np.amax(quark)
#print minpt
#print maxpt
plt.pcolormesh(mean_quark,cmap =plt.cm.jet)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
#v = np.arange(qminpt,qmaxpt)
v=np.arange(-1.0,1.0)
cb = plt.colorbar(ticks=v)
plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
plt.grid(True)
av1_imqfile ='av_image_before_step4-5_pythia3_4_5_.jpeg'
plt.savefig(os.path.join(path,av1_imqfile)) # Save image files in the given folder
plt.close()


gminpt=np.amin(gluon)
gmaxpt=np.amax(gluon)
#print minpt
#print maxpt
plt.pcolormesh(mean_gluon,cmap =plt.cm.jet)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
#v = np.arange(gminpt,gmaxpt)
v=np.arange(-1.0,1.0)
cb = plt.colorbar(ticks=v)
plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
plt.grid(True)
av1_imgfile ='av_image_before_step4-5_pyhtia1_2_.jpeg'
plt.savefig(os.path.join(path,av1_imgfile)) # Save image files in the given folder
plt.close()







for index,j in enumerate(gi_quark): # For all values and their corresponding indices in g_i
        if gi_quark[index-1]+1<gi_quark[index]:
                for k in range(0,x):
                        for j in range(0,y):
                                quark[j,k,index]=quark[j,k,index]-mean[j,k] #Step 4 : Zerocentering
                                quark[j,k,index]=quark[j,k,index]/(std[j,k]+r) #Step5: Standardize





for index,j in enumerate(gi_gluon): # For all values and their corresponding indices in g_i
        if gi_gluon[index-1]+1<gi_gluon[index]:
                for k in range(0,x):
                        for j in range(0,y):
                                gluon[j,k,index]=gluon[j,k,index]-mean[j,k] #Step 4 : Zerocentering
                                gluon[j,k,index]=gluon[j,k,index]/(std[j,k]+r) #Step5: Standardize



#minpt=np.amin(image)
#maxpt=np.amax(image)


for index,j in enumerate(gi_quark): # For all values and their corresponding indices in gi_quark
        if gi_quark[index-1]+1<gi_quark[index]:
                # Plotting
                #=========#
                quark_new=quark[:,:,index]
                #z_min, z_max = image_new.min(), np.abs(image_new).max()

                plt.rcParams['axes.facecolor'] = 'white'
                fig_size = plt.rcParams["figure.figsize"]
                fig_size[0] = 0.4155844155
                fig_size[1] = 0.4155844155
                plt.figure(frameon=False)
                plt.axis('off')
                plt.rcParams["figure.figsize"] = fig_size
                plt.axes().set_aspect('equal')
                plt.pcolormesh(quark_new,cmap=plt.cm.jet,vmin=-1.0,vmax=+1.0)
                plt.xlim([0,33])
                plt.ylim([0,33])
                plt.xticks([])
                plt.yticks([])

                #cb = plt.colorbar()
                #plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
                #plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
                #cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')#

                #plt.grid(True)
                if index<4000:
                        qimfile ='image_pythia3_4_5_%002d.jpeg'%index# Name image files as image_01.png
                        plt.savefig(os.path.join(train_qimage_path,qimfile),bbox_inches='tight', pad_inches = 0) # Save image files in the given folder
                        plt.close()
                else:
                        qimfile ='image_pythia3_4_5_%002d.jpeg'%index# Name image files as image_01.png
                        plt.savefig(os.path.join(test_qimage_path,qimfile),bbox_inches='tight', pad_inches = 0) # Save image files in the given folder
                        plt.close()


for index,j in enumerate(gi_gluon): # For all values and their corresponding indices in gi_quark
        if gi_gluon[index-1]+1<gi_gluon[index]:
                # Plotting
                #=========#
                gluon_new=gluon[:,:,index]
                #z_min, z_max = image_new.min(), np.abs(image_new).max()

                plt.rcParams['axes.facecolor'] = 'white'
                fig_size = plt.rcParams["figure.figsize"]
                fig_size[0] = 0.4155844155
                fig_size[1] = 0.4155844155
                plt.figure(frameon=False)
                plt.axis('off')
                plt.rcParams["figure.figsize"] = fig_size
                plt.axes().set_aspect('equal')
                plt.pcolormesh(gluon_new,cmap=plt.cm.jet,vmin=-1.0,vmax=+1.0)
                plt.xlim([0,33])
                plt.ylim([0,33])
                plt.xticks([])
                plt.yticks([])

                #cb = plt.colorbar()
                #plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
                #plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
                #cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')#

                #plt.grid(True)
                if index<4000:
                        gimfile ='image_pythia1_2_%002d.jpeg'%index# Name image files as image_01.png
                        plt.savefig(os.path.join(train_gimage_path,gimfile),bbox_inches='tight', pad_inches = 0) # Save image files in the given folder
                        plt.close()
                else:
                        gimfile ='image_pythia1_2_%002d.jpeg'%index# Name image files as image_01.png
                        plt.savefig(os.path.join(test_gimage_path,gimfile),bbox_inches='tight', pad_inches = 0) # Save image files in the given folder
                        plt.close()
                        







# Plotting Average Image
#========================

av_quark=np.mean(quark,axis=2)
print av_quark
mqminpt=np.amin(av_quark)
mqmaxpt=np.amax(av_quark)
plt.rcParams['axes.facecolor'] = 'white'
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 10
plt.figure(frameon=False)
plt.axis('off')

plt.pcolormesh(av_quark,cmap =plt.cm.bwr,vmin=-1.0,vmax=1.0)

#v = np.arange(mqminpt,mqmaxpt)
#cb = plt.colorbar(ticks=v)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
#cb = plt.colorbar()
#plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
#plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
#cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
plt.grid(True)
av2_imqfile ='av_image_pythia3_4_5.jpeg'#herwig3_4_5.jpeg'
plt.savefig(os.path.join(path,av2_imqfile)) # Save image files in the given folder
plt.close()



av_gluon=np.mean(gluon,axis=2)

print av_gluon

mgminpt=np.amin(av_gluon)
mgmaxpt=np.amax(av_gluon)
plt.rcParams['axes.facecolor'] = 'white'
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 10
plt.figure(frameon=False)
plt.axis('off')


plt.pcolormesh(av_gluon,cmap =plt.cm.bwr,vmin=-1.0,vmax=1.0)
#v = np.arange(mgminpt,mgmaxpt)
#cb = plt.colorbar(ticks=v)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
#cb = plt.colorbar()
#plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
#plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
#cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
plt.grid(True)
av2_imgfile ='av_image_pythia1_2.jpeg'#herwig1_2.jpeg'
plt.savefig(os.path.join(path,av2_imgfile)) # Save image files in the given folder
plt.close()


print (time.time()-start_time), "seconds" # Print the time taken for this program to execute this


