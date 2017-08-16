"""
The code reades data from the dataset and generate the pre-processed images.
The pre-processing steps are described in the paper Deep learning in color: towards automated quark/gluon
jet discrimination (https://arxiv.org/pdf/1612.01551.pdf) section 3.1.

"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from matplotlib.colors import LogNorm
import  matplotlib.transforms as transforms
import time
import pp
import os
import sys
from multiprocessing import Process , Queue
import pixel_jet_images



start_time = time.time()  # Time to execute the program.
path=''# Path of the program

#PATH OF FILES
#--------------
#1.Quarks
#--------

pathq1=''
pathq2=''
pathq3=''


#2.Gluons
#---------
pathg1 =''
pathg2=''

#3.Images
#--------
train_qimage_path='/home/k1629656/Jets/jet_images_sample/train/quarks/'
test_qimage_path='/home/k1629656/Jets/jet_images_sample/test/quarks/'


train_gimage_path='/home/k1629656/Jets/jet_images_sample/train/gluons/' 
test_gimage_path='/home/k1629656/Jets/jet_images_sample/test/gluons/'


#FILES
#======
#1.QUARKS
#--------

quarkFile1='herwig_gg2qqbar_40_100-110_jet_image.dat'  # The generated data f\rom Herwig simulation
quarkFile2='herwig_qq2qq_40_100-110_jet_image.dat'
quarkFile3='herwig_qqbar2qqbar_40_100-110_jet_image.dat'
df_quark1 = pd.read_csv(pathq1+quarkFile1,dtype=None,delimiter=",")
df_quark2 = pd.read_csv(pathq2+quarkFile2,dtype=None,delimiter=",")
df_quark3 = pd.read_csv(pathq3+quarkFile3,dtype=None,delimiter=",")
df_quark = pd.concat([df_quark1,df_quark2,df_quark3],ignore_index=True,axis=0)
df_quark.to_csv('herwig_quark_jet_120_100-110_jet_image.dat',index = False)
gquark= df_quark.loc[df_quark['ETA']=='@@@'] 
gi_quark = gquark.index # Indices of '@@@'#



#2.GLUONS
#---------

gluonFile1='herwig_gg2gg_60_100-110_jet_image.dat'  # The generated data from Herwig simulation
gluonFile2 ='herwig_qqbar2gg_60_100-110_jet_image.dat'
df_gluon1 = pd.read_csv(pathg1+gluonFile1,dtype=None,delimiter=",")
df_gluon2 = pd.read_csv(pathg2+gluonFile2,dtype=None,delimiter=",")
df_gluon = pd.concat([df_gluon1,df_gluon2],ignore_index=True,axis=0)
df_gluon.to_csv('herwig_gloun_jet_120_100-110_jet_image.dat',index = False)
ggluon= df_gluon.loc[df_gluon['ETA']=='@@@'] 
gi_gluon = ggluon.index # Indices of '@@@'#

gquark1= df_quark1.loc[df_quark1['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
gi_quark1 = gquark1.index # Indices of '@@@'#
gquark2= df_quark2.loc[df_quark2['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
gi_quark2 = gquark2.index # Indices of '@@@'#
gquark3= df_quark3.loc[df_quark3['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
gi_quark3 = gquark3.index # Indices of '@@@'#


ggluon1= df_gluon1.loc[df_gluon1['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
gi_gluon1 = ggluon1.index # Indices of '@@@'
ggluon2= df_gluon2.loc[df_gluon2['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
gi_gluon2 = ggluon2.index # Indices of '@@@'




delta = 0.8/33 # length of each pixel_jet_images
boxmax=0.4
r = 0.00001
total_pT = 0.00


quark_new=np.zeros((33,33))
gluon_new = np.zeros((33,33))


a1=np.size(gi_quark1)
a2=np.size(gi_quark2)
a3=np.size(gi_quark3)
b1=np.size(gi_gluon1)
b2=np.size(gi_gluon2)

quark1= np.zeros((33,33,a1),dtype=float)
quark2= np.zeros((33,33,a2),dtype=float)
quark3= np.zeros((33,33,a3),dtype=float)
gluon1= np.zeros((33,33,b1),dtype=float)
gluon2= np.zeros((33,33,b2),dtype=float)

print ("Read quark and gluon files")


#PRE-PROCESSING STEPS
#=====================

if __name__=='__main__':
    q1 = Queue()#
    p1=Process(target=pixel_jet_images.image_array,args=(gi_quark1,quark1,df_quark1,q1))
    p1.start()
    q2 = Queue()#
    p2=Process(target=pixel_jet_images.image_array,args=(gi_quark2,quark2,df_quark2,q2))
    p2.start()
    q3 = Queue()#
    p3=Process(target=pixel_jet_images.image_array,args=(gi_quark3,quark3,df_quark3,q3))
    p3.start()         
    g1= Queue()
    p4=Process(target=pixel_jet_images.image_array,args=(gi_gluon1,gluon1,df_gluon1,g1))
    p4.start()
    g2= Queue()
    p5=Process(target=pixel_jet_images.image_array,args=(gi_gluon2,gluon2,df_gluon2,g2))
    p5.start()
   

quark1=q1.get()
quark1=np.array(quark1).reshape((33,33,a1))
quark2=q2.get()
quark2=np.array(quark2).reshape((33,33,a2))
quark3=q3.get()
quark3=np.array(quark3).reshape((33,33,a3))
print "Quark image array created!"

gluon1=g1.get()
gluon1=np.array(gluon1).reshape((33,33,b1))
gluon2=g2.get()
gluon2=np.array(gluon2).reshape((33,33,b2))
print "Gluon image array created!"


p1.join()
p2.join()
p3.join()
p4.join()
p5.join()


quark01=np.zeros((33,33,(a1+a2+a3)),dtype=float)
quark01[:,:,:a1]=quark1
quark01[:,:,a1:a1+a2]=quark2
quark01[:,:,a2:a2+a3]=quark3

quark =pixel_jet_images.shuffle(quark01)


gluon01=np.zeros((33,33,(b1+b2)),dtype=float)
gluon01[:,:,:b1]=gluon1
gluon01[:,:,b1:b1+b2]=gluon2
x,y,z=np.shape(gluon01)

gluon=pixel_jet_images.shuffle(gluon01)


#Plotting average image before zero centering and standardizing
#==============================================================#
mean_quark= np.mean(quark[:,:,:100],axis=2)#Average of all images
std_quark= np.std(quark[:,:,:100],axis=2)#Std of all images


mquark = pd.DataFrame(mean_quark)
mquark.to_csv("mean_quark_gif.txt")

mean_gluon=np.mean(gluon[:,:,:100],axis=2)
std_gluon=np.std(gluon[:,:,:100],axis=2)

mgluons = pd.DataFrame(mean_gluon)
mgluons.to_csv("mean_gluons_gif.txt")

mean=np.zeros((33,33,2),dtype=float)
mean[:,:,0]=mean_quark
mean[:,:,1]=mean_gluon

std=np.zeros((33,33,2),dtype=float)
std[:,:,0]=std_quark
std[:,:,1]=std_gluon


mean_image=np.mean(mean,axis=2)
std_image=np.std(std,axis=2)

print "Mean images creating"



plt.pcolormesh(mean_quark,cmap =plt.cm.jet)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
v=np.arange(-1.0,1.0)
cb = plt.colorbar(ticks=v)
plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
plt.grid(True)
av1_imqfile ='av_image_before_step4-5_herwig3_4_5_.jpeg'
plt.savefig(os.path.join(path,av1_imqfile)) # Save image files in the given folder
plt.close()





plt.pcolormesh(mean_gluon,cmap =plt.cm.jet)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
v=np.arange(-1.0,1.0)
cb = plt.colorbar(ticks=v)
plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
plt.grid(True)
av1_imgfile ='av_image_before_step4-5_herwig1_2_.jpeg'
plt.savefig(os.path.join(path,av1_imgfile)) # Save image files in the given folder
plt.close()

print "Creating jet images :) "

#Plotting average image after zero centering and standardizing
#==============================================================#

for index,j in enumerate(gi_gluon): # For all values and their corresponding indices in g_i
        if gi_gluon[index-1]+1<gi_gluon[index]:
                for k in range(0,x):
                        for j in range(0,y):
                                gluon[j,k,index]=gluon[j,k,index]-mean_image[j,k] #Step 4 : Zerocentering
                                gluon[j,k,index]=gluon[j,k,index]/(std_image[j,k]+r) #Step5: Standardize


av_gluon=np.mean(gluon,axis=2)
plt.rcParams['axes.facecolor'] = 'white'
plt.axes().set_aspect('equal')
fig_size = plt.rcParams["figure.figsize"]
plt.figure(frameon=False)
plt.axis('off')
plt.pcolormesh(av_gluon,cmap =plt.cm.bwr,vmin=-1.0,vmax=1.0)
v = np.arange(-1.0,1.0)
cb = plt.colorbar(ticks=v)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
plt.grid(True)
av2_imgfile ='av_image_herwig1_2.jpeg'#herwig1_2.jpeg'
plt.savefig(os.path.join(path,av2_imgfile)) # Save image files in the given folder
plt.close()







for index,j in enumerate(gi_quark): # For all values and their corresponding indices in g_i
        if gi_quark[index-1]+1<gi_quark[index]:
                for k in range(0,x):
                        for j in range(0,y):
                                quark[j,k,index]=quark[j,k,index]-mean_image[j,k] #Step 4 : Zerocentering
                                quark[j,k,index]=quark[j,k,index]/(std_image[j,k]+r) #Step5: Standardize




av_quark=np.mean(quark,axis=2)
print av_quark
mqminpt=np.amin(av_quark)
mqmaxpt=np.amax(av_quark)
plt.rcParams['axes.facecolor'] = 'white'
plt.axes().set_aspect('equal')
fig_size = plt.rcParams["figure.figsize"]

plt.figure(frameon=False)
plt.axis('off')

plt.pcolormesh(av_quark,cmap =plt.cm.bwr,vmin=-1.0,vmax=1.0)

v = np.arange(-1.0,1.0)
cb = plt.colorbar(ticks=v)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
plt.grid(True)
av2_imqfile ='av_image_herwig3_4_5.jpeg'
plt.savefig(os.path.join(path,av2_imqfile)) # Save image files in the given folder
plt.close()                        


#Plotting Pre-processed Jet-Images
#===================================

quark_name='image_herwig3_4_5'
gluon_name='image_herwig1_2'

p6=Process(target=pixel_jet_images.plotting,args=(gi_gluon,gluon,gluon_name,train_gimage_path,test_gimage_path))
p6.start()
p7=Process(target=pixel_jet_images.plotting,args=(gi_quark,quark,quark_name,train_qimage_path,test_qimage_path))
p7.start()
p6.join()
p7.join()


plt.close()


print (time.time()-start_time), "seconds" 
