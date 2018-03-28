import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import mpl,pyplot
from PIL import Image
import numpy as np
import math
from matplotlib.colors import LogNorm
import  matplotlib.transforms as transforms
import time
#import pp
import os
import sys
from multiprocessing import Process , Queue
import pixel_c
import cv2


start_time = time.time()  # Time to execute the program.
path=''# Path of the program

#PATH OF FILES
#--------------
#1.Quarks
#--------

pathq1='/usr/Pythia/data/'#'/home/k1629656/Jets/Pythia/pythia_gg2qqbar/'
pathq2='/usr/Pythia/data/'#'/home/k1629656/Jets/Pythia/pythia_qq2qq/' 
pathq3='/usr/Pythia/data/'#'/home/k1629656/Jets/Pythia/pythia_qqbar2qqbar/'


#2.Gluons
#---------
pathg1 = '/usr/Pythia/data/'#'/home/k1629656/Jets/Pythia/pythia_gg2gg/'
pathg2='/usr/Pythia/data/'#'/home/k1629656/Jets/Pythia/pythia_qqbar2gg/' 

#3.Images
#--------
#train_qimage_path='/home/k1629656/Jets/Herwig/100-110GeV/color_sample/'#'/usr/Herwig/images_100-110GeV/train/gluons/' #'/home/k1629656/Jets/train/gluons/'#'/usr/Herwig/gluon_jets/images_100-110GeV'#\
#test_qimage_path='/home/k1629656/Jets/Herwig/100-110GeV/color_sample/'#'/usr//Herwig/images_100-110GeV/test/gluons/' #'/home/k1629656/Jets/test/gluons/'


train_qimage_path='/usr/Pythia/colour/images_200-220GeV/train/quarks/'#'/home/k1629656/Jets/herwig_100/train/quarks'#'/home/k1629656/Jets/train/quarks/'#'/usr/Herwig/gluon_jets/images_100-110GeV'#\
test_qimage_path='/usr/Pythia/colour/images_200-220GeV/test/quarks'#'/home/k1629656/Jets/herwig_100/test/quarks'#'/home/k1629656/Jets/test/quarks/'


#train_gimage_path='/home/k1629656/Jets/Herwig/100-110GeV/color_sample/'#'/usr/Herwig/images_100-110GeV/train/gluons/' #'/home/k1629656/Jets/train/gluons/'#'/usr/Herwig/gluon_jets/images_100-110GeV'#\
#test_gimage_path='/home/k1629656/Jets/Herwig/100-110GeV/color_sample/'#'/usr//Herwig/images_100-110GeV/test/gluons/' #'/home/k1629656/Jets/test/gluons/'
train_gimage_path='/usr/Pythia/colour/images_200-220GeV/train/gluons/'
test_gimage_path='/usr/Pythia/colour/images_200-220GeV/test/gluons/'

#FILES
#======
#1.QUARKS
#--------

quarkFile1='pythia_gg2qqbar_40000_200-220_jet_image.dat'  # The generated data from Herwig simulation
quarkFile2='pythia_qq2qq_40000_200-220_jet_image.dat'
quarkFile3='pythia_qqbar2qqbar_40000_200-220_jet_image.dat'
df_quark1 = pd.read_csv(pathq1+quarkFile1,dtype=None,delimiter=",")
df_quark2 = pd.read_csv(pathq2+quarkFile2,dtype=None,delimiter=",")
df_quark3 = pd.read_csv(pathq3+quarkFile3,dtype=None,delimiter=",")
df_quark = pd.concat([df_quark1,df_quark2,df_quark3],ignore_index=True,axis=0)
#df_quark.to_csv('herwig_quark_jet_120000_100-110_jet_image.dat',index = False)

gquark= df_quark.loc[df_quark['ETA']=='@@@'] 
gi_quark = gquark.index # Indices of '@@@'#
#gi_quark=gi_quark[:100000]


#2.GLUONS
#---------

gluonFile1='pythia_gg2gg_60000_200-220_jet_image.dat'  # The generated data from Herwig simulation
gluonFile2 ='pythia_qqbar2gg_60000_200-220_jet_image.dat'
df_gluon1 = pd.read_csv(pathg1+gluonFile1,dtype=None,delimiter=",")
df_gluon2 = pd.read_csv(pathg2+gluonFile2,dtype=None,delimiter=",")

df_gluon = pd.concat([df_gluon1,df_gluon2],ignore_index=True,axis=0)
#df_gluon.to_csv('herwig_gloun_jet_120000_100-110_jet_image.dat',index = False)

ggluon= df_gluon.loc[df_gluon['ETA']=='@@@'] 
gi_gluon = ggluon.index # Indices of '@@@'
#gi_gluon=gi_gluon[:100000]

gquark1= df_quark1.loc[df_quark1['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
gi_quark1 = gquark1.index # Indices of '@@@'##
#gi_quark1= gi_quark1[:40000]
gquark2= df_quark2.loc[df_quark2['ETA']=='@@@'] 
gi_quark2 = gquark2.index
#gi_quark2=gi_quark2[:30000] 
gquark3= df_quark3.loc[df_quark3['ETA']=='@@@'] 
gi_quark3 = gquark3.index 
#gi_quark3=gi_quark3[:30000]

ggluon1= df_gluon1.loc[df_gluon1['ETA']=='@@@']
gi_gluon1 = ggluon1.index
#gi_gluon1=gi_gluon1[:50000] 
ggluon2= df_gluon2.loc[df_gluon2['ETA']=='@@@'] 
gi_gluon2 = ggluon2.index 
#gi_gluon2=gi_gluon2[:50000]


delta = 0.8/33 # length of each pixel
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

quark1_c1= np.zeros((33,33,a1),dtype=float)
quark1_c2= np.zeros((33,33,a1),dtype=float)
quark1_c3= np.zeros((33,33,a1),dtype=float)

quark2_c1= np.zeros((33,33,a2),dtype=float)
quark2_c2= np.zeros((33,33,a2),dtype=float)
quark2_c3= np.zeros((33,33,a2),dtype=float)

quark3_c1= np.zeros((33,33,a3),dtype=float)
quark3_c2= np.zeros((33,33,a3),dtype=float)
quark3_c3= np.zeros((33,33,a3),dtype=float)

quark_image = np.zeros((33,33,3,(a1+a2+a3)),dtype=float)
x_q,y_q,z_q,n_q=np.shape(quark_image)

gluon1_c1= np.zeros((33,33,b1),dtype=float)
gluon1_c2= np.zeros((33,33,b1),dtype=float)
gluon1_c3= np.zeros((33,33,b1),dtype=float)

gluon2_c1= np.zeros((33,33,b2),dtype=float)
gluon2_c2= np.zeros((33,33,b2),dtype=float)
gluon2_c3= np.zeros((33,33,b2),dtype=float)


gluon_image= np.zeros((33,33,3,(b1+b2)),dtype=float)
x_g,y_g,z_g,n_g=np.shape(gluon_image)

print ("Read quark and gluon files")


if __name__=='__main__':
    q1 = Queue()
    p1=Process(target=pixel_c.image_array,args=(gi_quark1,quark1_c1,quark1_c2,quark1_c3,df_quark1,q1))
    p1.start()
    q2 = Queue()
    p2=Process(target=pixel_c.image_array,args=(gi_quark2,quark2_c1,quark2_c2,quark2_c3,df_quark2,q2))
    p2.start()
    q3 = Queue()
    p3=Process(target=pixel_c.image_array,args=(gi_quark3,quark3_c1,quark3_c2,quark3_c3,df_quark3,q3))
    p3.start()         
    g1= Queue()
    p4=Process(target=pixel_c.image_array,args=(gi_gluon1,gluon1_c1,gluon1_c2,gluon1_c3,df_gluon1,g1))
    p4.start()
    g2= Queue()
    p5=Process(target=pixel_c.image_array,args=(gi_gluon2,gluon2_c1,gluon2_c2,gluon2_c3,df_gluon2,g2))
    p5.start()
   

quark1_c1,quark1_c2,quark1_c3=q1.get()

quark1_c1=np.array(quark1_c1).reshape((33,33,a1))
quark1_c2=np.array(quark1_c2).reshape((33,33,a1))
quark1_c3=np.array(quark1_c3).reshape((33,33,a1))


quark2_c1,quark2_c2,quark2_c3=q2.get()

quark2_c1=np.array(quark2_c1).reshape((33,33,a2))
quark2_c2=np.array(quark2_c2).reshape((33,33,a2))
quark2_c3=np.array(quark2_c3).reshape((33,33,a2))


quark3_c1,quark3_c2,quark3_c3=q3.get()

quark3_c1=np.array(quark3_c1).reshape((33,33,a3))
quark3_c2=np.array(quark3_c2).reshape((33,33,a3))
quark3_c3=np.array(quark3_c3).reshape((33,33,a3))

print "Quark image array created!"

#gluon1_c1,gluon1_c2,gluon1_c3=pixel_c.image_array(gi_gluon1,gluon1_c1,gluon1_c2,gluon1_c3,df_gluon1)
#gluon2_c1,gluon2_c2,gluon2_c3=pixel_c.image_array(gi_gluon2,gluon2_c1,gluon2_c2,gluon2_c3,df_gluon2)


gluon1_c1,gluon1_c2,gluon1_c3=g1.get()
gluon2_c1,gluon2_c2,gluon2_c3=g2.get()

gluon1_c1=np.array(gluon1_c1).reshape((33,33,b1))
gluon1_c2=np.array(gluon1_c2).reshape((33,33,b1))
gluon1_c3=np.array(gluon1_c3).reshape((33,33,b1))

gluon2_c1=np.array(gluon2_c1).reshape((33,33,b2))
gluon2_c2=np.array(gluon2_c2).reshape((33,33,b2))
gluon2_c3=np.array(gluon2_c3).reshape((33,33,b2))


quark1 = np.concatenate((quark1_c1,quark1_c2,quark1_c3),axis=0)
quark2 = np.concatenate((quark2_c1,quark2_c2,quark2_c3),axis=0)
quark3 = np.concatenate((quark3_c1,quark3_c2,quark3_c3),axis=0)

gluon1 = np.concatenate((gluon1_c1,gluon1_c2,gluon1_c3),axis=0)
gluon2 = np.concatenate((gluon2_c1,gluon2_c2,gluon2_c3),axis=0)


p1.join()
p2.join()
p3.join()
p4.join()
p5.join()


#QUARKS#



quark01=np.zeros((99,33,(a1+a2+a3)),dtype=float)
quark01[:,:,:a1]=quark1
quark01[:,:,a1:a1+a2]=quark2
quark01[:,:,a2:a2+a3]=quark3

quark =pixel_c.shuffle(quark01)

quark_c1=quark[:33,:,:]
quark_c2=quark[33:66,:,:]
quark_c3=quark[66:,:,:]

gluon01=np.zeros((99,33,(b1+b2)),dtype=float)
gluon01[:,:,:b1]=gluon1
gluon01[:,:,b1:(b1+b2)]=gluon2

gluon=pixel_c.shuffle(gluon01)

gluon_c1=gluon[:33,:,:]
gluon_c2=gluon[33:66,:,:]
gluon_c3=gluon[66:,:,:]


print gluon_c1.shape
print gluon_c2.shape
print gluon_c3.shape

images=np.zeros((99,33,(200000)),dtype=float)
images[:,:,:100000]=quark[:,:,:100000]
images[:,:,100000:]=gluon[:,:,:100000]
images_c1=images[:33,:,:]
images_c2=images[33:66,:,:]
images_c3=images[66:,:,:]

mean_c1=np.mean(images_c1,axis=2)
mean_c2=np.mean(images_c2,axis=2)
mean_c3=np.mean(images_c3,axis=2)

mean_image = np.dstack((mean_c1,mean_c2,mean_c3))

std_c1=np.std(images_c1,axis=2)
std_c2=np.std(images_c2,axis=2)
std_c3=np.std(images_c3,axis=2)

std_image = np.dstack((std_c1,std_c2,std_c3))
images=0
images_c1=images_c2=images_c3=0
mean_c1=mean_c2=mean_c3=std_c1=std_c2=std_c3=0



#Plotting average image before zero centering and standardizing
#==============================================================#
mean_quark_c1= np.mean(quark_c1[:,:,:100000],axis=2)
mean_quark_c2= np.mean(quark_c2[:,:,:100000],axis=2)
mean_quark_c3= np.mean(quark_c3[:,:,:100000],axis=2)
mean_quark= np.dstack((mean_quark_c1,mean_quark_c2,mean_quark_c3))#Average of all images

std_quark_c1= np.std(quark_c1[:,:,:100000],axis=2)
std_quark_c2= np.std(quark_c2[:,:,:100000],axis=2)
std_quark_c3= np.std(quark_c3[:,:,:100000],axis=2)
std_quark= np.dstack((mean_quark_c1,mean_quark_c2,mean_quark_c3))#Std of all images


mean_gluon_c1=np.mean(gluon_c1[:,:,:100000],axis=2)
mean_gluon_c2=np.mean(gluon_c2[:,:,:100000],axis=2)
mean_gluon_c3=np.mean(gluon_c3[:,:,:100000],axis=2)
mean_gluon=np.dstack((mean_gluon_c1,mean_gluon_c2,mean_gluon_c3))

std_gluon_c1=np.std(gluon_c1[:,:,:100000],axis=2)
std_gluon_c2=np.std(gluon_c2[:,:,:100000],axis=2)
std_gluon_c3=np.std(gluon_c3[:,:,:100000],axis=2)
std_gluon=np.dstack((std_gluon_c1,std_gluon_c2,std_gluon_c3))

mean=np.zeros((33,33,3,2),dtype=float)
mean[:,:,:,0]=mean_quark
mean[:,:,:,1]=mean_gluon
#mean_image = np.mean(mean,axis=3)
#mean_image=np.mean(mean_image,axis=2)

std =np.zeros((33,33,3,2),dtype=float)
std[:,:,:,0]=std_quark
std[:,:,:,1]=std_gluon
#std_image=np.std(std,axis=3)
#std_image=np.std(std_image,axis=2)
print "Mean images creating"



av1_imqfile ='av_image_before_step4-5_pythia3_4_5_color.jpeg'
mean_quark = mean_quark*(256./np.amax(mean_quark))
quark_avg=cv2.imwrite(os.path.join(path,av1_imqfile), mean_quark)

av1_imgfile ='av_image_before_step4-5_pythia1_2_color.jpeg'
mean_gluon = mean_gluon*(256./np.amax(mean_gluon))
gluon_avg=cv2.imwrite(os.path.join(path,av1_imgfile), mean_gluon)

print "Creating jet images :) "



#gluon_c1=pixel_c.jet_images(gi_gluon,x_g,y_g,gluon_c1,mean_gluon_c1,std_gluon_c1)
#gluon_c2=pixel_c.jet_images(gi_gluon,x_g,y_g,gluon_c2,mean_gluon_c2,std_gluon_c2)
#gluon_c3=pixel_c.jet_images(gi_gluon,x_g,y_g,gluon_c3,mean_gluon_c3,std_gluon_c3)



for index,j in enumerate(gi_gluon): # For all values and their corresponding indices in g_i
	gluon_c1[:,:,index]=gluon_c1[:,:,index]-mean_image[:,:,0] #Step 4 : Zerocentering
	gluon_c1[:,:,index]=gluon_c1[:,:,index]/(std_image[:,:,0]+r) #Step5: Standardize
	
        gluon_c2[:,:,index]=gluon_c2[:,:,index]-mean_image[:,:,1] 
        gluon_c2[:,:,index]=gluon_c2[:,:,index]/(std_image[:,:,1]+r) 

        gluon_c3[:,:,index]=gluon_c3[:,:,index]-mean_image[:,:,2] 
        gluon_c3[:,:,index]=gluon_c3[:,:,index]/(std_image[:,:,2]+r) 

	gluon_image[:,:,:,index]= np.dstack((gluon_c1[:,:,index],gluon_c2[:,:,index],gluon_c3[:,:,index]))
 	#gluon_image[:,:,:,index]= gluon_image[:,:,:,index]/(std_image+r)gluon_image[:,:,:,index]-mean_image


 


av_gluon=np.mean(gluon_image,axis=3)
av_gluon = av_gluon*(256./np.amax(av_gluon))

av2_imgfile ='av_image_pythia1_2_color.jpeg'
avg_gluon=cv2.imwrite(os.path.join(path,av2_imgfile), av_gluon)

#quark_c1=pixel_c.jet_images(gi_quark,x_q,y_q,quark_c1,mean_quark_c1,std_quark_c1)
#quark_c2=pixel_c.jet_images(gi_quark,x_q,y_q,quark_c2,mean_quark_c2,std_quark_c2)
#quark_c3=pixel_c.jet_images(gi_quark,x_q,y_q,quark_c3,mean_quark_c3,std_quark_c3)


for index,j in enumerate(gi_quark): # For all values and their corresponding indices in g_i
	quark_c1[:,:,index]=quark_c1[:,:,index]-mean_image[:,:,0] #Step 4 : Zerocentering
	quark_c1[:,:,index]=quark_c1[:,:,index]/(std_image[:,:,0]+r) #Step5: Standardize

        quark_c2[:,:,index]=quark_c2[:,:,index]-mean_image[:,:,1] 
        quark_c2[:,:,index]=quark_c2[:,:,index]/(std_image[:,:,1]+r) 

        quark_c3[:,:,index]=quark_c3[:,:,index]-mean_image[:,:,2] 
        quark_c3[:,:,index]=quark_c3[:,:,index]/(std_image[:,:,2]+r) 

	quark_image[:,:,:,index]= np.dstack((quark_c1[:,:,index],quark_c2[:,:,index],quark_c3[:,:,index]))
	#quark_image[:,:,:,index]= quark_image[:,:,:,index]/(std_image+r)quark_image[:,:,:,index]-mean_image#




av_quark=np.mean(quark_image,axis=3)
av_quark=av_quark*(256./np.amax(av_quark))

#print av_quark
av2_imqfile ='av_image_pythia3_4_5_color.jpeg'
avg_gluon=cv2.imwrite((os.path.join(path,av2_imqfile)),av_quark) # Save image files in the given folder

quark_name='image_pythia3_4_5_color'
gluon_name='image_pythia1_2_color'



#if __name__=='__main__':
    
p1.join()
p2.join()
p3.join()
p4.join()
p5.join()
p6=Process(target=pixel_c.plotting,args=(gi_gluon,gluon_image,gluon_name,train_gimage_path,test_gimage_path))
p6.start()
p7=Process(target=pixel_c.plotting,args=(gi_quark,quark_image,quark_name,train_qimage_path,test_qimage_path))
p7.start()
p6.join()
p7.join()


#pixel.plotting(gi_gluon,gluon,gluon_name,train_gimage_path)

plt.close()

print (time.time()-start_time), "seconds" 
