# Jet-Images
To calssify gluon and quark jets generated from Pythia and Herwig

DATASET
========
The processes considered here are :
 i) gg->gg
 ii) qqbar->gg 
 iii) qq->qq
 iv) gg->qqbar
 v) qqbar->qqbar
 
 (i) & (ii) are combined to generate gluon jet images and the rest to generate quark jet images.The dataset has three columns eta, phi and pt. The first row in each dataset is the total eta, phi and pt of the jet and the remaining are eta, phi and for constituents of the jet.
 
 The name of the dataset implies : eventgenerator_process_number-of-Jets_pt-range_jet_image.dat
 Here I have included only jets generated from Herwig in pt range 100-110 GeV.
 
 CODES
 =====
 All codes are in python
 1. jet_images_sample_code.py
 ------------------------------
 The code reades data from the dataset and generate the pre-processed images.
 The pre-processing steps are described in the paper Deep learning in color: towards automated quark/gluon jet discrimination (https://arxiv.org/pdf/1612.01551.pdf) section 3.1.
 
 2. pixel_jet_images.py
 ----------------------
 I have written the some of the preprocessing steps in this code as function. Steps 1,2 and 3
 
 3. jet_images_ml.py
 -------------------
 The machine learning part is included in this code the deep CNN is created to classify the images (https://arxiv.org/pdf/1612.01551.pdf) section 3.2.
 
 4. dataset.py
 -------------
 To read the test and train images
 
 I modified train.py https://github.com/sankit1/cv-tricks.com/tree/master/Tensorflow-tutorials/tutorial-2-image-classifier to generate the 3 and used the same code https://github.com/sankit1/cv-tricks.com/tree/master/Tensorflow-tutorials/tutorial-2-image-classifier as 4. Thank you Ankit Sachen! :) Acknowledged!!
 
 IMAGES
 ======
 av_image_before_step4-5_herwig1_2_.jpeg
 ---------------------------------------
 Average gluon jet image before step 4 & 5 generated from herwig. This is an average image for 100000 images. For the dataset given it looks different.
 
 av_image_before_step4-5_herwig3_4_5_.jpeg
 ---------------------------------------
 Average quark jet image before step 4 & 5 generated from herwig. This is an average image for 100000 images. For the dataset given it looks different.
 
 av_image_herwig1_2.jpeg
 ------------------------
 Average gluon jet image after step 4 & 5 (all preprocessing) generated from herwig. This is an average image for 100000 images. For the dataset given it looks different.
 
 av_image_herwig3_4_5.jpeg
 ------------------------
 Average quark jet image after step 4 & 5 (all preprocessing) generated from herwig. This is an average image for 100000 images. For the dataset given it looks different.
 
 jet_images_sample.zip
 ----------------------
 The 240 images you get when you run the jet_images_sample_code.py
 
