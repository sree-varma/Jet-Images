# Jet-Images
To classify gluon and quark jets generated from Pythia and Herwig

## Setup
We will use `virtualenv` to manage python dependencies, to get it run
```bash
pip install virtualenv
```

Create a new `virtualenv` with 
 ```bash
virtualenv .
```
activate it with 

```bash
. bin/activate
```
install requirements with

```bash
pip install -r requirements.txt
```

## Usage

To starting training/validation run 
```bash
python jet_images_ml.py
```

You can monitor progress with Tensorboard using
```bash
tensorboard --logdir path/to/logdir
```

To evaluate the resulting model / extract features run

```bash
python evaluate.py
```
## Appendix

### Dataset
The processes considered here are :
 i) gg->gg
 ii) qqbar->gg 
 iii) qq->qq
 iv) gg->qqbar
 v) qqbar->qqbar
 
 (i) & (ii) are combined to generate gluon jet images and the rest to generate quark jet images.The dataset has three columns eta, phi and pt. The first row in each dataset is the total eta, phi and pt of the jet and the remaining are eta, phi and for constituents of the jet.
 
 The name of the dataset implies : eventgenerator_process_number-of-Jets_pt-range_jet_image.dat
 Here I have included only jets generated from Herwig in pt range 100-110 GeV.
 
 
 ### IMAGES
 
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
 
