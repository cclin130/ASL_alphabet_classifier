# ASL_alphabet_classifier

Final project for McGill AI Society Intro to ML Bootcamp (Winter 2019). 

Training data retrieved from [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet).

## Project description

This ASL Alphabet Classifier project is a web app that classifies American Sign Language gestures contained in an input image. I built
the classification model using Pytorch and the web app's backend using Flask. Furthermore, I retrieved training data from Kaggle,
performed data augmentation with OpenCV and torchvision, and utilized a vgg-11 CNN architecture.

## Running the app

To run the web app, install all packages in requirements.txt. Then, change into the main directory of this repository and run

```
python web_app.py
```

Lastly, open a browser and navigate to your http://localhost:5000.

## Repository organization

This repository contains the scripts used to both train the model and build the web app.

1. deliverables/
	* deliverables submitted to the MAIS Intro to ML Bootcamp organizers
2. model/
	* Data, final model (results), and Python scripts used to train the VGG model (I also attempted an SVM; the scripts for that are in this folder too)
	* CNN_vgg_gpu.py contains the Pytorch Dataloader class used for training
	* get_confusion_matrix.py loads the trained model and tests its performance on the validation set and personal photos
	* train_CNN_gpu.py is a script to train the model on Microsoft Azure's DSVM (with cuda enabled)
3. static/
	* CSS and javascript files for landing page
4. templates/
	* HTML template for landing page
5. predictor.py
	* python class to store model and its methods for web app
6. web_app.py
	* main python script to instantiate Flask server
