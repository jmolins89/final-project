![IronHack Logo](https://s3-eu-west-1.amazonaws.com/ih-materials/uploads/upload_d5c5793015fec3be28a63c4fa3dd4d55.png)

# Convolutional Neural Network to detect Pneumonia...

Final project - Data Analytics Iron Hack

## Pneumonia detector in chest X-Ray images

Pneumonia is an infection that inflames the air sacs in one or both lungs. 
The air sacs may be filled with fluid, causing cough, fever and difficulties 
on breathing. 

<p align="center"><img src="https://github.com/jmolins89/final-project/blob/master/output/concepto-neumonia_98396-172.jpg" width="40%" height="40%"></p>

*In the previous image you can see a the theoretical difference between a lung with or without pneumonia.*

If the doctor thinks you have pneumonia, he/she may recommend one or more of 
the following tests:

* **Chest x ray** to look for inflammation in your lungs. This test is the best 
to diagnosing pneumonia.
* **Blood tests** to see if your immune system is actively fighting an infection.
* **Blood culture** to find out whether you have a bacterial infection that has 
spread to your bloodstream

The pneumonia detection by chest x ray sometimes is difficult also to doctors 
who have studied long time. The new technologies could be helpful to detect
pneumonia because it doesn't depend on human factors: it will be automatic.

With machine learning we can teach the computer to detect pneumonia, specifically
creating a **convolutional neural network**, which is going to learn how to detect 
a pneumonia in a chest x ray image. The network is going to receive a lot of images
of chest x ray and they are going to be studied by the NN in the way to learn how 
to detect pneumonia.

I've used an [image database from Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
to train the model.

The images of the database are like the following:

![alt text](https://github.com/jmolins89/final-project/blob/master/output/example-images.png)

**We can't differentiate whether or not there is pneumonia** if we have not
studied about chest X-Ray images reading. It doesn't matter, because the neural
network is going to do this for us. It is going to receive the images and 
the label if it's a pneumonia or not, and the **NN is going to learn when an
image has pneumonia or not**.

The image database is compounded by three folders of images: train, test and
validation, and each of this three folders is compounded by two: NORMAL and PNEUMONIA,
differentiating images with pneumonia and without pneumonia.

In the following graph we can observe that **the data is unbalanced**, 
because we have more Pneumonia images than Normal. This is going to work 
against our NN, because it is going to learn very good to detect a pneumonia but
it isn't going to detect good the cases without pneumonia.

![alt text](https://github.com/jmolins89/final-project/blob/master/output/plotting-unbalanced-dataset.png)

We have to **rebalance** the data in the way to improve the detection of 
normal cases. We are going to **generate random images of Normal cases**.
In this version i'm going to generate images manually, by doing zoom on 
images with the Normal label, as you can see in the following image.

![alt text](https://github.com/jmolins89/final-project/blob/master/output/generating-third-images-zoom.png)

In the following graph you can observe the final image distribution. Now 
the image database is balanced.

![alt text](https://github.com/jmolins89/final-project/blob/master/output/final-not-unbalanced-data-distribution.png)

We can start training the Convolutional Neural Network.

In the following images you can see the results of the final model trained:

<p align="center"><img src="https://github.com/jmolins89/final-project/blob/master/output/accuracy-loss-evolution-final-model.png" width="40%"/> <img src="https://github.com/jmolins89/final-project/blob/master/output/loss-evolution-final-model.png" width="40%"/></p>

<p align="center" display:inline-block><img src="https://github.com/jmolins89/final-project/blob/master/output/roc-auc-final-model.png" width="40%"/> <img src="https://github.com/jmolins89/final-project/blob/master/output/confusion-matrix-final-model.png" width="40%"/></p>


## Next steps:



