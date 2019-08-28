![IronHack Logo](https://s3-eu-west-1.amazonaws.com/ih-materials/uploads/upload_d5c5793015fec3be28a63c4fa3dd4d55.png)

# Convolutional Neural Network to detect Pneumonia...

Final project - Data Analytics Iron Hack

## Pneumonia detector in chest X-Ray images

Pneumonia is an infection that inflames the air sacs in one or both lungs. 
The air sacs may be filled with fluid, causing cough, fever and difficulties 
on breathing. 

<img src="https://github.com/jmolins89/final-project/blob/master/output/concepto-neumonia_98396-172.jpg" 
alt="pneumonia" style="width:100px;height:100px;margin-left: auto;margin-right: auto"/>

*In the previous image you can see a lung with pneumonia.*

If the doctor thinks you have pneumonia, he/she may recommend one or more of 
the following tests:

* **Chest x ray** to look for inflammation in your lungs. This test is the best 
to diagnosing pneumonia.
* **Blood tests** to see if your immune system is actively fighting an infection.
* **Blood culture** to find out whether you have a bacterial infection that has 
spread to your bloodstream

The pneumonia detection by chest x ray sometimes is difficult also to doctors 
who have studied long time. The new technologies could be helpful to detect
pneumonia because it doesn't depend on human factors, it will be automatic.

With machine learning we can teach the computer to detect pneumonia, specifically
creating a **convolutional neural network**, which is going to learn how to detect 
a pneumonia in a chest x ray image.

I've used an [image database from Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
to train the model.

The images of the database are like the following:

![alt text](https://github.com/jmolins89/final-project/blob/master/output/example-images.png)

**We can't differentiate whether or not there is pneumonia** if we have not
studied about chest X-Ray images reading. It doesn't matter, because the neural
network is going to do this for us. It is going to receive the images and 
the label if it's a pneumonia or not, and the **NN is going to learn when an
image has pneumonia or not**.

The first step to do this is studying the images of the database, and 

In the following graph we can observe that **the data is unbalanced**, because we have more Pneumonia cases than Normal.

We have to **rebalance** the data to train better a neural network.

![alt text](https://github.com/jmolins89/final-project/blob/master/output/plotting-unbalanced-dataset.png)

We are going to **generate random images of Normal cases**.

![alt text](https://github.com/jmolins89/final-project/blob/master/output/example-different-way-to-duplicate-images.png)


