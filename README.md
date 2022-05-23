# Project3: Machine Learning models for real-world use
Project for CS_6364 Machine Learning

## Description

This project will take the CNN project (project_2) and complete the following analyses:

* Generate test cases and datasets
* Discuss what biases may be present in the data and the modeling
* Try to explain what the model has learned using saliency maps

For dataset information, please see: [CNNs and Transfer Learning](https://github.com/N3l1magak/CNNs-and-Transfer-Learning)

## Getting Started
### Part 1: Generating test cases

**Generate a dataset of just three images, one for each class of the dogs**

![image](https://user-images.githubusercontent.com/105015948/169915872-7ad224fa-89b5-4aa4-befb-718327ffa8eb.png)

**Test the model**

![image](https://user-images.githubusercontent.com/105015948/169915939-ca5b0237-fb51-4be1-b3c5-1ec6ae791678.png)

As shown above, the predictions are all correct.

**Generate three datasets of inputs, where each has only two of the classes**

I suppose that the datasets of Great_Dane vs Norfolk_terrier and Great_Dane vs Norwich_terrier would be have a better result when using the re-trained model (since more distinctive between these two breeds) comparing to the datasets of Norfolk_terrier and Norwich_terrier (which look almost the same).

![image](https://user-images.githubusercontent.com/105015948/169916139-245cbf82-0246-440e-9c44-51a62cf8e16e.png)

![image](https://user-images.githubusercontent.com/105015948/169916313-7f4b827f-3e95-4b1b-b0ce-ba0f609d93c3.png)

![image](https://user-images.githubusercontent.com/105015948/169916224-864ed3de-9ccb-479e-b3b8-6b8455c702ec.png)

![image](https://user-images.githubusercontent.com/105015948/169916241-e5ab9c97-05b2-4d89-ab83-40c0879c0295.png)

![image](https://user-images.githubusercontent.com/105015948/169916257-e18d67ab-59ec-42ea-b50f-e2d942d2ed67.png)

The model has great performance distinguishing Great_Dane vs Norfolk_terrier and Great_Dane vs Norwich_terrier (100% acc), but for Norwich_terrier vs Norwich_terrier, the model has trouble distinguish the breed (approx. 80% acc). This is exactly as expected, since it is less distinguishable between Norfolk_terrier and Norwich_terrier.

**Generate a dataset from original dataset where 20% of the classes in one class are mis-labelled as the remaining two classes**

I suppose the overall performance of the model will decrease, since the dataset itself has error in it.

![image](https://user-images.githubusercontent.com/105015948/169916479-d404fd06-545c-405b-b4c1-9fa72a9c1954.png)

![image](https://user-images.githubusercontent.com/105015948/169916502-64840d56-6955-44d5-b8e1-6c2180cc4ebc.png)

![image](https://user-images.githubusercontent.com/105015948/169916523-443aaf75-6baa-428b-a93a-c5e62973ade6.png)

As shown above, the result performance greatly reduced due to the error in the dataset.

### Part 2: Biases in the modeling

**Aspects of the image might be influencing the decision-making of the model**

* the background of the images (e.g. color)
* The proportion of dogs in the picture
* Some of the images have people in it
* The exposure of the images (e.g. too dark/too bright)
* The angle of the dog in the images

**Calculate the "average image" across all pixels of each of the classes in the training dataset**

![image](https://user-images.githubusercontent.com/105015948/169916786-63805ca3-c625-4909-af9b-a4069b210cd3.png)

The above result shows the average R,G,B for each image of the first 10 images in each class, as it shows, the result is non-consistent with other images in the same class. So there is a lot of noise.

**Is the data biased in any way that could impact the results?**

Yes, for example, the Norfolk_terrier and Norwich_terrier seems including dogs from a large variaty of age, whereas Great_Dane are mostly adult. Another bias could be due to the size of the dog, as Great_Dane is larger than the other two breeds, the background of the images could have more details (more different RGB pixels) than the other two breeds. These bias would mislead the model to learn false patterns.

I would implement image segmentation to drop out the background, and also adjust the exposure of the image to further reduce the impact of bias on the model learning.

### Part 3: Model uncertainty and explainability

**Re-train model, but this time don't use a pre-trained version**

![image](https://user-images.githubusercontent.com/105015948/169916972-4f6fa2a2-1e51-4248-9174-c9a34d430391.png)

![image](https://user-images.githubusercontent.com/105015948/169916999-64b240e8-40f6-4164-ad9a-32f3e1a35156.png)

![image](https://user-images.githubusercontent.com/105015948/169917028-21222d14-b293-4b6d-8ded-5bf66f5bff17.png)

The model is learning all the features from beginning, so the learning time is much longer than the pre-trained one. In fact, my model has 50 epochs, and the learning result performance is still much lower than the pre-trained one with 10 epochs (which even overfitted).

When building a model on just the head of the dog, the model should be better than the current one. As the image is focusing on the head, there would be less noise for the model, so that the feature learned by it will less likely to be false.

When crop the dog out of the image, the model is learning merely on the backgroung instead of the target, so the model result should be much worse compare to the current one.

**Implementing saliency maps for all images**

![image](https://user-images.githubusercontent.com/105015948/169917246-82be1912-ab72-427b-bef0-c0463b2831d3.png)

![image](https://user-images.githubusercontent.com/105015948/169917273-daebd207-ed5a-4fd7-a526-b32ffd763466.png)

![image](https://user-images.githubusercontent.com/105015948/169917300-37d7bb5f-4efc-4c7d-a257-7d1df2124a2b.png)

![image](https://user-images.githubusercontent.com/105015948/169917324-a45523e1-967a-433a-a0b6-2edfda5b60c5.png)

From the saliency map, the hot spot is mostly around the dog (the head, the face, the body, etc.), so the model is learning features from the dog itself, which is correct. But there are several images which the hot spot is on the background, so the model does learn incorrect features as well.

### Before executing ipynb

the following libraries are needed:
* pandas
* Seaborn
* numpy
* matplotlib.pyplot
* sklearn
* imblearn

## Author

Reynolds_Z @ 2022

