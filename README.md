# cnn-dog-breed-classifier
PyTorch implementation of a dog breed classifier using convolutional neural nets. Part of __[Udacity's Deep Learning Nanodegree program](https://eu.udacity.com/course/deep-learning-nanodegree--nd101?gclid=Cj0KCQiAheXiBRD-ARIsAODSpWMPNTRMr6ecpZ3sUWoLF5I45JYfwFsngcOqfJFUxYT_TnvSsXaecCMaAuFfEALw_wcB)__. 

The dataset contains 8352 images in total, divided across 133 breeds. On average, there are 50 images per breed, with a minimum of 26 for the Norwegian buhund and a maximum of 77 for the Alaskan malamute.


# Datasets
- Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in this project's home directory, at the location `/dogImages`. 
- Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the home diretcory, at location `/lfw`.  

# Contents
The repository consists of the following three main files:
- **from_scratch.ipynb**: Here I analyze in depth the influence of different augmentation techniques and architectural improvements on the performance of a dog breed classifier trained from scratch (i.e. no transfer learning). The final model achieves an accuracy of about 61% on the test set (random chance is < 1%). This notebook also questions the usual choice of replacing a fully connected layer with global average pooling. I show that for this dataset global **max** pooling significantly outperforms global average pooling and I argue that it is a better fit in the philosophy of convolutional networks. 
- **misc.py**: Most of the imports, function and class definitions used in the above. 
- **dog_app.ipynb**: The Udacity project. It consists of the following steps:
    - Assessing the performance of a Haar cascade classifier for human face detection. 
    - Assessing the performance of six different pretrained networks for dog detection, irrespective of the breed. 
    - Creating an architecture for classifying dog breeds from scratch. Here I use the results of the **from_scratch** notebook. 
    - Using transfer learning for classifying dog breeds. 
    - Putting it all together. 
- **fix_truncated_images.ipynb**: One of the training images is truncated. More info and fix in this notebook. 
    
# Usage
The easiest way to run everything is to create an environment from `Requirements.txt` with the command:<br>
`conda create -n <environment_name_here> --file Requirements.txt`