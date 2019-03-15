## Prereq for workshop conductors.
### GroundTruth lab
Before start, get emails id of attendees

Add attendees you your private workforce

Create S3 bucket for dataset

Copy Cat and Dog dataset

### Create GroundTruth Job:
Open Sagemaker, go to GroundTruth

Create new labeling job

In source - select s3 bucket where dataset is, create manifest file

give output bucket

next screen, choose private workforce you created

Create two classes - cat and dog

Make sure automatic labeling checkbox is enabled

set one worker for one image

Create job


### Labeling:
Open web link for labeling

Classify images

### Training:
Use following Sagemaker notebook for training

https://github.com/mahendrabairagi/DeeplensWorkshop/blob/master/SageMakerImageClassification/cat-dog-classification-groundtruth.ipynb

### Deploy:
Once training is done - use lab https://github.com/mahendrabairagi/DeeplensWorkshop/blob/master/SageMakerImageClassification/CatAndDogClassification.md
(may needs changes) to deploy model on Deeplens
