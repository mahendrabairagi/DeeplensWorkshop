# DeepLens-workshops

In this workshop you will learn how to build a sentiment analysis project for your DeepLens.

In this project you will learn to build a deep learning model to identify and analyze the sentiments of your audience

## In this workshop you will learn the following:

1. How to build and train a face detection model in SageMaker
2. Modify the DeepLens inference lambda function to upload cropped faces to S3
3. Deploy the inference lambda function and face detection model to DeepLens
4. Create a lambda function to trigger Rekognition to identify emotions
5. Create a DynamoDB table to store the recognized emotions
6. Analyze using CloudWatch

![image](https://user-images.githubusercontent.com/11222214/37996605-1ba4be34-31cd-11e8-9e25-ba3a1cdbc9db.png)

The workshop consists of 4 hands-on lab sessions:

# Hands-on Lab 1: Register and configure your DeepLens device (You can skip this lab if device registration is already complete)

Follow instructions here: [Registration and Deployment lab](https://github.com/mahendrabairagi/DeeplensWorkshop/tree/master/Registration%20and%20project%20deployment)


# Hands-on Lab 2: Build and train a face detection model in SageMaker

In this lab, you will build and train a face detection model. Follow instructions here: [SageMaker lab](https://github.com/mahendrabairagi/DeeplensWorkshop/tree/master/SageMaker%20lab)

# Hands-on Lab 3: Build a project to detect faces and send the cropped faces to S3 bucket

#### IAM Roles:

First, we need to add S3 permissions to the DeepLens Lambda role so the lambda on the device can call Put Object into the bucket of interest.

Go to [IAM Console](https://console.aws.amazon.com/iam/home?region=us-east-1#/home)

Choose Roles and look up AWSDeepLensGreenGrassGroupRole

Click on the role, and click Attach Policy

Search for AmazonS3FullAccess and choose the policy by checking the box and click on Attach Policy

#### Create Bucket:

We need to create an S3 bucket that we can upload faces to.

Go to [AWS Management console](https://console.aws.amazon.com/console/home?region=us-east-1) and search for S3

Choose 'Create bucket'

Name your bucket : face-detection-your-name

Click on Create 

#### Create Inference lambda function:

Go to [AWS Management console](https://console.aws.amazon.com/console/home?region=us-east-1) and search for Lambda

Click 'Create function'

Choose 'Blueprints'

In the search bar, type “greengrass-hello-world” and hit Enter

Choose the python blueprint and click Configure

Name the function: DeepLens-sentiment-your-name
Role: Choose an existing role
Existing Role: AWSDeepLensLambdaRole

Click Create Function
Replace the default script with the [inference script](https://github.com/mahendrabairagi/DeeplensWorkshop/blob/master/Inference%20Lambda/inference-lambda.py)

You can select the inference script, by selecting Raw in the Github page and choosing the script using ctrl+A/ cmd+A . Copy the script and paste it into the lambda function (make sure you delete the default code).

In the script, you will have to provide the name for your S3 bucket. Insert your bucket name in the code below

![code bucket](https://user-images.githubusercontent.com/11222214/38719807-b46169fa-3ea8-11e8-8ff2-69af5455ede7.jpg)

Click Save

Under the “Actions” drop-down menu, Click “Publish new version” and publish.

Note:It is important that you publish the lambda  function, else you cannot access it from DeepLens console.

#### Deploy project:

**Step 1- Create Project**

The AWS DeepLens console should open on the Projects screen, select Create new project on the top right (if you don’t see the project list view, click on the hamburger menu on the left and select Projects)

![create project](https://user-images.githubusercontent.com/11222214/38657905-82207e44-3dd7-11e8-83ef-52049e229e33.JPG)

Choose a blank template and scroll down the screen to select Next

Provide a name for your project: face-detection-your-name

Click on Add Models and choose face_detection (One you created during SageMaker Lab)

Click on Add function and choose the lambda function you just created: Deeplens-sentiment-your-name

Click Create

**Step 2- Deploy to device**
In this step, you will deploy the Face detection project to your AWS DeepLens device.

Select the project you just created from the list by choosing the radio button


Select Deploy to device.


![choose project-edited-just picture](https://user-images.githubusercontent.com/11222214/38657988-eb9d98b6-3dd7-11e8-8c94-7273fcfa6e1b.jpg)

On the Target device screen, choose your device from the list, and select **Review.**

![target device](https://user-images.githubusercontent.com/11222214/38658011-088f81d2-3dd8-11e8-972a-9342b7b3e291.JPG)

Select Deploy.

![review deploy](https://user-images.githubusercontent.com/11222214/38658032-223db2e8-3dd8-11e8-9bdf-04779cd0e0e6.JPG)

On the AWS DeepLens console, you can track the progress of the deployment. It can take a few minutes to transfer a large model file to the device. Once the project is downloaded, you will see a success message displayed and the banner color will change from blue to green.

**Confirmation/ verification**

You will find your cropped faces uplaod to your S3 bucket.


# Hands-on Lab 4: Identify emotions

**Step 1- Create DynamoDB table**

Go to [AWS Management console](https://console.aws.amazon.com/console/home?region=us-east-1) and search for Dynamo

Click on Create Table.

Name of the table: recognize-emotions-your-name
Primary key: s3key

Click on Create. This will create a table in your DynamoDB.

**Step 2- Create a role for cloud lambda function**

Go to [AWS Management console](https://console.aws.amazon.com/console/home?region=us-east-1) and search for IAM

Choose 'Create Role'

Select “AWS Service”

Select “Lambda” and choose "Next:Permissions"

Attach the following policies: 

* AmazonDynamoDBFullAcces
* AmazonS3FullAccess
* AmazonRekognitionFullAccess
* CloudWatchFullAccess

Click Next

Provide a name for the role: rekognizeEmotions

Choose 'Create role'


**Step 3- Create a lambda function that runs in the cloud**

The inference lambda function that you deployed earlier will upload the cropped faces to your S3. On S3 upload, this new lambda function gets triggered and runs the Rekognize Emotions API by integrating with Amazon Rekognition. 

Go to [AWS Management console](https://console.aws.amazon.com/console/home?region=us-east-1) and search for Lambda

Click 'Create function'

Choose 'Author from scratch'

Name the function: recognize-emotion-your-name.  
Runtime: Choose Python 2.7
Role: Choose an existing role
Existing role: rekognizeEmotions

Choose Create function

Replace the default script with the script in [recognize-emotions.py](https://github.com/mahendrabairagi/DeeplensWorkshop/blob/master/Integrate%20with%20Rekognition/rekognize-emotions.py). You can select the script by selecting Raw in the Github page and choosing the script using ctrl+A/ cmd+A . Copy the script and paste it into the lambda function (make sure you delete the default code).

Make sure you enter the table name you created earlier in the section highlighted below:

![dynamodb](https://user-images.githubusercontent.com/11222214/38838790-b8b72116-418c-11e8-9a77-9444fc03bba6.JPG)


Next, we need to add the event that triggers this lambda function. This will be an “S3:ObjectCreated” event that happens every time a face is uploaded to the face S3 bucket. Add S3 trigger from designer section on the left. 

Configure with the following:

Bucket name: face-detection-your-name (you created this bucket earlier)
Event type- Object Created
Prefix- faces/
Filter- .jpg
Enable trigger- ON (keep the checkbox on)

Save the lambda function

Under 'Actions' tab choose **Publish**

**Step 4- View the emotions on a dashboard**

Go to [AWS Management console](https://console.aws.amazon.com/console/home?region=us-east-1) and search for Cloudwatch

Create a dashboard called “sentiment-dashboard-your-name”

Choose Line in the widget

Under Custom Namespaces, select “string”, “Metrics with no dimensions”, and then select all metrics.

Next, set “Auto-refresh” to the smallest interval possible (1h), and change the “Period” to whatever works best for you (1 second or 5 seconds)

NOTE: These metrics will only appear once they have been sent to Cloudwatch via the Rekognition Lambda. It may take some time for them to appear after your model is deployed and running locally. If they do not appear, then there is a problem somewhere in the pipeline.


### With this we have come to the end of the session. As part of building this project, you learnt the following:

1.	How to build and train a face detection model in SageMaker
2.	Modify the DeepLens inference lambda function to upload cropped faces to S3
3.	Deploy the inference lambda function and face detection model to DeepLens
4.	Create a lambda function to trigger Rekognition to identify emotions
5.	Create a DynamoDB table to store the recognized emotions
6.	Analyze using CloudWatch
