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

# Hands-on Step 1: Register and configure your DeepLens device (You can skip this lab if device registration is already complete)

Follow instructions here: [Registration and Deployment lab](https://github.com/mahendrabairagi/DeeplensWorkshop/tree/master/Registration%20and%20project%20deployment)


# Hands-on Step 2: Build and train a face detection model in SageMaker

In this lab, you will build and train a face detection model. Follow instructions here: [SageMaker lab](https://github.com/mahendrabairagi/DeeplensWorkshop/tree/master/SageMaker%20lab)

# Hands-on Step 3: Build a project to detect faces and send the cropped faces to S3 bucket

#### IAM Roles: (Optional step - if IAM role exists then skip this step)

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

A DeepLens **Project** consists of two things:
* A model artifact: This is the model that is used for inference.
* A Lambda function: This is the script that runs inference on the device.

Before we deploy a project to DeepLens, we need to create a custom lambda function that will use the face-detection model on the device to detect faces and push crops to S3.

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

**Note**: In the script, you will have to provide the name for your S3 bucket. Insert your bucket name in the code below

![code bucket](https://user-images.githubusercontent.com/11222214/38719807-b46169fa-3ea8-11e8-8ff2-69af5455ede7.jpg)

Click Save

![Alt text](/screenshots/deeplens_lambda_0.png)


```python
#
# Copyright Amazon AWS DeepLens, 2017
#

import os
import sys
import datetime
import greengrasssdk
from threading import Timer
import time
import awscam
import cv2
from threading import Thread
import urllib
import zipfile

#boto3 is not installed on device by default.

boto_dir = '/tmp/boto_dir'
if not os.path.exists(boto_dir):
    os.mkdir(boto_dir)
urllib.urlretrieve("https://s3.amazonaws.com/dear-demo/boto_3_dist.zip", "/tmp/boto_3_dist.zip")
with zipfile.ZipFile("/tmp/boto_3_dist.zip", "r") as zip_ref:
    zip_ref.extractall(boto_dir)
sys.path.append(boto_dir)

import boto3

# Creating a greengrass core sdk client
client = greengrasssdk.client('iot-data')

# The information exchanged between IoT and clould has
# a topic and a message body.
# This is the topic that this code uses to send messages to cloud
iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])

ret, frame = awscam.getLastFrame()
ret, jpeg = cv2.imencode('.jpg', frame)

Write_To_FIFO = True

class FIFO_Thread(Thread):
    def __init__(self):
        ''' Constructor. '''
        Thread.__init__(self)

    def run(self):
        fifo_path = "/tmp/results.mjpeg"
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)
        f = open(fifo_path, 'w')
        client.publish(topic=iotTopic, payload="Opened Pipe")
        while Write_To_FIFO:
            try:
                f.write(jpeg.tobytes())
            except IOError as e:
                continue

def push_to_s3(img, index):
    try:
        bucket_name = "your-bucket"

        timestamp = int(time.time())
        now = datetime.datetime.now()
        key = "faces/{}_{}/{}_{}/{}_{}.jpg".format(now.month, now.day,
                                                   now.hour, now.minute,
                                                   timestamp, index)

        s3 = boto3.client('s3')

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, jpg_data = cv2.imencode('.jpg', img, encode_param)
        response = s3.put_object(ACL='public-read',
                                 Body=jpg_data.tostring(),
                                 Bucket=bucket_name,
                                 Key=key)

        client.publish(topic=iotTopic, payload="Response: {}".format(response))
        client.publish(topic=iotTopic, payload="Face pushed to S3")
    except Exception as e:
        msg = "Pushing to S3 failed: " + str(e)
        client.publish(topic=iotTopic, payload=msg)

def greengrass_infinite_infer_run():
    try:
        modelPath = "/opt/awscam/artifacts/mxnet_deploy_ssd_FP16_FUSED.xml"
        modelType = "ssd"
        input_width = 300
        input_height = 300
        prob_thresh = 0.25
        results_thread = FIFO_Thread()
        results_thread.start()

        # Send a starting message to IoT console
        client.publish(topic=iotTopic, payload="Face detection starts now")

        # Load model to GPU (use {"GPU": 0} for CPU)
        mcfg = {"GPU": 1}
        model = awscam.Model(modelPath, mcfg)
        client.publish(topic=iotTopic, payload="Model loaded")
        ret, frame = awscam.getLastFrame()
        if ret == False:
            raise Exception("Failed to get frame from the stream")

        yscale = float(frame.shape[0]/input_height)
        xscale = float(frame.shape[1]/input_width)

        doInfer = True
        while doInfer:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            # Raise an exception if failing to get a frame
            if ret == False:
                raise Exception("Failed to get frame from the stream")

            # Resize frame to fit model input requirement
            frameResize = cv2.resize(frame, (input_width, input_height))

            # Run model inference on the resized frame
            inferOutput = model.doInference(frameResize)

            # Output inference result to the fifo file so it can be viewed with mplayer
            parsed_results = model.parseResult(modelType, inferOutput)['ssd']
            # client.publish(topic=iotTopic, payload = json.dumps(parsed_results))
            label = '{'
            for i, obj in enumerate(parsed_results):
                if obj['prob'] < prob_thresh:
                    break
                offset = 25
                xmin = int( xscale * obj['xmin'] ) + int((obj['xmin'] - input_width/2) + input_width/2)
                ymin = int( yscale * obj['ymin'] )
                xmax = int( xscale * obj['xmax'] ) + int((obj['xmax'] - input_width/2) + input_width/2)
                ymax = int( yscale * obj['ymax'] )

                crop_img = frame[ymin:ymax, xmin:xmax]

                push_to_s3(crop_img, i)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 20), 4)
                label += '"{}": {:.2f},'.format(str(obj['label']), obj['prob'] )
                label_show = '{}: {:.2f}'.format(str(obj['label']), obj['prob'] )
                cv2.putText(frame, label_show, (xmin, ymin-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 20), 4)
            label += '"null": 0.0'
            label += '}'
            client.publish(topic=iotTopic, payload=label)
            global jpeg
            ret, jpeg = cv2.imencode('.jpg', frame)

    except Exception as e:
        msg = "Test failed: " + str(e)
        client.publish(topic=iotTopic, payload=msg)

    # Asynchronously schedule this function to be run again in 15 seconds
    Timer(15, greengrass_infinite_infer_run).start()


# Execute the function above
greengrass_infinite_infer_run()


# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    return
```

Once you've copied and pasted the code, click "Save" as before, and this time you'll also click "Actions" and then "Publish new version".

![Alt text](/screenshots/deeplens_lambda_1.png)

Then, enter a brief description and click "Publish."

![Alt text](/screenshots/deeplens_lambda_2.png)

Before we can run this lambda on the device, we need to attach the right permissions to the right roles. While we assigned a role to this lambda, "AWSDeepLensLambdaRole", it's only a placeholder. Lambda's deployed through greengrass actually inherit their policy through a greengrass group role.

We need to add permissions to this role for the lambda function to access S3. To do this, go to the IAM dashboard, find the "AWSDeepLensGreenGrassGroupRole", and attach the policy "AmazonS3FullAccess". 

### Create & Deploy DeepLens Project

With the lambda created, we can now make a project using it and the built-in face detection model.

From the DeepLens homepage dashboard, select "Projects" from the left side-bar:

![Alt text](/screenshots/deeplens_project_0.png)

Then select "Create new project"

![Alt text](/screenshots/deeplens_project_1.png)

Next, select "Create a new blank project" then click "Next".

![Alt text](/screenshots/deeplens_project_2.png)

Now, name your deeplens project.

![Alt text](/screenshots/deeplens_project_3.png)

Next, select "Add model". From the pop-up window, select "deeplens-face-detection" then click "Add model".

![Alt text](/screenshots/deeplens_project_4.png)

Next, select "Add function". from the pop-up window, select your deeplens lambda function and click "Add function".

![Alt text](/screenshots/deeplens_project_5.png)

Finally, click "Create".

![Alt text](/screenshots/deeplens_project_6.png)

Now that the project has been created, you will select your project from the project dashboard and click "Deploy to device".

![Alt text](/screenshots/deeplens_project_7.png)

Select the device you're deploying too, then click "Review" (your screen will look different here).

![Alt text](/screenshots/deeplens_project_8.png)

Finally, click "Deploy" on the next screen to begin project deployment.

![Alt text](/screenshots/deeplens_project_9.png)

You should now start to see deployment status. Once the project has been deployed, your deeplens will now start processing frames and running face-detection locally. When faces are detected, it will push to your S3 bucket. Everything else in the pipeline remains the same, so return to your dashboard to see the new results coming in!

**Note**: If your model download progress hangs at a blank state (Not 0%, but **blank**) then you may need to reset greengrass on DeepLens. To do this, log onto the DeepLens device, open up a terminal, and type the following command:
`sudo systemctl restart greengrassd.service --no-block`. After a couple minutes, you model should start to download.


**Confirmation/ verification**

You will find your cropped faces uplaod to your S3 bucket.



# Hands-on Step 4: Identify emotions

**Step I- Create DynamoDB table**

Go to [AWS Management console](https://console.aws.amazon.com/console/home?region=us-east-1) and search for Dynamo

Click on Create Table.

Name of the table: recognize-emotions-your-name
Primary key: s3key

Click on Create. This will create a table in your DynamoDB.

**Step II- Create a role for cloud lambda function** (Optional step - skip this step if Role already exists)

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


**Step III- Create a lambda function that runs in the cloud**

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

**Step IV- View the emotions on a dashboard**

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
