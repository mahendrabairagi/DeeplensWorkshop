# Use case
Use case for this workshop is detect person through the Deeplens and once person is detected retrieve information associated with the person.

## In this workshop you will learn the following:

1. Create unique ids for each face image using AWS Rekogtion

2. Create profile for each face ids using AWS DynamoDB

3. Detect faces through AWS Deeplens, save cropped faces to AWS s3. 

4. Identify faces stored in S3 and retrieve profile through AWS Lambda on Cloud and DynamoDB 

5. Publish profiles to AWS IoT and AWS SNS


### Prerequsite:

I. Register and configure your DeepLens device (You can skip this lab if device registration is already complete)
Follow instructions here: 

[Lab 0 Link](https://github.com/mahendrabairagi/DeeplensWorkshop/blob/master/lab0.md)

II. Deploy out of box face detection model

[Lab 1 Link](https://github.com/mahendrabairagi/DeeplensWorkshop/blob/master/lab1.md)


### Step 1: Create unique ids for each face image using AWS Rekogtion
We will use AWS CLI (command line interface) to create Face ID, Dynamodb table

We will first setup AWS cli ( you can skip this step if AWS CLI is already setup)

I. Setup AWS CLI
On your machine  https://docs.aws.amazon.com/lambda/latest/dg/setup-awscli.html

Or using cloud9 online development environment https://docs.aws.amazon.com/cloud9/latest/user-guide/setup-express.html

II. Open terminal

Let’s create unique S3 bucket first

On terminal, execute following command, replace S3 bucket <s3bucket> with bucket name you want. I am using bucket name "Deeplens-rekognition-xxxxxx"

**** Make sure you are in US-EAST-1 region

```
aws s3 mb s3://deeplens-rekognition-xxxxxx
```

Upload photos to s3.

aws s3 cp <my picture> s3://<my-bucket>

e.g.

```
aws s3 cp myphoto.jpg s3://deeplens-rekognition-xxxxxx
```


Crete Rekognition collection

```
aws rekognition create-collection --collection-id rekognition-collection
```
Please make note of the CollectionArn from command above


Output of above command should look like this:
```
{
    "StatusCode": 200,
    "CollectionArn": "aws:rekognition:us-east-1:xxxxx:collection/rekognition-collection",
    "FaceModelVersion": "3.0"
}
```

Please make not of the face id

"FaceId": "xxxxxxxxxxxxxxxxxxxxxxxxx",

### Step 2. Create profile for each face ids using AWS DynamoDB

We will use AWS terminal to create dynamoDB table. In this example the table name is rekognition-collection

```
aws dynamodb create-table --table-name rekognition-collection \
--attribute-definitions AttributeName=FaceId,AttributeType=S \
--key-schema AttributeName=FaceId,KeyType=HASH \
--provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
--region us-east-1
```

Add record in the DynamoDB table

```
aws dynamodb put-item --table-name analyst-rekognition-collection --item '{ "FaceId": {"S": "xx_Face_Id_from_Step1xx"}, "Bio": {"S": "This Mahendra Bairagis Bio and latest research"}}'
```



### Step 3: Detect faces through AWS Deeplens, save cropped faces to AWS s3. 

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
		            
                if ymin >25:
                   ymin = ymin - 25
           
                if ymax <275:
                   ymax = ymax + 25
                   
                if xmin > 25:
                   xmin = xmin - 25
                
                if xmax <275:
                   xmax = xmax + 25   
		            
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

`sudo systemctl restart greengrassd.service --no-block`. 

After a couple minutes, you model should start to download.


**Confirmation/ verification**

You will find your cropped faces uplaod to your S3 bucket.



### Step 4: Identify faces stored in S3 and retrieve profile through AWS Lambda on Cloud and DynamoDB 

**I- Create a role for cloud lambda function**

Go to [AWS Management console](https://console.aws.amazon.com/console/home?region=us-east-1) and search for IAM

Choose 'Create Role'

Select “AWS Service”

Select “Lambda” and choose "Next:Permissions"

Attach the following policies: 

* AmazonDynamoDBFullAcces

* AmazonS3FullAccess

* AmazonRekognitionFullAccess

* CloudWatchFullAccess

* AWSIoTFullAccess

![Alt text](/screenshots/roles0.png)

Click Next

Provide a name for the role: rekognizePerson

Choose 'Create role'


**II- Create a lambda function that runs in the cloud**

The inference lambda function that you deployed earlier will upload the cropped faces to your S3. On S3 upload, this new lambda function gets triggered and runs the Rekognize search_faces_by_image by integrating with Amazon Rekognition. 

Go to [AWS Management console](https://console.aws.amazon.com/console/home?region=us-east-1) and search for Lambda

Click 'Create function'

Choose 'Author from scratch'

Name the function: rekognize-person

Runtime: Choose Python 2.7

Role: Choose an existing role

Existing role: rekognizePerson

Choose Create function

![Alt text](/screenshots/lambda_create_function0.png)

Replace the default script with the script in [rekognize-person.py](https://github.com/mahendrabairagi/DeeplensWorkshop/blob/master/Integrate%20with%20Rekognition/rekognize-person.py). You can select the script by selecting Raw in the Github page and choosing the script using ctrl+A/ cmd+A . Copy the script and paste it into the lambda function (make sure you delete the default code).

***Make sure you enter the "CollectionId=" (line 40) as one you created e.g. "rekognition-collection"

***Make sure you enter the "table =" (line 14) as one you created e.g. table = dynamodb.Table('rekognition-collection')

***Check the IoT topic name (line 52), in this example it is "rekognition" e.g. iotdata.publish(topic='rekognition',qos=0, payload=json.dumps(profile))


Next, we need to add the event that triggers this lambda function. This will be an “S3:ObjectCreated” event that happens every time a face is uploaded to the face S3 bucket. Add S3 trigger from designer section on the left. 

Configure with the following:

Bucket name: face-detection-your-name (you created this bucket earlier)

Event type- Object Created

Prefix- faces/

Filter- .jpg

Enable trigger- ON (keep the checkbox on)

![Alt text](/screenshots/s3_trigger0.png)

![Alt text](/screenshots/s3_trigger1.png)

Save the lambda function

Under 'Actions' tab choose **Publish**


### Step 5. Publish profiles to AWS IoT and AWS SNS

Go to [AWS Management console](https://console.aws.amazon.com/console/home?region=us-east-1) and search for IoT Core

Select "Test" from left side menu

![Alt text](/screenshots/iot_Test0.png)


Then select "Subscribe to a topic"

![Alt text](/screenshots/iot_Test1.png)


In Topic give name as one in lambda e.g. rekognition

You will start seeing messages 

![Alt text](/screenshots/iot_Test1.png)

To add setup SNS messages create IoT rule to send SNS per link
https://docs.aws.amazon.com/iot/latest/developerguide/iot-sns-rule.html

replace topic name to topic name used in the lambda e.g. analyst_rekognition


### With this we have come to the end of the session. As part of building this project, you learnt the following:

1.	Use AWS Rekogntion and DynamoDB to create unique profile for each person based on face id

2.	Modify the DeepLens inference lambda function to upload cropped faces to S3

3.	Deploy the inference lambda function and face detection model to DeepLens

4.	Create a lambda function to trigger Rekognition to identify face

5.	Analyze using AWS IoT and AWS SNS

