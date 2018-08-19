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
import mo

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
        bucket_name = "<BUCKET_NAME>"

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
        modelType = "ssd"
        model_name = "model"
        input_width = 300
        input_height = 300
        modelPath = "/opt/awscam/artifacts/mxnet_deploy_ssd_FP16_FUSED.xml"
        #error, modelPath = mo.optimize(model_name,input_width,input_height)

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
