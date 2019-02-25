# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import greengrasssdk
from threading import Timer
import time
import awscam
from threading import Thread

# Creating a greengrass core sdk client
client = greengrasssdk.client('iot-data')

# The information exchanged between IoT and cloud has 
# a topic and a message body.
# This is the topic that this code uses to send messages to cloud
iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
modelPath = "/opt/awscam/artifacts"

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(modelPath,'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(modelPath, 'mscoco_label_map.pbtxt')

def greengrass_infinite_infer_run():
    try:
        # Load the TensorFlow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)
            
        client.publish(topic=iotTopic, payload="Model loaded")
        
        tensor_dict = {}
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes']:
            tensor_name = key + ':0'
            tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)
        #load label map
        label_dict = {}
        with open(PATH_TO_LABELS, 'r') as f:
                id=""
                for l in (s.strip() for s in f):
                        if "id:" in l:
                                id = l.strip('id:').replace('\"', '').strip()
                                label_dict[id]=''
                        if "display_name:" in l:
                                label_dict[id] = l.strip('display_name:').replace('\"', '').strip()

        client.publish(topic=iotTopic, payload="Start inferencing")
        while True:
            ret, frame = awscam.getLastFrame()
            if ret == False:
                raise Exception("Failed to get frame from the stream")
            expanded_frame = np.expand_dims(frame, 0)
            # Perform the actual detection by running the model with the image as input
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: expanded_frame})
            scores = output_dict['detection_scores'][0]
            classes = output_dict['detection_classes'][0]
            #only want inferences that have a prediction score of 50% and higher
            msg = '{'
            for idx, val in enumerate(scores):
                if val > 0.5:
                    msg += '"{}": {:.2f},'.format(label_dict[str(int(classes[idx]))], val*100)
            msg = msg.rstrip(',')
            msg +='}'
            
            client.publish(topic=iotTopic, payload = msg)
            
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
