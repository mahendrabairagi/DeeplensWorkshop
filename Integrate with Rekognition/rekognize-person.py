from __future__ import print_function

import boto3
import urllib
import datetime
import json

print('Loading function')

rekognition = boto3.client('rekognition')
cloudwatch = boto3.client('cloudwatch')
iotdata = boto3.client('iot-data')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('rekognition-collection')


# --------------- Helper Function to call CloudWatch APIs ------------------

def push_to_cloudwatch(name, value):
    try:
        response = cloudwatch.put_metric_data(
            Namespace='string',
            MetricData=[
                {
                    'MetricName': name,
                    'Value': value,
                    'Unit': 'Percent'
                },
            ]
        )
        print("Metric pushed: {}".format(response))
    except Exception as e:
        print("Unable to push to cloudwatch\n e: {}".format(e))
        return True

# --------------- Helper Functions to call Rekognition APIs ------------------

def detect_faces(bucket, key):
    print("Key: {}".format(key))
    response = rekognition.search_faces_by_image(CollectionId="rekognition-collection", Image={"S3Object":
                                               {"Bucket": bucket,
                                                "Name": key}})

    face_matches=response['FaceMatches']
    if face_matches:
        for face_match in face_matches: 
            face = face_match['Face']
            profile = {}
            profile['FaceId'] = face['FaceId']
            response=table.get_item(Key=profile)
            profile=response['Item']
            iotdata.publish(topic='rekognition',qos=0, payload=json.dumps(profile))
            print (profile)
    else:
        print ("No Face Details Found!")

    return profile


# --------------- Main handler ------------------


def lambda_handler(event, context):
    '''Demonstrates S3 trigger that uses
    Rekognition APIs to detect faces, labels and index faces in S3 Object.
    '''

    # Get the object from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.unquote_plus(event['Records'][0]['s3']['object']['key'].encode('utf8'))

    try:
        # Calls rekognition DetectFaces API to detect faces in S3 object
        response = detect_faces(bucket, key)

        return response
    except Exception as e:
        print("Error processing object {} from bucket {}. ".format(key, bucket) +
              "Make sure your object and bucket exist and your bucket is in the same region as this function.")
        raise e
