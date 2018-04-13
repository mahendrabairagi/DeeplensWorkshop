from __future__ import print_function

import boto3
import urllib
import datetime

print('Loading function')

rekognition = boto3.client('rekognition')
cloudwatch = boto3.client('cloudwatch')


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
    response = rekognition.detect_faces(Image={"S3Object":
                                               {"Bucket": bucket,
                                                "Name": key}},
                                        Attributes=['ALL'])

    if not response['FaceDetails']:
        print ("No Face Details Found!")
        return response

    push = False
    dynamo_obj = {}
    dynamo_obj['s3key'] = key

    for index, item in enumerate(response['FaceDetails'][0]['Emotions']):
        print("Item: {}".format(item))
        if int(item['Confidence']) > 10:
            push = True
            dynamo_obj[item['Type']] = str(round(item["Confidence"], 2))
            push_to_cloudwatch(item['Type'], round(item["Confidence"], 2))

    if push:  # Push only if at least on emotion was found
        table = boto3.resource('dynamodb').Table('rekognize-faces')
        table.put_item(Item=dynamo_obj)

    return response

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
