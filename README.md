# DeepLens-workshops

In this workshop you will learn how to register and configure your DeepLens and deploy a face detection project to your DeepLens. You will see a demo of sentiment analysis project. This project was submitted by Ricardo Mota and Jidesh Veeramachaneni as part of the DeepLens Hackathon challenge. The project was built by extending the face detection project and integrating it with Amazon Rekognition. 

## Register AWS DeepLens

Visit [AWS Management Console](https://console.aws.amazon.com/console/home?region=us-east-1). Make sure you are on US-East (N.Virginia) region.

Search for DeepLens in the search bar and select AWS DeepLens to open the console.

On the AWS DeepLens console screen, find the Get started section on the right hand side and select Register Device.

### Step 1- Provide a name for your device.

Enter a name for your DeepLens device (for example, “MyDevice”), and select Next.

### Step 2- Provide permissions

AWS DeepLens projects require different levels of permissions, which are set by AWS Identity and Access Management (IAM) roles. When registering your device for the first time, you'll need to create each one of these IAM roles.

<details> <summary> IAM role for AWS DeepLens </summary>

  select Create a role in IAM.
  Use case is selected by default. Click Next:Permissions
  Click Next:Review
  Click Create role 

</details>

<details> <summary> IAM role for AWS Greengrass </summary>
<details> <summary> IAM group role for AWS Greengrass </summary>
<details> <summary> IAM role for Amazon SageMaker </summary>
<details> <summary> IAM role for AWS Lambda </summary>

