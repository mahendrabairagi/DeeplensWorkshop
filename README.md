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

### Role 1- IAM role for AWS DeepLens

Select Create a role in IAM.

Use case is selected by default. Click Next:Permissions

Click Next:Review

Click Create role 

Return back to the Set Permission page, select on Refresh IAM roles and select the newly created Role name **AWSDeepLensServiceRole.**

### Role 2- IAM role for AWS Greengrass 

Select Create a role in IAM.

Use case is selected by default. Click Next:Permissions

Click Next:Review

Click Create role 

Return back to the Set Permission page, select on Refresh IAM roles and select the newly created Role name **AWSDeepLensGreengrassRole.**

### Role 3- IAM group role for AWS Greengrass

Select Create a role in IAM.
Use case is selected by default. Click Next:Permissions
Click Next:Review
Click Create role 
Return back to the Set Permission page, select on Refresh IAM roles and select the newly created Role name **AWSDeepLensGreengrassGroupRole.**

### Role 4- IAM role for Amazon SageMaker

Select Create a role in IAM.
Use case is selected by default. Click Next:Permissions
Click Next:Review
Click Create role 
Return back to the Set Permission page, select on Refresh IAM roles and select the newly created Role name **AWSDeepLensSageMakerRole.**

### Role 5- IAM role for AWS Lambda

Select Create a role in IAM.
Use case is selected by default. Click Next:Permissions
Click Next:Review
Click Create role 
Return back to the Set Permission page, select on Refresh IAM roles and select the newly created Role name **AWSDeepLensLambdaRole.**

Note: These roles are very important. Make sure that you select the right role for each one, as you can see in the screenshot.

Once you have all the roles correctly created and populated, select **Next.**

### Step 3- Download certificate
In this step, you will download and save the required certificate to your computer. You will use it later to enable your DeepLens to connect to AWS.

Select Download certificate and note the location of the certificates.zip file. Select Register.

Note: Do not open the zip file. You will attach this zip file later on during device registration.

## Configure your DeepLens

In this step, you will connect the device to a Wi-Fi/Ethernet connection, upload the certificate and review your set-up. Then you're all set!

Power ON your device

<details> <summary>If you are connected over monitor setup </summary>
  
  Make sure the middle LED is blinking. If it is not, then use a pin to reset the device. The reset button is located at the back of the device
  
  Navigate to the setup page at **192.168.0.1.**
  
</details>
  
<details> <summary>If you are connected in headless mode </summary>
  
  Make sure the middle LED is blinking. If it is not, then use a pin to reset the device. The reset button is located at the back of the device
  
  Locate the SSID/password of the device’s Wi-Fi. You can find the SSID/password on the underside of the device.
  
  Connect to the device network via the SSID and provide the password
  
  Navigate to the setup page at **192.168.0.1.**
</details>

### Step 4- Connect to your network

Select your local Wi-Fi network ID from the dropdown list and enter your WiFi password. If you are using ethernet, choose Use Ethernet option instead.

Select Next.

### Step 5- Attach Certificates

Select Browse in the Certificate section. Select the zip file you downloaded in Step 4 

Select Next.

### Step 6- Device set up.

If you are on the device summary page- Please do not make changes to the password.

Instead, if you are presented with the below screen, type the device password as Aws2017! . 

### Step 7- Select Finish

Congratulations! You have successfully registered and configured your DeepLens device. To verify, return to [AWS DeepLens console](https://console.aws.amazon.com/deeplens/home?region=us-east-1#projects) and select **Devices** in the left side navigation bar and verify that your device has completed the registration process. You should see a green check mark and Completed under Registration status.
