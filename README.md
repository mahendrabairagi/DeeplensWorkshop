# DeepLens-workshops

In this workshop you will learn how to register and configure your DeepLens and deploy a face detection project to your DeepLens. You will see a demo of sentiment analysis project. This project was submitted by Ricardo Mota and Jidesh Veeramachaneni as part of the DeepLens Hackathon challenge. The project was built by extending the face detection project and integrating it with Amazon Rekognition. 

The workshop consists of 2 hands-on lab sessions:

### Hands-on Lab 1: Build and train a face detection model in SageMaker

In this lab, you will build and train a face detection model. You can find the instructions here: [SageMaker lab](https://github.com/fibbonnaci/DeepLens-workshops/tree/master/SageMaker%20lab)

### Hands-on Lab 2: Register your DeepLens and deploy to device.

The instructions for this hands-on lab is provided below.

## Register AWS DeepLens

Visit [AWS Management Console](https://console.aws.amazon.com/console/home?region=us-east-1). Make sure you are on US-East (N.Virginia) region.

Search for DeepLens in the search bar and select AWS DeepLens to open the console.

On the AWS DeepLens console screen, find the Get started section on the right hand side and select Register Device.

![register device landing page](https://user-images.githubusercontent.com/11222214/38656972-a73f8bd4-3dd2-11e8-8275-0486f8d78d2d.JPG)

### Step 1- Provide a name for your device.

Enter a name for your DeepLens device (for example, “MyDevice”), and select Next.

![name device](https://user-images.githubusercontent.com/11222214/38656982-b8d2b3d0-3dd2-11e8-9d00-060ccf015d0c.JPG)

### Step 2- Provide permissions

AWS DeepLens projects require different levels of permissions, which are set by AWS Identity and Access Management (IAM) roles. When registering your device for the first time, you'll need to create each one of these IAM roles.

### Role 1- IAM role for AWS DeepLens

Select Create a role in IAM.

![create role-deeplens](https://user-images.githubusercontent.com/11222214/38657020-e6c85cd6-3dd2-11e8-8f02-a737e1eef657.JPG)

Use case is selected by default. Click Next:Permissions

Click Next:Review

Click Create role 

![service role review page](https://user-images.githubusercontent.com/11222214/38657029-f6ea2aae-3dd2-11e8-99f6-0d7230a1eaae.JPG)

Return back to the Set Permission page, select on Refresh IAM roles and select the newly created Role name **AWSDeepLensServiceRole.**

![refresh role- service roles](https://user-images.githubusercontent.com/11222214/38657050-09a0a7ea-3dd3-11e8-8a80-403d61a6a7e3.JPG)


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

![all roles](https://user-images.githubusercontent.com/11222214/38657064-1e278a8a-3dd3-11e8-9dd9-65bbffb22a92.JPG)

Once you have all the roles correctly created and populated, select **Next.**

### Step 3- Download certificate
In this step, you will download and save the required certificate to your computer. You will use it later to enable your DeepLens to connect to AWS.

Select Download certificate and note the location of the certificates.zip file. Select Register.

![download certificate](https://user-images.githubusercontent.com/11222214/38657089-3219184c-3dd3-11e8-8f06-2609898b07cc.JPG)

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
  
  ![set up guide](https://user-images.githubusercontent.com/11222214/38657118-5266e610-3dd3-11e8-8c64-23fd362e708a.JPG)
  
</details>

### Step 4- Connect to your network

Select your local Wi-Fi network ID from the dropdown list and enter your WiFi password. If you are using ethernet, choose Use Ethernet option instead.

Select Save.

![network connection](https://user-images.githubusercontent.com/11222214/38657139-77c96aa4-3dd3-11e8-8cba-97dc3c47fc66.JPG)

### Step 5- Attach Certificates

Select Browse in the Certificate section. Select the zip file you downloaded in Step 4 

Select Next.

![upload certificate](https://user-images.githubusercontent.com/11222214/38657156-8cc8c5b2-3dd3-11e8-9261-dda8a8925cca.JPG)

### Step 6- Device set up.

If you are on the device summary page- Please do not make changes to the password.

Instead, if you are presented with the below screen, type the device password as Aws2017! . 

![device settings](https://user-images.githubusercontent.com/11222214/38657201-c44385fe-3dd3-11e8-8497-7add710be21b.JPG)

### Step 7- Select Finish

![set up summary finish](https://user-images.githubusercontent.com/11222214/38657410-ea300d36-3dd4-11e8-9312-c3ef909a1771.JPG)


Congratulations! You have successfully registered and configured your DeepLens device. To verify, return to [AWS DeepLens console](https://console.aws.amazon.com/deeplens/home?region=us-east-1#projects) and select **Devices** in the left side navigation bar and verify that your device has completed the registration process. You should see a green check mark and Completed under Registration status.

## Deploy Face Recognition project

### Step 1- Create Project

The console should open on the Projects screen, select Create new project on the top right (if you don’t see the project list view, click on the hamburger menu on the left and select Projects)

Choose, Use a **project template** as the Project type, and select **Face Detection** from the project templates list.

Scroll down the screen and select **Next**

Change the Project name as Face-detection-your-name

Scroll down the screen and select **Create**

### Step 2- Deploy to device
In this step, you will deploy the Face detection project to your AWS DeepLens device.

Select the project you just created from the list by choosing the radio button

Select Deploy to device.

On the Target device screen, choose your device from the list, and select **Review.**

Select Deploy.

On the AWS DeepLens console, you can track the progress of the deployment. It can take a few minutes to transfer a large model file to the device. Once the project is downloaded, you will see a success message displayed and the banner color will change from blue to green.

To view the output, open a terminal (on the desktop, choose the top left button and search for terminal) and enter the following command:

`mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg`



