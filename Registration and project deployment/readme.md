Register AWS DeepLens
Visit AWS Management Console. Make sure you are on US-East (N.Virginia) region.

Search for DeepLens in the search bar and select AWS DeepLens to open the console.

On the AWS DeepLens console screen, find the Get started section on the right hand side and select Register Device.

register device landing page

Step 1- Provide a name for your device.
Enter a name for your DeepLens device (for example, “MyDevice”), and select Next.

name device

Step 2- Provide permissions
AWS DeepLens projects require different levels of permissions, which are set by AWS Identity and Access Management (IAM) roles. When registering your device for the first time, you'll need to create each one of these IAM roles.

Role 1- IAM role for AWS DeepLens
Select Create a role in IAM.

create role-deeplens

Use case is selected by default. Click Next:Permissions

Click Next:Review

Click Create role

service role review page

Return back to the Set Permission page, select on Refresh IAM roles and select the newly created Role name AWSDeepLensServiceRole.

refresh role- service roles

Role 2- IAM role for AWS Greengrass
Select Create a role in IAM.

Use case is selected by default. Click Next:Permissions

Click Next:Review

Click Create role

Return back to the Set Permission page, select on Refresh IAM roles and select the newly created Role name AWSDeepLensGreengrassRole.

Role 3- IAM group role for AWS Greengrass
Select Create a role in IAM. Use case is selected by default. Click Next:Permissions Click Next:Review Click Create role Return back to the Set Permission page, select on Refresh IAM roles and select the newly created Role name AWSDeepLensGreengrassGroupRole.

Role 4- IAM role for Amazon SageMaker
Select Create a role in IAM. Use case is selected by default. Click Next:Permissions Click Next:Review Click Create role Return back to the Set Permission page, select on Refresh IAM roles and select the newly created Role name AWSDeepLensSageMakerRole.

Role 5- IAM role for AWS Lambda
Select Create a role in IAM. Use case is selected by default. Click Next:Permissions Click Next:Review Click Create role Return back to the Set Permission page, select on Refresh IAM roles and select the newly created Role name AWSDeepLensLambdaRole.

Note: These roles are very important. Make sure that you select the right role for each one, as you can see in the screenshot.

all roles

Once you have all the roles correctly created and populated, select Next.

Step 3- Download certificate
In this step, you will download and save the required certificate to your computer. You will use it later to enable your DeepLens to connect to AWS.

Select Download certificate and note the location of the certificates.zip file. Select Register.

download certificate

Note: Do not open the zip file. You will attach this zip file later on during device registration.

Configure your DeepLens
In this step, you will connect the device to a Wi-Fi/Ethernet connection, upload the certificate and review your set-up. Then you're all set!

Power ON your device

If you are connected over monitor setup
If you are connected in headless mode
Step 4- Connect to your network
Select your local Wi-Fi network ID from the dropdown list and enter your WiFi password. If you are using ethernet, choose Use Ethernet option instead.

Select Save.

network connection

Step 5- Attach Certificates
Select Browse in the Certificate section. Select the zip file you downloaded in Step 4

Select Next.

upload certificate

Step 6- Device set up.
If you are on the device summary page- Please do not make changes to the password.

Note: Instead, if you are presented with the below screen, type the device password as Aws2017! .

device settings

Step 7- Select Finish
set up summary finish

Congratulations! You have successfully registered and configured your DeepLens device. To verify, return to AWS DeepLens console and select Devices in the left side navigation bar and verify that your device has completed the registration process. You should see a green check mark and Completed under Registration status.
