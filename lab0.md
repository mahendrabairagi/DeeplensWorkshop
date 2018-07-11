## Register AWS DeepLens

Visit [AWS Management Console](https://console.aws.amazon.com/console/home?region=us-east-1). Make sure you are on US-East (N.Virginia) region.

Search for DeepLens in the search bar and select AWS DeepLens to open the console.

On the AWS DeepLens console screen, find the Get started section on the right hand side and select Register Device.

![register device landing page](https://user-images.githubusercontent.com/11222214/38656972-a73f8bd4-3dd2-11e8-8275-0486f8d78d2d.JPG)

### Step 1- Provide a name for your device.

Enter a name for your DeepLens device (for example, “MyDevice”), and select Next.

![name device](https://user-images.githubusercontent.com/11222214/38656982-b8d2b3d0-3dd2-11e8-9d00-060ccf015d0c.JPG)

### Step 2- Provide permissions

AWS DeepLens projects require different levels of permissions, which are set by AWS Identity and Access Management (IAM) roles. When registering your device for the first time, choose **Create roles** under Permissions to have the required IAM roles created. 

![create roles](https://user-images.githubusercontent.com/11222214/41578790-cad777a4-7348-11e8-97b1-b12f9a8f6549.jpg)

Then choose **Next**

![create roles- next](https://user-images.githubusercontent.com/11222214/41578802-e0c3ccc0-7348-11e8-9690-27adb740049c.jpg)

### Step 3- Download certificate
In this step, you will download and save the required certificate to your computer. You will use it later to enable your DeepLens to connect to AWS.

Select Download certificate and note the location of the certificates.zip file. Note: Do not open the zip file. You will attach this zip file later on during device registration.

![download certificate](https://user-images.githubusercontent.com/11222214/41578863-2fe8e4e8-7349-11e8-999b-8b4890bd4136.JPG)

Select **Continue.**

![download certificate- continue](https://user-images.githubusercontent.com/11222214/41578893-5b928842-7349-11e8-867b-bf79d293bd2f.JPG)


## Configure your DeepLens

In this step, you will connect the device to a Wi-Fi/Ethernet connection, upload the certificate and review your set-up. Then you're all set!

Power ON your device

<details> <summary>If you are connected over monitor setup </summary>
  
  Make sure the middle LED is blinking. If it is not, then use a pin to reset the device. The reset button is located at the back of the device
  
 Navigate to the setup page by choosing **Complete the setup** 
 
 ![last step](https://user-images.githubusercontent.com/11222214/41578985-c854efba-7349-11e8-8c73-62267c61091a.JPG)
  
</details>
  
<details> <summary>If you are connected in headless mode </summary>
  
  Make sure the middle LED is blinking. If it is not, then use a pin to reset the device. The reset button is located at the back of the device
  
  Locate the SSID/password of the device’s Wi-Fi. You can find the SSID/password on the underside of the device.
  
  Connect to the device network via the SSID and provide the password
  
  Navigate to the setup page by choosing **Complete the setup** 
  
  ![last step](https://user-images.githubusercontent.com/11222214/41578985-c854efba-7349-11e8-8c73-62267c61091a.JPG)
  
</details>

### Step 4- Connect to your network

If you are using Ethernet, choose Ethernet option

If yoy are using Wi-fi then select your local Wi-Fi network ID from the dropdown list and enter your WiFi password. 

Select Save.

![network connection](https://user-images.githubusercontent.com/11222214/38657139-77c96aa4-3dd3-11e8-8cba-97dc3c47fc66.JPG)

If this is your first time registering the device, you will see the updates available screen. Choose **Install and Reboot** It will take couple of minutes for the updates to come through. After the updates are installed, the device will reboot automatically

****If you are using monitor setup and ethernet then ***ignore wifi setup instructions*** and choose **Install and Reboot**

![](/screenshots/installandreboot1.jpg)

![install and reboot](https://user-images.githubusercontent.com/11222214/41579269-14d84e30-734b-11e8-8894-c4a76f1715d5.JPG)

On rebooting, the device will come back to the install and reboot screen. From the URL, delete the #softwareUpdate

****If you are using headerless mode, i.e. with no monitor,keyboard setup then follow instructions on the screen and choose **Install and Reboot**

![software update remove](https://user-images.githubusercontent.com/11222214/41579379-a3deed32-734b-11e8-894a-c209cb7a7cca.JPG)

Click on link on step4 in above screenshot, "deeplens.amazon.net" to complete the setup.

### Step 5- Attach Certificates

Select Browse in the Certificate section. Select the zip file you downloaded in Step 4 

Select Next.

![upload certificate](https://user-images.githubusercontent.com/11222214/38657156-8cc8c5b2-3dd3-11e8-9261-dda8a8925cca.JPG)

### Step 6- Download Streaming Certificate

Choose **Download** . Then choose **Next**

![download streaming cert](https://user-images.githubusercontent.com/11222214/41579452-e253dca8-734b-11e8-9a47-5a7f48b6a0db.JPG)

### Step 7- Device set up.

If you are on the device summary page- Please do not make changes to the password.

Note: Instead, if you are presented with the below screen, type the device password as Aws2017! . 

![device settings](https://user-images.githubusercontent.com/11222214/38657201-c44385fe-3dd3-11e8-8497-7add710be21b.JPG)

### Step 7- Select Finish

![summary](https://user-images.githubusercontent.com/11222214/41579495-0d5235e4-734c-11e8-987a-18a0b83259cc.JPG)


Congratulations! You have successfully registered and configured your DeepLens device. To verify, return to [AWS DeepLens console](https://console.aws.amazon.com/deeplens/home?region=us-east-1#projects) and select **Devices** in the left side navigation bar and verify that your device has completed the registration process. You should see a green check mark and Registered under Registration status.

![device registered successfully](https://user-images.githubusercontent.com/11222214/41579540-3cc3560a-734c-11e8-82c0-6fc18c3952c8.JPG)

