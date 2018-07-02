## Deploy Face Recognition project

### Step 1- Create Project

The console should open on the Projects screen, select Create new project on the top right (if you don’t see the project list view, click on the hamburger menu on the left and select Projects)

![create project](https://user-images.githubusercontent.com/11222214/38657905-82207e44-3dd7-11e8-83ef-52049e229e33.JPG)

Choose, Use a **project template** as the Project type, and select **Face Detection** from the project templates list.

![project template](https://user-images.githubusercontent.com/11222214/38657922-958edd7c-3dd7-11e8-830b-ec129d9363e6.JPG)

Scroll down the screen and select **Next**

![project template-next](https://user-images.githubusercontent.com/11222214/38657930-a3f6c1a4-3dd7-11e8-96a9-3f3cebb1712e.JPG)

Change the Project name as Face-detection-your-name

![face detection your name](https://user-images.githubusercontent.com/11222214/38657948-b8cc049a-3dd7-11e8-948f-1d32948408d1.JPG)

Scroll down the screen and select **Create**


![click create](https://user-images.githubusercontent.com/11222214/38657969-d573db7c-3dd7-11e8-9f45-fc6d1eb25a4b.JPG)


### Step 2- Deploy to device
In this step, you will deploy the Face detection project to your AWS DeepLens device.

Select the project you just created from the list by choosing the radio button


Select Deploy to device.


![choose project-edited-just picture](https://user-images.githubusercontent.com/11222214/38657988-eb9d98b6-3dd7-11e8-8c94-7273fcfa6e1b.jpg)

On the Target device screen, choose your device from the list, and select **Review.**

![target device](https://user-images.githubusercontent.com/11222214/38658011-088f81d2-3dd8-11e8-972a-9342b7b3e291.JPG)

Select Deploy.

![review deploy](https://user-images.githubusercontent.com/11222214/38658032-223db2e8-3dd8-11e8-9bdf-04779cd0e0e6.JPG)

On the AWS DeepLens console, you can track the progress of the deployment. It can take a few minutes to transfer a large model file to the device. Once the project is downloaded, you will see a success message displayed and the banner color will change from blue to green.

## View Output

<details> <summary>Deeplens connected over monitor setup </summary>
  
 **Option 1: View over mplayer**
 
To view the output, open a terminal (on the Deeplens desktop UI, choose the top left button and search for terminal) and enter the following command:

`mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg`

 
**Option 2: View over browser**

Step 1- From the left navigation, Choose Devices. Choose your device. Choose View Output

![view output](https://user-images.githubusercontent.com/11222214/41580046-41fab7d8-734e-11e8-8e1f-74e772f4f520.JPG)

Step 2- Choose Firefox browser for Windows and Mac. Follow the instructions

![step 1 view output](https://user-images.githubusercontent.com/11222214/41580333-67a45326-734f-11e8-9219-503499a118dc.JPG)

Step 3- Open a browser tab and navigate to https://0.0.0.0:4000

View output and enjoy!
</details>

<details> <summary>Connected over headless mode and using browser on laptop/desktop </summary>
 
 Step 1- From the left navigation, Choose Devices. Choose your device. Choose View Output

![view output](https://user-images.githubusercontent.com/11222214/41580046-41fab7d8-734e-11e8-8e1f-74e772f4f520.JPG)

Step 2- Choose your browser. Follow the instructions

![step 1 view output](https://user-images.githubusercontent.com/11222214/41580333-67a45326-734f-11e8-9219-503499a118dc.JPG)

Step 3- Click on **View stream**

View the output and enjoy!
</details>

<details> <summary>Connected over headless mode and using SSH </summary>

**Option 3: View over SSH **

if you are accessing Deeplens over SSH then use following command over SSH

`ssh aws_cam@$ip_address cat /tmp/\*results.mjpeg |mplayer –demuxer lavf -cache 8092 -lavfdopts format=mjpeg:probesize=32 -`

For streaming over SSH you may need to install mplayer on your laptop by

`sudo apt-get install mplayer`

</details>

Please visit https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-viewing-device-output-on-device.html for more options to view the device stream

