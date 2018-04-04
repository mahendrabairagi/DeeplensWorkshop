In this lab session, you will learn how to build and train a model in Amazon SageMaker

### Step 1- Create notebook instance

Go to SageMaker console: https://console.aws.amazon.com/sagemaker

Click on Create Notebook instance


![sagemaker home](https://user-images.githubusercontent.com/11222214/38313489-01929ca2-37d9-11e8-9ffb-4385e8d13da3.JPG)

### Step 2- Notebook specifications

1. Provide the name of the notebook as face-detection
2. Notebook instance type- ml.t2.medium
3. IAM role- Click 'Create a new role'

In the dialog box that opens up:

1. Click 'Any S3 bucket'
2. Leave the rest as is. Click 'Create role'

![create iam role](https://user-images.githubusercontent.com/11222214/38313888-e07281e4-37d9-11e8-8b99-dd322a76ced6.JPG)


Leave VPC, Lifecycle and Encryption key as defaults. Dont make any changes

![notebook instance setting](https://user-images.githubusercontent.com/11222214/38313994-2916257c-37da-11e8-823a-733f2572f61d.JPG)

Click 'Create notebook instance'

### Step 3- View notebook instances

You can view all your notebook instances by choosing Notebook on the left menu. It will take couple of minutes for the notebook instance to be created.

![instances](https://user-images.githubusercontent.com/11222214/38314549-541e9140-37db-11e8-89eb-ec9be1677271.JPG)

### Step 4- Open notebook

You can choose your notebook and click on 'Open'. 

This will open your Jupyter notebook.

![jupyter](https://user-images.githubusercontent.com/11222214/38314946-427aa6e4-37dc-11e8-91bf-658ebe7b2a7b.JPG)

### Step 5- Upload notebook

1. Download the SSD_Object_Detection_SageMaker_v3.ipynb from the github repo

2. Upload the notebook by choosing upload and uploading the file that you just downloaded

3. Click on the uploaded file and execute the cells by clicking on run button or using shift+ enter on your keyboard

![run](https://user-images.githubusercontent.com/11222214/38316244-21a07194-37df-11e8-9821-21d5d6e57976.JPG)

