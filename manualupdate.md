### Step1: Update os and install awscam

```
sudo apt update

sudo apt install awscam
```

### Step2: check deeplens software version - make sure it shows version 1.3.14+
``` 
dpkg -l awscam
```

### Step3: reset the device by pressing pin in reset (back panel), wait 10 sec. 

### Step4: Open Firefox browser and type http://deeplens.config (make sure its not www.deeplens.config)

### if site doesnt show up then restart softap/local webserver by following command

```
sudo systemctl restart softap.service
```

