# STCCR model

We give three datasets (weeplace gowalla-tky and gowalla-nyc ).

### Dataset link: 

OneDrive Disk:

https://lyrics5816-my.sharepoint.com/:f:/g/personal/9732_officehao_vip/ErMkz7RQpDhArpOsYaAicp0BT0BR-rSPshM91zOITEsclg?e=UXkccW

Google Disk:

https://drive.google.com/drive/folders/1SMmpyCuL3tnmpHfEW-dIKODT_U5uY93g?usp=drive_link

Baidu Disk:
https://pan.baidu.com/s/1DPsIBfU-J9aPY0VkR-86wg
password：gce2

Please download the dataset to the directory ./STCCR_data 

We have different configurations for three missions.

- ### model training (POI):
  
  - 1. Enter "STCCR-main" (root directory).
  - 2. Run "python train_STCCR.py --config config/STCCR_wee_POI.conf --dataroot ./STCCR_data/"
- ### model training (TUL):
  
  - 1. Enter "STCCR-main" (root directory).
  - 2. Run "python train_STCCR.py --config config/STCCR_wee_TUL.conf --dataroot ./STCCR_data/"
- ### model training (TP):
  
  - 1. Enter "STCCR-main" (root directory).
  - 2. Run "python train_STCCR_TP.py --config config/STCCR_wee_TP.conf --dataroot ./STCCR_data/"
