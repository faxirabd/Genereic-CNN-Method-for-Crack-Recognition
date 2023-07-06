# Genereic-CNN-Method-for-Crack-Recognition
This research project utilizes CNN models based on transfer learning for the recognition of cracks in glass panels and concrete surfaces. Specifically, two models, VGG16 and ResNet50, pre-trained on the ImageNet dataset with 1000 classes of natural images, are adopted for this task. There are two main approaches for transfer learning:

(1) Using a pre-trained model as a feature extractor: In this approach, the convolutional layers of the pre-trained model are employed for feature extraction without retraining the model. The resulting feature map is then fed into new fully connected layers designed specifically for the crack recognition task. The new fully connected layers perform classification into two labels: "cracked" and "non-cracked".

(2) Transfer learning through fine-tuning: This approach involves retraining (fine-tuning) some or all of the pre-trained model's convolutional layers, starting from the weights learned during a previous task through transfer learning. By incorporating both the previously learned features and the newly learned features through fine-tuning, the model aims to effectively classify the new crack recognition task.

For this project, the transfer learning based on feature extraction is used due to its speed and high performance for the intended task. Retraining is done only on the fully connected layers. The fully connected layers in the model comprise flatten, dense, and dropout layers, as illustrated in the following figure.

![image](https://github.com/faxirabd/Genereic-CNN-Method-for-Crack-Recognition/assets/115953037/3e226b44-7c37-47ff-b450-0eabb52127de)
