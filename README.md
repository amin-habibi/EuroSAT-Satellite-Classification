🛰️ EuroSAT Image Classification

A deep learning project for satellite image classification and land cover recognition using the EuroSAT RGB dataset and a pretrained ResNet18 model.

The goal of this project is to build an image classification model capable of recognizing different types of land cover from satellite imagery.

📌 Overview

Remote sensing and satellite imagery provide valuable information for monitoring and understanding the Earth's surface.

In this project, a transfer learning approach is used to classify satellite images into 10 different land cover categories.

The model is based on ResNet18 pretrained on ImageNet, with the final classification layer modified to predict the 10 EuroSAT classes.

Key Features
🛰️ EuroSAT RGB satellite imagery
🧠 ResNet18 with transfer learning
🔟 10 land cover classes
🔄 80/20 training-validation split
⚡ GPU acceleration with CUDA when available
📊 Validation accuracy evaluation
💾 Model weight saving
🛰️ Dataset

The project uses the EuroSAT RGB dataset, which contains satellite images covering different types of land use and land cover.

The dataset contains 10 classes:

Class	Description
AnnualCrop	Annual agricultural crops
Forest	Forest areas
HerbaceousVegetation	Herbaceous vegetation
Highway	Highways and roads
Industrial	Industrial areas
Pasture	Pasture land
PermanentCrop	Permanent crops
Residential	Residential areas
River	Rivers
SeaLake	Seas and lakes

The images are resized to 64 × 64 pixels before being passed to the model.

🧠 Model

The classification model is based on ResNet18, a convolutional neural network architecture using residual connections.

Instead of training the entire network from scratch, a pretrained ImageNet model is used and its final fully connected layer is replaced with a new layer containing 10 output classes.
