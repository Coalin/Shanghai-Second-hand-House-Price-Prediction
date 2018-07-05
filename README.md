# UI of Shanghai Second-hand House Price Prediction [![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.md)
This is a Course Project for Data Mining in School of Data Science, Fudan University.</br>

### Requirements: 
* **Tensorflow r1.0.1**
* Python 3.6

### Training data:
Crawled from [Centanet](http://sh.centanet.com/) at April 2018.</br>

### Model:
Structure: 2 Hidden-layer Full Connect Neural Network with 200 nodes each, as well as a Dropout with keep ratio of 75%.</br> 
Epoch: 10000</br>
MSE(on testing set): < 0.03

### Example:
A simple example can be seen as follows, witch means the fair price of the dormitory we lived may worth the value up to ￥5.8 million.</br> 
The output includes the average price and the total price of the target real easte.</br>
</br>
![image](https://github.com/Coalin/User-Interface-of-Shanghai-Second-hand-House-Price-Prediction/blob/master/Images/Example.jpg)
