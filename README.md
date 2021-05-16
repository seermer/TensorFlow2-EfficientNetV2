# EfficientNetV2_TensorFlow2
a TensorFlow2(keras model) reimplementation of EfficientNetV2\
This is NOT an official implementation.\
Now the implementation has been modified to match the # of params and architecture of official repo\
~~As official implementation is not yet released by the time I write this code, this model has not been verified and therefore cannot be guarateed to match exactly with official code~~ 

## efficientnetV2.py
an implementation in TensorFlow2 Keras,\
~~currently, only EfficientNetV2-s is included~~\
EfficientNetV2-s, m, l, xl are now all implemented\
this implementation is based on description from paper\
https://arxiv.org/abs/2104.00298 \
EfficientNetV2: Smaller Models and Faster Training\
by Mingxing Tan, Quoc V. Le\
and official repo\
https://github.com/google/automl/tree/master/efficientnetv2 \
Codes are partially inspired and adapted from official repo\
~~Codes are partially inspired and adapted from TensorFlow.keras.application MobileNet Code~~


## ghost_efficientnetV2.py
a custom version of EfficientNetV2,\
replaced most Convolutional layers with Ghost Modules introduced in paper\
https://arxiv.org/abs/1911.11907 \
GhostNet: More Features from Cheap Operations\
by Han et al.\
Ghost Modules significantly reduces number of parameter in the model

also, instead of ResNet-C downsampling, this version uses ResNet-D downsampling\
see paper https://arxiv.org/abs/1812.01187v2 \
Bag of Tricks for Image Classification with Convolutional Neural Networks\
by He et al.

reduction ratio of SE module is also slightly changed

**with above changes, this custom version has only ~65% of original # of params** \
**However, notice that this custom version has approximately the same or even longer training time comparing to the original version on GPU,** \
this is because of the hardware limitation on DepthWiseConv computation.\
According to the GhostNet paper, mobile devices and other devices with limited resource can benefit from this.\
And there are also some special cases that ghost modules can perform better than normal convs.
