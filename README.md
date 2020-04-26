# BlazeFace

The code contains fully implemented SDD detection framework with BlazeFace network as the backbone. Currently, the code is taking random inputs in the batches, this was done so as to check the  working of the architecture. Hence the all the files are complete in their implementation except utils.py, which is to be modified according to a particular dataset.<br/>

network.py contains the implementation of the BlazeFace architecture. <br />
loss.py contains the implementation of smooth l1 loss. <br />
main.py contains the main training loop.<br />
utils.py contains utility functions for parsing the dataset and creating the inputs and targets for the network.<br />

Requirements:<br />
python 3.6<br />
tensorflow 1.14.0<br />
numpy <br />
cv2 <br />
Pillow <br />
