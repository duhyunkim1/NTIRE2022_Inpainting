# Test

Edit your data and checkpoint directory of `test.py` and run it. Each checkpoint of 4 datasets(Places, WikiArt, ImageNet, FFHQ)is in `ckpt` folder. 
You can download check poionts here : https://drive.google.com/file/d/1lNEEY7-Srf8OQXAfFEGBt1HQVcncicPA/view?usp=sharing

I assume the following structure of data.
```
Image, Mask
- test
|   +-- Completion
|		+- 000000.png
|		+- 000001.png
|			...
|   +-- ThinStrokes
			...
```
# Train

Edit your data directory of `train.py` and run it. 


```
Image, Mask
- train
|  - 000000.png
|  - 000001.png
|  - 000002.png
|  - 000003.png
|		...

