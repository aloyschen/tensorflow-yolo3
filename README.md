# tensorflow-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)


## tensorflow implementation of YOLOV3

---

## Detection

1、If use the pretrain model, download YOLOV3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).  
2、Modify yolo3_weights_path in the config.py  
3、Run detect.py  

```
wget https://pjreddie.com/media/files/yolov3.weights  
python detect.py --image_file ./test.jpg  
```
![result](https://raw.githubusercontent.com/aloyschen/tensorflow-yolo3/master/result.jpg)


## Training

1、Download the COCO2017 dataset from [COCO_website](http://cocodataset.org)  
2、Modify the train and val data path in config.py  
3、If you want to use original pretrained weights for YOLOv3, rename it as darknet53.weights, and modify the darknet53_weights_path in the config.py 

```
wget https://pjreddie.com/media/files/darknet53.conv.74`  
```  
4、Modify the data augmentation parameters and train parameters  
5、Run yolo_train.py  

## Evaluation
1、Modify the pre_train_yolo3 and model_dir in config.py  
2、Run detect.py  

```
python detect.py --image_file ./test.jpg
```
## Notice

If you want to modify the Gpu index, please modify gpu_index in config.py

## Credit
```
@article{yolov3,
	title={YOLOv3: An Incremental Improvement},
	author={Redmon, Joseph and Farhadi, Ali},
	journal = {arXiv},
	year={2018}
}
```

## Reference
* [keras-yolo3](https://github.com/qqwweee/keras-yolo3)
