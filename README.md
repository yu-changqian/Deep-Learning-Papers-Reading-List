# Deep-Learning-Papers-Reading-List
## Table of Contents
- [Papers](#papers)
- [Datasets](#datasets)
- [Software and Skills](#software-and-skills)

## Papers
### Recognition
- Handwritten Digit Recognition with a Back-Propagation Network(**LeNet**) [[paper]](http://yann.lecun.com/exdb/publis/pdf/lecun-90c.pdf)
- ImageNet Classification with Deep Convolutional Neural Networks(**AlexNet**) [[paper]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- Deep Sparse Rectifier Neural Networks(**ReLU**) [paper](http://proceedings.mlr.press/v15/glorot11a.html)
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift(**Batch-Norm**) [[paper]](https://arxiv.org/abs/1502.03167)
- Dropout: A Simple Way to Prevent Neural Networks from Overfitting(**Dropout**) [[paper]](http://jmlr.org/papers/v15/srivastava14a.html)
- Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition(**SPP**) [[paper]](https://arxiv.org/abs/1406.4729)
- Very Deep Convolutional Networks For Large-Scale Image Recognition(**VGG**) [[paper]](https://arxiv.org/abs/1409.1556)
- Network In Network
- Highway Networks
- Going Deeper with Convolutions(**GoogleNet**)
- Rethinking the Inception Architecture for Computer Vision(**Inception v3**) [[paper]](https://arxiv.org/abs/1512.00567)
- PolyNet: A Pursuit of Structural Diversity in Very Deep Networks(**PolyNet**) [[paper]](https://arxiv.org/abs/1611.05725)
- PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection(**PVANet**) [[paper]](https://arxiv.org/abs/1608.08021)
- Deep Residual Learning for Image Recognition(**ResNet**)
- Identity Mappings in Deep Residual Networks 
- Wide Residual Networks(**Wide-ResNet**)
- Aggregated Residual Transformations for Deep Neural Networks
- Xception: Deep Learning with Depthwise Separable Convolutions(**Xception**) [[paper]](https://arxiv.org/abs/1610.02357)
- Densely Connected Convolutional Networks(**DenseNet**)
- Squeeze-and-Excitation Networks(**SENet**) [[paper]](https://arxiv.org/abs/1709.01507)
- MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications(**MobileNet**) [[paper]](https://arxiv.org/abs/1704.04861)
- ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices(**ShuffleNet**) [[paper]](https://arxiv.org/abs/1707.01083)

### Detection
- Rich feature hierarchies for accurate object detection and semantic segmentation(**RCNN**)
- Fast R-CNN 
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- DenseBox: Unifying Landmark Localization with End to End Object Detection(**DenseBox**) [[paper]](https://arxiv.org/abs/1509.04874)
- You Only Look Once: Unified, Real-Time Object Detection(**YOLO**) [[paper]](https://arxiv.org/abs/1506.02640)
- SSD: Single Shot MultiBox Detector(**SSD**) [[paper]](https://arxiv.org/abs/1512.02325)
- DSSD : Deconvolutional Single Shot Detector(**DSSD**) [[paper]](https://arxiv.org/abs/1701.06659)
- R-FCN: Object Detection via Region-based Fully Convolutional Networks(**RFCN**) [[paper]](https://arxiv.org/abs/1605.06409)
- Feature Pyramid Networks for Object Detection(**FPN**) [[paper]](https://arxiv.org/abs/1612.03144)
- Mask R-CNN [[paper]](https://arxiv.org/abs/1703.06870)
- Focal Loss for Dense Object Detection(**RetinaNet**) [[paper]](https://arxiv.org/abs/1708.02002)
- RON: Reverse Connection with Objectness Prior Networks for Object Detection(**RON**) [[paper]](https://arxiv.org/abs/1707.01691)
- Deformable Convolutional Networks [[paper]](https://arxiv.org/abs/1703.06211)
- Single-Shot Refinement Neural Network for Object Detection [[paper]](https://arxiv.org/abs/1711.06897)
- Light-Head R-CNN: In Defense of Two-Stage Object Detector [[paper]](https://arxiv.org/abs/1711.07264)

### Segmentation
#### semantic segmentation
- Fully Convolutional Networks for Semantic Segmentation(**FCN**)
- Learning Deconvolution Network for Semantic Segmentation(**Deconv**)
- Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
- Conditional Random Fields as Recurrent Neural Networks(**CRFasRNN**)
- Semantic Image Segmentation via Deep Parsing Network(**DPN**)
- Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation
- Exploring Context with Deep Structured models for Semantic Segmentation
- Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs(**Deeplab v1**)
- DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution,and Fully Connected CRFs(**Deeplab v2**)
- RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation(**RefineNet**)
- Understanding Convolution for Semantic Segmentation(**DUC**)
- Wider or Deeper: Revisiting the ResNet Model for Visual Recognition
- Not All Pixels Are Equal: Difficulty-aware Semantic Segmentation via Deep Layer Cascade
- Loss Max-Pooling for Semantic Image Segmentation
- Pyramid Scene Parsing Network(**PSPNet**)
- Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network(**GCN**)
- Rethinking Atrous Convolution for Semantic Image Segmentation(**Deeplab v3**)
- Global-residual and Local-boundary Refinement Networks for Rectifying Scene Parsing Predictions
- Stacked Deconvolutional Network for Semantic Segmentation(**SDN**)

#### instance segmentation
- Instance-aware Semantic Segmentation via Multi-task Network Cascades(**MNC**) [[paper]](https://arxiv.org/abs/1512.04412)
- Proposal-free Network for Instance-level Object Segmentation [[paper]](https://arxiv.org/abs/1509.02636)
- Learning to Segment Object Candidates(**DeepMask**) [[paper]](https://arxiv.org/abs/1506.06204)
- Learning to Refine Object Segments(**SharpMask**) [[paper]](https://arxiv.org/abs/1603.08695)
- FastMask: Segment Multi-scale Object Candidates in One Shot(**FastMask**) [[paper]](https://arxiv.org/abs/1612.08843)
- Instance-sensitive Fully Convolutional Networks(**Instance-sensitive FCN**) [[paper]](https://arxiv.org/abs/1603.08678)
- Associative Embedding: End-to-End Learning for Joint Detection and Grouping [[paper]](https://arxiv.org/abs/1611.05424)
- Fully Convolutional Instance-aware Semantic Segmentation(**FCIS**) [[paper]](https://arxiv.org/abs/1611.07709)
- Mask R-CNN [[paper]](https://arxiv.org/abs/1703.06870)
- Learning to Segment Every Thing [[paper]](https://arxiv.org/abs/1711.10370)
- MaskLab: Instance Segmentation by Refining Object Detection with Semantic and Direction Features [[paper]](https://arxiv.org/abs/1712.04837v1)

#### fast segmentation
- SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation(**SegNet**)
- ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation(**ENet**)
- ICNet for Real-Time Semantic Segmentation(**ICNet**)

#### video segmentation

#### weakly segmentation
- Weakly- and Semi-Supervised Learning of a Deep Convolutional Network for Semantic Image Segmentation
- BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation
- Constrained Convolutional Neural Networks for Weakly Supervised Segmentation
- Augmented Feedback in Semantic Segmentation under Image Level Supervision
- Webly Supervised Semantic Segmentation
- Object Region Mining with Adversarial Erasing: A Simple Classification to Semantic Segmentation Approach
- Exploiting Saliency for Object Segmentation from Image Level Labels
- Discovering Class-Specific Pixels for Weakly-Supervised Semantic Segmentation

#### saliency
- A Model of Saliency-based Visual Attention for Rapid Scene Analysis [[paper]](https://pdfs.semanticscholar.org/e5ae/9c2093699913a480bc0b25c3cd3b958a6b18.pdf)
- Saliency Detection: A Spectral Residual Approach
- Large-Scale Optimization of Hierarchical Features for Saliency Prediction in Natural Images(**eDN**)
- SALICON: Reducing the Semantic Gap in Saliency Prediction by Adapting Deep Neural Networks
- SALICON: Saliency in Context Ming
- Recurrent Attentional Networks for Saliency Detection
- DHSNet: Deep Hierarchical Saliency Network for Salient Object Detection
- Deeply supervised salient object detection with short connections
- What do different evaluation metrics tell us about saliency models?
- Deep Level Sets for Salient Object Detection Ping
- Non-Local Deep Features for Salient Object Detection
- A Stagewise Refinement Model for Detecting Salient Objects in Images
- Amulet: Aggregating Multi-level Convolutional Features for Salient Object Detection
- Deep Contrast Learning for Salient Object Detection
- Instance-Level Salient Object Segmentation
- S4Net: Single Stage Salient-Instance Segmentation
- Salient Object Detection: A Survey
- Salient Object Detection: A Benchmark

## Datasets
### Segmentation
- [PASCAL VOC 2012](http://host.robots.ox.ac.uk:8080/pascal/VOC/)
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [MIT ADE 20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [COCO Stuff](http://cocodataset.org/#stuff-challenge2017)

### Saliency
- [MSARA10K](http://mmcheng.net/zh/msra10k/)
- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [SALICON](http://salicon.net/challenge-2017/)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [DUTS](http://saliencydetection.net/duts/)

## Software and Skills
### Framework
- Keras [[docs]](https://keras.io/)
- Caffe [[install&docs]](http://caffe.berkeleyvision.org/)
- Caffe2 [[install&docs]](http://caffe2.ai/)
- PyTorch [[install]](http://pytorch.org/) [[docs]](http://pytorch.org/docs/0.3.0/)
- Mxnet/Gluon [[install&docs]](http://mxnet.incubator.apache.org/)
- TensorFlow [[install&docs]](https://www.tensorflow.org/)

### Skills
- [git](http://rogerdudler.github.io/git-guide/index.zh.html)
- [python tutorial](https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/0013747381369301852037f35874be2b85aa318aad57bda000)
- tmux
- vim
- markdown
- latex
