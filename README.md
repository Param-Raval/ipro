# ipro

This project implements three services using Computer Vision: shadow detection & removal ([ShadowSight](https://github.com/Param-Raval/shadow-sight)), pose detection, and visual question-answering.

## Run

### Install requirements
`pip install -r final_requirements.txt`

Note: You don't necessarily need GPUs since this project uses pretrained models and checkpoints. However, if you wish to use GPUs, you would need to install the respective PyTorch and torchvision wheel versions from [here](https://download.pytorch.org/whl/torch_stable.html). Copy the wheel links from the PyTorch website and replace the PyTorch and torchvision entries in `final_requirements.txt` with them.

### Download Pre-trained models
You can download checkpoints for ShadowSight from [here](https://drive.google.com/drive/folders/1J1l21k5AoUXHxic-Bj3eXBFP--YzjFXO?usp=sharing). Download and place them inside _app_ in a folder named _**checkpoints**_.

For the rest of the services, download the pre-trained weights from here (download the entire folder) and place them inside _app_ in a folder named _**pickles**_.

### Run App
`python run.py`

While all paths in the scripts are relative, in case of any discrepancies look through _init_.py, test.py, and views.py in the _app_ folder and change values for path variables accordingly.

## Implementations
* https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation
* https://github.com/IsHYuhi/ST-CGAN_Stacked_Conditional_Generative_Adversarial_Networks
* https://github.com/jiasenlu/HieCoAttenVQA

## References
* Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal, Jifeng Wang<sup>∗</sup>, Xiang Li<sup>∗</sup>, Le Hui, Jian Yang, **Nanjing University of Science and Technology**, [[arXiv]](https://arxiv.org/abs/1712.02478)

* Lu, Jiasen, et al. "Hierarchical question-image co-attention for visual question answering." Advances in neural information processing systems. 2016.

* Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh, ‘‘Realtime multi-person 2D pose estimation using part affinity fields,’’ in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Jul. 2017, pp. 1302–1310.
