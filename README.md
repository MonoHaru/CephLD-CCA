# CephLD-CCA: Cephalometric Landmark Detection with Cartesian Coordinate Channel Attention
*(A cephalometric landmark detection network for lateral skull X-ray images using Cartesian coordinate–based channel attention)*

CephLD-CCA is a deep learning–based model that takes lateral skull X-ray (cephalogram) images as input and automatically estimates the locations of cephalometric landmarks. In particular, we redesign the SE (Squeeze-and-Excitation) block, which is a channel attention technique, and design CCA as a Cartesian coordinate-based channel attention block. Through this design, the model aims to improve landmark detection performance.


## 🏆 Awards
### Awards
- **Competition**: 2021 SW-Centered University Convergence SW Education Center AI Competition
- **Period**: 2021.06 - 2021.07
- **Host**: Ministry of Science and ICT
- **Award**: 🥇 **1st Prize**


## ⚙️ Tech Stacks
- U-Net
- SE (Squeeze-and-Excitation) / Channel Attention
- PyTorch
- Python
- CUDA
- OpenCV


## ✨ Features
1. **Design of a Cartesian coordinate-based channel attention module (CCA)**
2. Incorporate coordinate information into channel attention to **enhance subtle positional cues for landmarks**
3. Achieve **higher landmark detection performance** compared to vanilla U-Net and SE attention-based U-Net


## 🧭 Overview
<img src="https://github.com/MonoHaru/CephLD-CCA/blob/main/assets/overview.png" alt="process" width="700">


## 🚀 Train
#### Train Vanilla U-Net
`python train_unet.py`

#### Train U-Net with SE channel attention
`python train_unet_w_se.py`

#### Train CephLD-CCA with Cartesian coordinate-based channel attention
`python train_unet_w_cartesian_se.py`


## 🛠️ Train Experimental Settings
- Optimizer: Adam
- Learning Rate: 1e-10
- Learning Rate Scheduler: CosineAnnealingWarmUpRestarts
- Loss function: L2 loss
- Batch size: 1


## 🧪 Test
`python val_test.py`


## 🎯 Results
#### Table 1. Compared deteciton performance wit Vanilla U-Net, SE U-Net, and CephLD-CCA
| Model | Error Rate ↓ |
| :------ | :---: |
| Vamilaa U-Net | 0.0053 |
| U-Net w/ SE | 0.0008 |
| CephLD-CCA (Ours) | 0.0006 |


## 🔮 Future Work
1. The batch size is currently fixed at 1 and batch normalization is not used, so training can be unstable. Increase the batch size and introduce normalization to improve training stability.
2. Further improve landmark detection performance by extending or modifying the vanilla U-Net-based architecture.
3. Improve generalization by collecting more data or applying data augmentation techniques.
4. Optimize inference time while maintaining performance. Knowledge distillation can be an option, since U-Net inference can be heavy and slow.


## 📜 License
The code in this repository is released under the GPL-3.0 License.