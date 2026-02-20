🌐 Languages: [[English](README.md)] | [[한국어](README-KR.md)]

# CephLD-CCA: Cephalometric Landmark Detection with Cartesian Coordinate Channel Attention
*(데카르트 좌표 기반 채널 어텐션을 적용한 두부 측면 X-ray 계측점 자동 검출 네트워크)*


CephLD-CCA는 두부 측면의 X-ray(cephalogram) 이미지를 입력으로 받아 계측점(cephalometric landmark)의 위치를 자동으로 추정하는 딥러닝 기반 모델입니다. 특히, 채널 어텐션 기법인 SE(Squeeze-and-Excitation)-block을 변형하여 데카르트 좌표(Cartesian coordinate) 기반의 채널 어텐션(Channel Attention) 블록인 CCA를 설계했으며, 이를 통해 계측점 검출 성능 향상을 목표로 합니다.


## 🏆 Awards
### 수상
- **대회명**: 2021 SW중심대학 융합SW 교육원 AI 경진대회
- **기간**: 2021.06 - 2021.07
- **주최**: 과학기술정보통신부
- **수상**: 🥇 **1등상**


## ⚙️ Tech Stacks
- U-Net
- SE (Squeeze-and-Excitation) / Channel Attention
- PyTorch
- Python
- CUDA
- OpenCV


## ✨ Features
1. **데카르트 좌표 기반 채널 어텐션 모듈(CCA) 설계**
2. 좌표 정보를 채널 어텐션에 반영하여 **landmark의 미세한 위치 단서를 강화**
3. Vanilla U-Net 및 SE 어텐션 기반 U-Net 대비 **더 높은 계측점 검출 성능 달성**


## 🧭 Overview
<img src="https://github.com/MonoHaru/CephLD-CCA/blob/main/assets/overview-kr.png" alt="process" width="700">


## 🚀 Train
#### Vanilla U-Net 학습
`python train_unet.py`

#### SE 채널 어텐션을 활용한 U-Net 학습
`python train_unet_w_se.py`

#### Cartesian Coordinate 기반 채널 어텐션을 활용한 CephLD-CCA 학습
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
1. 현재 배치 사이즈를 1로 고정하고 배치 정규화(batch normalization)를 사용하지 않아 학습이 불안정해질 수 있으므로, 배치 사이즈를 늘리고 정규화를 도입하여 학습 안정성을 확보
2. Vanilla U-Net 기반 구조를 확장/변형하여 랜드마크 검출 성능을 추가로 향상
3. 더 많은 데이터 확보 또는 데이터 증강 기법을 적용하여 일반화 성능 향상
4. U-Net의 무겁고 느린 추론 시간을 개선하기 위해서 지식 증류(knowledge distillation) 등을 통해 성능을 유지하면서 추론 시간 최적화


## 📜 License
The code in this repository is released under the GPL-3.0 License.