### DDMDN: Diffusion-Based Trustworthy Human Trajectory Prediction

 !["Screenshot..."](images/ddmdn_examples.png "Screenshot...")


#### This is the official implementation for: 

_**Diffusion-Based Trustworthy Human Trajectory Prediction with Dual Mixture Density Networks and Uncertainty Self-Calibration**_

**ICRA 2026 Paper:** [![IEEE Badge](https://img.shields.io/badge/IEEE-00629B?logo=ieee&logoColor=fff&style=for-the-badge)](https://doi.org) [![arXiv Badge](https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&logoColor=fff&style=flat)](https://arxiv.org) [![ResearchGate Badge](https://img.shields.io/badge/ResearchGate-0CB?logo=researchgate&logoColor=fff&style=flat)](https://www.researchgate.net)

_**Abstract**_ --  Human Trajectory Forecasting (HTF) aims to predict future human movements from past trajectories and environmental context, with applications in Autonomous Driving (AD), Smart Surveillance, and Human-Machine Interaction. While prior work emphasizes forecasting accuracy, social interaction handling, and hypothesis diversity, no assessment is made of the correctness of distribution uncertainty and calibration. Furthermore, the accuracy of forecasts at short observation periods is often not evaluated. However, downstream tasks, such as AD path planning and robotic collision avoidance, require fast, reliable, and calibrated forecasts with meaningful uncertainty estimates. We introduce an end-to-end probabilistic HTF model (DD-MDN) that unifies positional accuracy and reliable uncertainty handling. Built on a few‑shot denoising diffusion backbone, a dual Mixture Density Network generates self-calibrated future residence areas with probability‑ranked anchor paths, from which diverse discrete trajectory hypotheses are derived. This formulation ensures self‑calibrated uncertainty, multimodality, and accurate probability assignments, all learned directly from data without predefined anchors or endpoints. Experiments on the ETH/UCY, SDD, inD, and IMPTC datasets demonstrate state-of-the-art positional accuracy and robustness under short observation periods, enabled by reliable, calibrated uncertainty modeling.

**Cite:**

    @article{ddmdn_2026,
        title={Diffusion-Based Trustworthy Human Trajectory Prediction with Dual Mixture Density Networks and Uncertainty Self-Calibration},
        author={Hetzel, M. and Turcan, K. and Reichert, H. and Doll, K. and Sick, B.},
        journal={IEEE International Conference on Robotics and Automation (ICRA)},
        year={2026},
        organization={IEEE Robotics and Automation Society},
        doi={tba}
    }

![til](./images/collage_1.gif)
![til](./images/collage_2.gif)

_**Architecture**_ -- DD-MDN is an end-to-end framework that consists of three parts (Encoding, Probabilistic Modeling, and Deterministic Hypotheses Generation). An architectural overview is illustrated by the figure above. Classic encoder networks encode input data; LSTMs are used for temporal inputs (past agent motion); CNNs are used for spatial data (map information); and transformers are used for self- and social-attention tasks. Probabilistic modeling processes temporal and spatial input features and is achieved by a dual MDN, consisting of a shared denoising diffusion backbone and three probabilistic heads that derive two types of distributional representations: a per-timestep and a per-anchor-trajectory representation. The architecture can handle various continuous probability distributions, including Gaussian, Laplace, Cauchy, and others. For this work, we use Gaussian Mixtures (GM). The per-timestep representation is necessary for trustworthy uncertainty modeling and calibration. The future-state distribution at each time point is represented by a multimodal GM. The per-anchor-trajectory representation is necessary for natural and realistic discrete hypothesis generation, yielding fully time-stable, individually weighted M future trajectory distributions in trajectory space. The individual core parameters' means and variances are shared across the two representations; however, their arrangements differ between them. The weights are separated. Deterministic modeling processes temporal and social input features to generate K uncertainty-related discrete future-trajectory hypotheses via affine reparameterization sampling. A detailed description of the architecture is provided in the full paper.

!["Screenshot..."](images/ddmdn_arch.png "Screenshot...")

---
### Table of contents:
* [Quick Start](#quick_start)
* [Requirements](#requirements)
* [Datasets](#datasets)
* [Pretrained Models](#pretrained)
* [Training](#training)
* [Evaluation](#evaluation)
* [License](#license)

---
<a name="quick_start"></a>
### Quick Start


**1.) Pull docker image:**
```bash
docker pull asterix19/ddmdn:1.0 
```

**2.) Download preprocessed datasets and pretrained models:**
--> [Download DDMDN Data](https://drive.google.com/drive/folders/1CVIPTMtpfind3CArV0yvZ_zT_UHOWoTN?usp=sharing)

```bash
# Create a workspace folder on your local machine
mkdir .../ddmdn_workspace/

# Extract "benchmarks", "full", and "trained_models" directories to:
.../ddmdn_workspace/datasets/

# Result:
.../ddmdn_workspace/datasets/benchmarks
.../ddmdn_workspace/datasets/full
.../ddmdn_workspace/datasets/trained_models
```

**3.) Clone repository:**
```bash
Download from: https://anonymous.4open.science/r/ddmdn-45F6

# Extract repository to:
.../ddmdn_workspace/repos/

# Result:
.../ddmdn_workspace/repos/ddmdn
```

**4.) Set paths and user id in user.env:**
```bash
# User.env file path:
cd .../ddmdn_workspace/repos/ddmdn/docker

# Define:
USER_ID=...
DATA_PATH='/.../ddmdn_workspace/datasets/'            ---> Is mapped in the container to: /workspace/data
REPOS_PATH='/.../ddmdn_workspace/repos/ddmdn'      ---> Is mapped in the container to: /workspace/repos
```

**5.) Start the container:**
```bash
cd .../ddmdn_workspace/repos/ddmdn/docker
docker compose --env-file user.env build up -d
```

**6.) Connect to the container:**
Connect with your preferred IDE or via terminal to the Docker container, and you're ready to go. To run an evaluation or training, see [Training](#training) and [Evaluation](#evaluation) sections below.

```bash
#--- ETH pretrained model test ---#
# Attach to container
docker exec -it ddmdn_framework_dev bash

# Switch to test script directory
cd /workspace/repos/framework/ddmdn

# Run ETH evaluation of pretrained model
python3 test.py --cfg=benchmarks/hotel_benchmark.json --gpu=0 --print --bar 
```


---
<a name="requirements"></a>
### Requirements

The framework uses the following system configuration. All specific Python requirements are listed in the corresponding requirements.txt file and are already pre-installed in the provided Docker container.

```
# Software
Ubuntu 22.04 LTS
Python 3.10
Pytorch 2.5.0
CUDA 12.6
CuDNN 9.5

# Used Hardware:
Ryzen 9950X
32 GB RAM
RTX 4090
```

### Docker
**From scratch:** We used and recommend the Nvidia NGC Pytorch 24.10 image. Run the image and install the corresponding Python requirements.txt, and you're ready to go.
- [[Nvidia NGC Pytorch 24.10]](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-10.html) [[Nvidia NGC Pytorch Release Notes]](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

**Prebuild:** Simply pull our prebuild Docker image from Docker Hub, attach your preferred IDE (we use Visual Studio Code), and you're ready to go.

- [[DDMDN Official Docker Image]](https://hub.docker.com/r/asterix19/ddmdn)
```bash
docker pull asterix19/ddmdn:1.0 
```

---
<a name="datasets"></a>
### Datasets
We use pedestrian trajectory data from multiple popular road traffic and surveillance datasets. 

#### Raw Data:
Below we list the publications and raw data sources:

**SDD:** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_33) [[Data]](https://cvgl.stanford.edu/projects/uav_data/)


**IMPTC:** [[Paper]](https://ieeexplore.ieee.org/document/10186776) [[Data]](https://github.com/kav-institute/imptc-dataset)


**inD:** [[Paper]](https://ieeexplore.ieee.org/document/9304839) [[Data]](https://github.com/ika-rwth-aachen/drone-dataset-tools)


**ETH/UCY:** [[Paper]](https://ieeexplore.ieee.org/document/5459260) [[Data]](https://paperswithcode.com/dataset/eth)


#### Preprocessed Data: 
Already preprocessed datasets can loaded from here: [[Download Preprocessed Datasets]](https://drive.google.com/drive/folders/1CVIPTMtpfind3CArV0yvZ_zT_UHOWoTN?usp=sharing)
(Due to copyright, we cannot provide data from the inD dataset directly; reach out to the authors for access.)

#### Dataset Preprocessing Scripts:
To self-preprocess the raw data, we also provide dataset preprocessing scripts in the code repository under:
```bash
../framework/datasets/
``` 

#### Final Data Structure: 
The final data structure is as follows. Extract and move the dataset, pretrained model weights, and code repository to this structure.
```bash
../ddmdn/
        ├── repos
        │   ├── framework
        │       ├── datasets
        │       ├── ddmdn
        │   ├── docker
        ├── datasets
        │   ├── benchmarks
        │       ├── eth
        │           ├── database
        │           ├── masks
        │           ├── pkl
        │           ├── raw
        │           ├── topview
        │       ├── hotel
        │       ├── univ
        │       ├── zara01
        │       ├── zara02
        │       ├── sdd
        │       ├── ind
        │
        │   ├── full
        │       ├── imptc
        │           ├── database
        │           ├── masks
        │           ├── raw
        │           ├── topview
        │       ├── ind
        │       ├── sdd
        ├── trained_models
        │   ├── benchmarks
        │   ├── full
``` 

---
<a name="pretrained"></a>
### Pretrained Models
Pretrained models can loaded from here: [[Download Pretrained Models]](https://drive.google.com/drive/folders/1CVIPTMtpfind3CArV0yvZ_zT_UHOWoTN?usp=sharing)

**The final result/checkpoint structure must look like this:**
```bash
../trained_models/
                  │── benchmarks
                  │   ├── eth_benchmark
                  │       ├── checkpoints
                  │           ├── weights.pt
                  │       ├── eth_benchmark.json
                  │   ├── hotel_benchmark
                  │   ├── univ_benchmark
                  │   ├── zara01_benchmark
                  │   ├── zara02_benchmark
                  │   ├── sdd_benchmark
                  │   ├── ind_benchmark
                  │
                  │── full
                  │   ├── imptc_full
                  │       ├── checkpoints
                  │           ├── weights.pt
                  │       ├── imptc_full.json
                  │   ├── ind_full
                  │   ├── sdd_full
``` 

---
<a name="evaluation"></a>
### Model Evaluation
To run a model evaluation, you can use the following command. All relevant information is provided in the configuration .json file. It contains all necessary paths, parameters, and configurations for training and evaluation. For every dataset type, one can create an unlimited number of different configuration files. After training, the configuration file is copied to the results directory alongside the checkpoints. The test script explicitly references this configuration file.
**Keep in Mind:** Within the config file, the dataset paths must match the correct data locations!
```bash
# Start an evaluation using the ETH benchmark dataset with the default config:

cd ../framework/ddmdn
python3 test.py --cfg=benchmarks/eth_benchmark.json --gpu=0 --print --bar

# Arguments:
--cfg: target dataset specific config in configs-directory
--gpu: gpu id to be used for the evaluation
--bar: Show evaluation progress as a progress bar
--print: Show evaluation feedback information in console
```

---
<a name="training"></a>
### Model Training
To run a training session, use the following command. All relevant information is provided in the configuration .json file. It contains all necessary paths, parameters, and configurations for training and evaluation. For every dataset type, one can create an unlimited number of different configuration files.
**Keep in Mind:** Within the config file, the dataset paths must match the correct data locations and the checkpoint name!
```bash
# Start a training using the ETH benchmark dataset with the default config:

cd ../framework/ddmdn
python3 train.py --cfg=benchmarks/eth_benchmark.json --gpu=0 --print --bar

# Arguments:
--cfg: target dataset specific config in configs-directory
--gpu: gpu id to be used for the training
--bar: Show training progress as a progress bar
--print: Show training feedback information in console
```

---
<a name="license"></a>
## License:
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details
