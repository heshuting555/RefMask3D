# RefMask3D: Language-Guided Transformer for 3D Referring Segmentation

**[📄[arXiv]](https://arxiv.org/abs/2407.18244)**  &emsp; **[📄[PDF]](https://arxiv.org/pdf/2407.18244)** 

This repository contains code for ACM MM 2024 paper:

> [RefMask3D: Language-Guided Transformer for 3D Referring Segmentation](https://arxiv.org/abs/2407.18244)  
> Shuting He,  Henghui Ding  
> ACM MM 2024

## Code structure

We adapt the codebase of  [Mask3D](https://github.com/JonasSchult/Mask3D) which provides a highly modularized framework for 3D instance Segmentation based on the MinkowskiEngine.

```
RefMask3D
├── benchmark                     <- evaluation metric
├── conf                          <- hydra configuration files
├── datasets
│   ├── preprocessing             <- folder with preprocessing scripts
│   ├── semseg.py                 <- ScanRefer dataset loader
│   └── utils.py
├── models                        <- RefMask3D modules
├── trainer
│   ├── __init__.py
│   └── trainer.py                <- train loop
└── utils
├── data
│   └──processed                  <- folder for preprocessed ScanNet and ScanRefer					
├── scripts                       <- train scripts
├── README.md
└── saved                         <- folder that stores models and logs
```

### Dependencies :memo:

The main dependencies of the project are the following:

```yaml
python: 3.10.9
cuda: 11.7
torch: 1.13.1
```

You can set up a conda environment as follows, following [Mask3D](https://github.com/JonasSchult/Mask3D)

```
conda create -n refmask3d python=3.10 
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install "cython<3.0.0" && pip install --no-build-isolation pyyaml==6.0.1
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

mkdir third_party
cd third_party

git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas

cd ..
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet/Segmentator
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make

cd ../../pointnet2
python setup.py install

cd ../../
pip install pytorch-lightning==1.7.2
```

### Data preprocessing :hammer:

After installing the dependencies, You need to download [ScanNet](https://github.com/ScanNet/ScanNet) and [ScanRefer](https://github.com/daveredrum/ScanRefer) datasets. Then, we preprocess the ScanNet datasets and download mask3d checkpoint.

```
mkdir -p checkpoints/scannet && cd checkpoints/scannet
wget https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet/scannet_val.ckpt

python -m datasets.preprocessing.scannet_preprocessing preprocess \
--data_dir="PATH_TO_RAW_SCANNET_DATASET" \
--save_dir="data/processed/" \
--git_repo="third_party/ScanNet/" \
--scannet200=False
```

### Training and testing :train2:

Train and Test RefMask3D on the ScanNet dataset:

```bash
sh scripts/refmask3d.sh
```
Note: We train on a A6000 machine (48G) using 8 cards with 4 sample on each card, taking about 18 hours.
If you use other cards, you may need to change batch size and learning rate.

## Trained checkpoints :floppy_disk:
TBD
☁️ [Google Drive]()

## BibTeX :pray:
Please consider to cite RefMask3D if it helps your research.

```bibtex
@inproceedings{RefMask3D,
  title={{RefMask3D}: Language-Guided Transformer for 3D Referring Segmentation},
  author={He, Shuting and Ding, Henghui},
  booktitle={ACM MM},
  year={2024}
}
```

