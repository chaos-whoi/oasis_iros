[comment]: <> (# OASIS: Real-Time Opti-Acoustic Sensing for Intervention Systems in Unstructured Environments)

<p align="center">
  <h1 align="center">OASIS: Real-Time Opti-Acoustic Sensing for Intervention Systems in Unstructured Environments</h1>
  <p align="center">
    <a href="https://amyphung.github.io/"><strong>Amy Phung</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=NBVg87gAAAAJ&hl=en"><strong>Richard Camilli</strong></a>
  </p>

<h3 align="center"><a href="https://youtu.be/8Vq9NFSO1cE">Video</a> | <a href="https://oasis-iros.github.io/">Project Page</a></h3>
<div align="center"></div>

<br>

<!-- [comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://arxiv.org/abs/2412.12392">Paper</a> | <a href="https://youtu.be/wozt71NBFTQ">Video</a> | <a href="https://edexheim.github.io/mast3r-slam/">Project Page</a></h3>
  <div align="center"></div>

<p align="center">
    <img src="./media/teaser.gif" alt="teaser" width="100%">
</p>
<br> -->

# Getting Started
## Installation

Clone the repo and install the dependencies

```bash
git clone git@github.com:chaos-whoi/oasis.git
cd oasis
conda env create -f environment.yaml
conda activate oasis
```

## Running example on dataset
```bash
python reconstruct.py  --config configs/tank.yaml
python fuse_camera.py  --config configs/tank.yaml
```
