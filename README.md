# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a template for solving ASR task with PyTorch. This template branch is a part of the [HSE DLA course](https://github.com/markovka17/dla) ASR homework. Some parts of the code are missing (or do not follow the most optimal design choices...) and students are required to fill these parts themselves (as well as writing their own models, etc.).

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation

Follow these steps to install the project:

1. Clone the repository:

   ```bash
   git clone https://github.com/alekseevpavel04/s_hw2.git
   ```

2. Navigate to the project directory:

   ```bash
   cd s_hw2
   ```

3. Create a virtual environment:

   ```bash
   python3 -m venv venv
   ```

4. Install PyTorch (to avoid compatibility issues):

   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

5. Install the remaining dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Download the necessary data (models, language models, and vocabulary):

   ```bash
   sh ./download_all_we_need.sh
   ```

## Quick Inference with Metrics Calculation

Run the following command to perform inference and calculate metrics:

```bash
python inference.py
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
