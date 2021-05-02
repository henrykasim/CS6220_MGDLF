# CS6220_MGDLF
Multi-Granularity Deep Learning Framework

Dataset: https://drive.google.com/file/d/1YsE7zPZFPy1cF-0vcFzToB2LJP61PLWX/view?usp=sharing
<br> Place the unzip csv file in data folder


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This is implementation for Multi-Granularity Deep Learning Framework (MGDLF) for crowd flow prediction. The contribution of this project in the following aspects:
* Variant of GCN to capture spatial correlations between different regions,
* Use of GRUs to capture temporal interactions between different temporal trend,
* Multi-granularity fusion module based on Fully-connected Neural Network (FNN) to fuse multiple latent representations from different temporal granularity.


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
List of Python library required, as listed in requirements.txt
```sh
   pip install -r requirements.txt
```

### Installation

Clone the repo
   ```sh
   git clone https://github.com/henrykasim/CS6220_MGDLF.git
   ```


<!-- USAGE EXAMPLES -->
## Usage

To run MGDLF run main.py
   ```sh
   python main.py
   ```

The list of parameters are:
* --gpu - for using gpu. default 0
* --batch_size - training batch size. default 32
* --epochs - epoch value. default 200
* --hidden - number of hidden GRU units. default 16
* --interval - number of time steps. default 5
* --weight-decay - weight decay. default 5e-4
* --lr - learning rate. default 1e-2
