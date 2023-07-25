# isoCycle
[![preprint](https://img.shields.io/badge/preprint-bioRxiv-blue)]()
[![pip](https://img.shields.io/badge/pip%20install-PyPI-orange)](https://pypi.org/project/isoCycle/)
### A Deep Network-Based Decoder for Isolating Single Cycles of Neural Oscillations in Spiking Activity

<!-- <p align="justify"> -->
Neural oscillations are prominent features of neuronal population activity in the brain, manifesting in various forms such as frequency-specific power changes in electroencephalograms (EEG) and local field potentials (LFP), as well as phase locking between different brain regions, modulated by modes of activity. Despite the intrinsic relation between neural oscillations and the ***spiking*** activity of single neurons, identification of oscillations has predominantly relied on indirect measures of neural activity like EEG or LFP, overlooking direct exploration of oscillatory patterns in the ***spiking*** activity, which serve as the currency for information processing and information transfer in neural systems. Recent advancements in densely recording large numbers of neurons within a local network have enabled direct evaluation of changes in network activity over time by examining population spike count variations across different time scales. Here we leverage the power of deep neural networks to robustly isolate single cycles of neural oscillations from the spiking of densely recorded populations of neurons. isoCycle effectively identifies individual cycles in the temporal domain, where cycles from different time scales may have been combined in various ways to shape spiking probability. The reliable identification of single cycles of neural oscillations in spiking activity across various time scales will deepen our understanding of the dynamics of neural activity.
<!-- </p> -->

<p align="center">
  <img alt="https://github.com/esiabri/isoCycle/blob/main/isoCycle/files/decoder_schematics.jpg" src="https://github.com/esiabri/isoCycle/blob/main/isoCycle/files/decoder_schematics.jpg" align="center" style="pointer-events: auto;" width="600px"/>
</a>

<p align="center">
  <img alt="https://github.com/esiabri/isoCycle/blob/main/isoCycle/files/gammaCycleAvgShape.png" src="https://github.com/esiabri/isoCycle/blob/main/isoCycle/files/gammaCycleAvgShape.png" align="center" style="pointer-events: auto;" width="600px"/>
</a>



## Extract the Cycle Times in Your Spiking Data on Google Colab
<p align="center">
<a class="new-tab-link" href="https://colab.research.google.com/github/esiabri/isoCycle/blob/main/isoCycle_yourData_Colab.ipynb" target="_blank" style="pointer-events: none;">
  <img alt="https://colab.research.google.com/assets/colab-badge.svg" src="https://colab.research.google.com/assets/colab-badge.svg" align="center" style="pointer-events: auto;" width="250px"/>
</a>

On the Google Colab linked above, you can use isoCycle online to extract the times of the cycle you choose (e.g., gamma, theta, etc.) from your data. All you need is to have your recorded spike times (from all the nearby clusters combined) saved in a .npy file. If you're using kilosort/phy for spike sorting, you can use this [matlab function](https://github.com/esiabri/isoCycle/blob/main/isoCycle/files/isoCycleInput_build.m) to generate the .npy file for this google colab, otherwise this [python function](https://github.com/esiabri/isoCycle/blob/main/isoCycle/decoder.py#L646) could be helpful to include the desired spike clusters into the cycle detection.</p>

The default figures in this Colab notebook show the average shape of gamma and beta cycles, as well as slower cycles, in the spiking activity of 131 neurons simultaneously recorded in the mouse primary visual cortex (V1) during a passive visual stimulation experiment. You can regenerate these figures using this [data file](https://drive.google.com/file/d/1USRsBorZa9iL2ukuZJOt5P8Eu8tZL3aJ/view?usp=sharing). For the local machine version of this notebook see [example](https://github.com/esiabri/isoCycle/tree/main/isoCycle/example).</p>

## GPU acceleration
isoCycle employs a decoder developed with TensorFlow. Leveraging a GPU highly reduces execution time, a factor particularly crucial for long recordings. However, configuring your GPU to work with TensorFlow requires additional steps post-package installation. For quick guidance, refer to [Setting up TensorFlow with GPU Support](#setting-up-tensorflow-with-gpu-support). 
    
## Installation
<p>
  
**quick**: install isoCycle with pip, consider a new environment as conflicts are likely

```buildoutcfg
pip install isoCycle
```

Steps for installation from scratch using Anaconda:

1. First, ensure that you have Anaconda installed on your computer. If you don't have Anaconda, you can download it from the official Anaconda [website](https://www.anaconda.com/downloads). Anaconda is a popular distribution of Python that comes with many pre-installed packages and a package manager called conda, making it convenient for data analysis and scientific computing tasks.

2. Once you have Anaconda installed, open a terminal or command prompt on your computer, and create a new conda environment by executing the following command:
```buildoutcfg
conda create --name myenv
```
Replace myenv with the desired name for your environment.

3. Activate the newly created environment with the following command:
```buildoutcfg
conda activate myenv
```
Again, replace myenv with the name of your environment.

4. Install isoCycle and its dependencies by running the command:
```buildoutcfg
conda install isoCycle
```
5. After the installation is complete, you can import isoCycle into your Python scripts or notebooks using the statement
```buildoutcfg
import isoCycle
```

Now, with Anaconda installed, a new environment created, and isoCycle successfully installed, you are ready to analyze your data using isoCycle. You can use [Jupyter Notebook](https://jupyter.org/try-jupyter/retro/notebooks/?path=notebooks/Intro.ipynb) to import and use isoCycle
  
6. To run the [isoCycle_example.ipynb](https://github.com/esiabri/isoCycle/blob/main/isoCycle/example/isoCycle_example.ipynb) refer to [example](https://github.com/esiabri/isoCycle/tree/main/isoCycle/example)
</p>

## Setting up TensorFlow with GPU Support

To utilize the power of GPU acceleration with TensorFlow, ensure that you have the correct hardware and software setup. 

You will need a CUDA-capable GPU and you'll need to install the CUDA Toolkit and the cuDNN library. Once these are installed, you can install TensorFlow with built-in GPU support.

To check if TensorFlow is using the GPU, you can use the following Python code:

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```
  
You can also log device placements with:
  
```python
tf.debugging.set_log_device_placement(True)
```
  
For more detailed information, please refer to the following guides:

1. [Using a GPU with TensorFlow.](https://www.tensorflow.org/guide/gpu)
2. CUDA Toolkit and cuDNN installation guides on the [NVIDIA website](https://developer.nvidia.com/cuda-toolkit).
Please follow the respective instructions carefully when installing these components.
