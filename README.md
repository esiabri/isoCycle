# isoCycle
#### A Deep Network-Based Decoder for Isolating Single Cycles of Neural Oscillations in Spiking Activity

<p align="justify">
Neural oscillations are prominent features of neuronal population activity in the brain, manifesting in various forms such as frequency-specific power changes in electroencephalograms (EEG) and local field potentials (LFP), as well as phase locking between different brain regions, modulated by modes of activity. Despite the intrinsic relation between neural oscillations and the spiking activity of single neurons, identification of oscillations has predominantly relied on indirect measures of neural activity like EEG or LFP, overlooking direct exploration of oscillatory patterns in the spiking activity, which serve as the currency for information processing and information transfer in neural systems. Recent advancements in densely recording large number of neurons within a local network have enabled direct evaluation of changes in network activity over time by examining population spike count variations across different time scales. Here we leverage the power of deep neural networks to robustly isolate single cycles of neural oscillations from the spiking of densely recorded populations of neurons. isoCycle effectively identifies individual cycles in the temporal domain, where cycles from different time scales may have been combined in various ways to shape spiking probability. The reliable identification of single cycle of neural oscillations in spiking activity across various time scales will deepen our understanding about the dynamics of neural activity.
</p>

## Demo Jupyter Notebook
<!--  -->
[isoCycle_example.ipynb](https://github.com/esiabri/isoCycle/blob/main/isoCycle_example.ipynb) demonstrates the code used to extract gamma and beta cycles, as well as slower cycles, from the spiking activity of 131 neurons simultaneously recorded in mouse V1 during visual stimulation.</p>

You can run the demo on Google Colab <a class="new-tab-link" href="https://colab.research.google.com/github/esiabri/isoCycle/blob/main/isoCycle_example_Colab.ipynb" target="_blank" style="pointer-events: none;">
  <img alt="https://github.com/esiabri/isoCycle/blob/main/isoCycle/files/colab-badge_green.svg" src="https://github.com/esiabri/isoCycle/blob/main/isoCycle/files/colab-badge_green.svg" align="center" style="pointer-events: auto;" width="150px"/>
</a> or on your own machine after installing the isoCycle package.

## Extract the Cycle Times in Your Spiking Data on Google Colab
<p align="center">
<a class="new-tab-link" href="https://colab.research.google.com/github/esiabri/isoCycle/blob/main/isoCycle_yourData_Colab.ipynb" target="_blank" style="pointer-events: none;">
  <img alt="https://colab.research.google.com/assets/colab-badge.svg" src="https://colab.research.google.com/assets/colab-badge.svg" align="center" style="pointer-events: auto;" width="250px"/>
</a>

On this Google Colab (linked above), you can use isoCycle online to analyze your own data. All you need to do is upload your recorded spike times to your Google Drive in a .npy file. IsoCycle will then extract the times of the cycle you choose (e.g., gamma, theta, etc.) from your data.
<br>If you're using kilosort/phy for spike sorting, you can use this [matlab function](https://github.com/esiabri/isoCycle/blob/main/isoCycle/files/isoCycleInput_build.m) to generates the .npy file for this google colab.</p>

## GPU acceleration
<p> isoCycle employs a decoder developed with TensorFlow. Leveraging a GPU significantly reduces execution time, a factor particularly crucial for long recordings. However, configuring your GPU to work with TensorFlow requires additional steps post-package installation. For quick guidance, please refer to the section [Setting up TensorFlow with GPU Support](#setting-up-tensorflow-with-gpu-support).</p> 
    
## Installation
<p>
If you are comfortable with Python, you can install isoCycle in your environment of choice with pip:

```buildoutcfg
pip install isoCycle
```
You can then import isoCycle and run the example Notebook: 
<!-- [MOVE THE EXAMPLE DOC TO ITS OWN FOLDER WITH A LINK HERE----TO RUN THE NOTEBOOK ON THE SAME ENV IT NEEDS EXTRA STEPS (what if the env name is known?, running the notebook with a predefined kernel, but it needs to have the additional code for adding the kernel to the jupyter kernel list----] -->
```buildoutcfg
from isoCycle import decoder
decoder.run_example()
```

Here are the detailed steps for installation from scratch using Anaconda:

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
pip install isoCycle
```
5. After the installation is complete, you can import isoCycle into your Python scripts or notebooks using the statement
```buildoutcfg
import isoCycle
```

Now, with Anaconda installed, a new environment created, and isoCycle successfully installed, you are ready to analyze your data using isoCycle. You can use [Jupyter Notebook](https://jupyter.org/try-jupyter/retro/notebooks/?path=notebooks/Intro.ipynb) to import and use isoCycle
  
6. To run the [isoCycle_example.ipynb](https://github.com/esiabri/isoCycle/blob/main/isoCycle_example.ipynb) open a terminal on your computer and execute the following commands:
  <br>(replace myenv with the name of your environment)
```buildoutcfg
conda activate myenv
```
  
```buildoutcfg
python
```
```buildoutcfg
from isoCycle import decoder
decoder.run_example()
```
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
