# isoCycle
### A Deep Network-Based Decoder for Isolating Single Cycles of Neural Oscillations in Spiking Activity

<p align="justify">
Neural oscillations are prominent features of neuronal population activity in the brain, manifesting in various forms such as frequency-specific power changes in electroencephalograms (EEG) and local field potentials (LFP), as well as phase locking between different brain regions, modulated by modes of activity. Despite the intrinsic relation between neural oscillations and the spiking activity of single neurons, identification of oscillations has predominantly relied on indirect measures of neural activity like EEG or LFP, overlooking direct exploration of oscillatory patterns in the spiking activity, which serve as the currency for information processing and information transfer in neural systems. Recent advancements in densely recording large number of neurons within a local network have enabled direct evaluation of changes in network activity over time by examining population spike count variations across different time scales. In this study, we introduce isoCycle, which leverages the power of deep neural networks to robustly isolate single cycles of neural oscillations from the spiking of densely recorded populations of neurons. isoCycle effectively identifies individual cycles in the temporal domain, where cycles from different time scales may have been combined in various ways to shape spiking probability. The reliable identification of single cycle of neural oscillations in spiking activity across various time scales will deepen our understanding about the dynamics of neural activity.
</p>

## Demo Notebook
<p align="justify">
isoCycle_example.ipynb show case the code used to extract gamma and beta cycles from the spiking activity of 131 neurons simultaneously recorded in mouse V1 during visual stimulation.
<\p>
