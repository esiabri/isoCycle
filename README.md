# isoCycle
A Deep Network-Based Decoder for Isolating Single Cycles of Neural Oscillations in Spiking Activity

Neural oscillations are prominent features of neuronal population activity in the brain observed across different species, manifesting in various forms such as frequency-specific power changes in electroencephalograms (EEG) and local field potentials (LFP), as well as phase locking between different brain regions, modulated by modes of activity. Despite the specific relation between neural oscillations and the spiking activity of single neurons, their identification has predominantly relied on indirect measures of neural activity like EEG or LFP, overlooking direct exploration of oscillatory patterns in the spiking activity, which serve as the currency for information processing and information transfer in neural systems. Recent advancements in densely recording large number of neurons within a local network have enabled direct evaluation of changes in network activity over time by examining population spike count variations across different time scales. In this study, we introduce isoCycle, which leverages the power of deep neural networks to robustly isolate single cycles of neural oscillations from the spiking of densely recorded populations of neurons. isoCycle effectively identifies individual cycles in the temporal domain, where cycles from different time scales may have been combined in various ways to shape spiking probability. We demonstrate the utility of isoCycle by employing it to align trials in sensory stimulation experiments based on sensory-driven gamma cycles, offering an alternative to relying solely on external events such as stimulus onset. By accounting for biological jitters, isoCycle enables the retrieval of undistorted neural dynamics in response to sensory stimulation, thus enhancing our understanding of the underlying mechanisms.
