import numpy as np
from isoCycle import utility
import os
import csv
import pickle

# Here we generate an estimation of network spiking probability for a vector of spike times that are passed to the function
# inputs:
# 1 - spikeTimes: the vector of spike times, the times should be in seconds
# 2 - startTime: the start of the signal (if not provided, then the smallest spike time is considered as the signal start)
# 3 - endTime: the end of the signal (if not provided, then the largest spike time is consdiered as the sinal end)

# optional inputs:
# dt: the temporal resolution of the probability vector to be generated, the default value is 1 ms
# windowSize: the window size for counting the spikes, the default value is 50 ms
# slidingWindow: (default True) using sliding windows for calculating the probability, if slideingWindow is set
# to false then the dt (temporal resolution) is set to be equall to the windowsize regardless of the value that
# is passed to the function

# outputs:
# 1- timePoints: the vector of time points regarding the output probability vector
# 2- SP: spiking probability

def spikingProbability(spikeTimes,startTime=None,endTime=None,dt=1e-3,windowSize=50e-3,slidingWindow=True,\
                smooth=False, smoothingWindowDurTime=20e-3, causal=True):

    # sort the spike times ascendingly 
    spikeTimes = np.sort(spikeTimes)

    # if stratTime of the output vector is not determined then the start time is set to the time of the first spike
    if startTime is None:
        startTime = np.min(spikeTimes)

    # if endTime of the output vector is not determined then the end time is set to the time of the last spike
    if endTime is None:
        endTime = np.max(spikeTimes)
    
    # if sliding window is set to false then the temporal resolution is set to the window size
    if slidingWindow is False:
        dt = windowSize

    # the vector of time points that are used to calculate probability
    if causal:
        timePoints = np.arange(startTime+windowSize,endTime,dt)
    else:
        timePoints = np.arange(startTime+windowSize/2,endTime,dt)

    # the probability vector
    SP = np.zeros(timePoints.shape)

    # in each iteration of this loop we count the spikes in one window and estimate the probability in that window
    for windowCounter in range(len(timePoints)):

        # finding the index of the first spike in the window (index: the spike number)
        indsStart = np.searchsorted(spikeTimes, startTime + ((windowCounter)*dt))

        #finding the index of the last spike in the window
        indsEnd = np.searchsorted(spikeTimes, startTime + ((windowCounter)*dt)+windowSize)
                 
        # estimated probability in the window: (number of spikes in the window)/(window duration)
        SP[windowCounter] = (indsEnd - indsStart)/windowSize

        
    return timePoints, SP


def eventTriggeredSpiking(spikesTime, eventsTime, responseWindowEnd = 3, responseWindowStart = 1):

    TriggeredSpikesAllTrials = []

    for trialTime in eventsTime:

        SpikeTimesTrial = spikesTime[(spikesTime<(trialTime+responseWindowEnd))&\
                                (spikesTime>(trialTime-responseWindowStart))] - trialTime

        TriggeredSpikesAllTrials.append(SpikeTimesTrial)
 
    return np.array(TriggeredSpikesAllTrials, dtype=object)

from scipy.spatial import KDTree
def eventTriggeredSpikingKDTree(spikesTime, eventsTime, responseWindowEnd=3, responseWindowStart=1, histBinWidth=50e-3):

    spikesTime = spikesTime.reshape(-1, 1)
    eventsTime = eventsTime.reshape(-1, 1)
    
    kdtree = KDTree(spikesTime)
    
    # Create batches of eventsTime for querying
    batch_size = 1000  # You can experiment with different batch sizes
    num_batches = len(eventsTime) // batch_size + 1
    
    TriggeredSpikesAllTrials = []
    
    # Loop through each batch
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        
        # Query the KDTree in batches
        indices_list = kdtree.query_ball_point(eventsTime[start_idx:end_idx], r=responseWindowEnd)
        
        # Process the result of the batch query
        for j, index_list in enumerate(indices_list):
            if index_list:  # Check if index_list is not empty
                index_list = np.array(index_list, dtype=int)  # Convert to integer type
                triggered_spikes = spikesTime[index_list] - eventsTime[start_idx + j]
                TriggeredSpikesAllTrials.append(triggered_spikes)
    
    return np.array(TriggeredSpikesAllTrials, dtype=object)


def loadSpikesFromPhy(dataFileBaseFolder=None):

    if dataFileBaseFolder==None:
        dataFileBaseFolder = os.getcwd()

    print('use the pop-up window to select the spike_times.py in the kilosort output folder')

    spikeFileAdd, spikeSortingBaseFolder = \
        utility.loadFilePath(dataFileBaseFolder,fileExtension="*.npy",fileDesdription="spike_times")
    

    spikesSampleFileAdd = spikeSortingBaseFolder + '/' + 'spike_times.npy'
    spikeClusterFileAdd = spikeSortingBaseFolder + '/' + 'spike_clusters.npy'

    paramsFileAdd = spikeSortingBaseFolder + '/' + 'params.py'

    local_vars = {}
    with open(paramsFileAdd, 'r') as file:
        exec(file.read(), globals(), local_vars)
        fs = local_vars['sample_rate']

    # print(fs)


    spikeClusters = np.load(spikeClusterFileAdd)  #the file is saved after Ctl+s in Phy
    spikesSample = np.load(spikesSampleFileAdd)

    clusterLabelFileAdd = spikeSortingBaseFolder + '/' + 'cluster_info.tsv'

    clusterId = []
    clusterLabel = []

    if os.path.exists(clusterLabelFileAdd):
        with open(clusterLabelFileAdd) as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t")
            for row in reader:
                clusterId.append(row[0])
                clusterLabel.append(row[8])

    else: # if phy has not been run on the data, direct labels from KS
        clusterLabelFileAdd = spikeSortingBaseFolder + '/' + 'cluster_KSLabel.tsv'

        with open(clusterLabelFileAdd) as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t")
            for row in reader:
                clusterId.append(row[0])
                clusterLabel.append(row[1])

        
    clusterId = np.array(clusterId[1:])
    clusterLabel = np.array(clusterLabel[1:])

    # MUA_clusters = clusterId[np.where(np.array(clusterLabel)=='mua')[0]].astype('int')
    # SUA_clusters = clusterId[np.where(np.array(clusterLabel)=='good')[0]].astype('int')

    spikesTimes = spikesSample/fs
    
    return spikesTimes, spikeClusters, clusterLabel, clusterId #,SUA_clusters, MUA_clusters