
import pickle
import tensorflow as tf
import numpy as np
from scipy import signal

from isoCycle import spike



def cycleDetection(spikeTimes, decoderAdd='isoCycle/isoCycle_publishedModel_2022-06-28.pkl',\
        regionName='V1', cycleName='gamma', wholeSession=False, cycleDetectionDur=10,\
               detectionThreshold=0.035, showFigs=True,\
                decoderOutpuVisualization=False, saveFigs=True, templateWidth_3s=3,\
                continuousInputSampingRate=30, framesStartTime=None, inputName='populationSpiking',\
                    sessionName=None, interCycleIntervalFig=False, highPassFilter=False, histBinWidth=0,\
                        subjectName=None,\
                            savePDF = 'True', notShowFig=True, distWindowLenghtIni=0):
    
    '''
         detect cycles from population spike times recorded in a local network
         inputs:
            spikeTimes          : the time of spikes from all neurons in seconds#
            decoderAdd          : address to the trained decoder folder, default: isoCycle/isoCycle_publishedModel_2022-06-28.pkl if both the pkl file and the folder with the name are in the same folder as this file
            cycleName           : default: 'gamma' - options: 'highGamma', 'gamma', 'beta', 'alpha', 'theta' ,'delta'
            wholeSession        : default: False - if True the decoder is run on the entire session, if Fasle the cycles are detected in the first "cycleDetectionDur" seconds
            cycleDetectionDur   : default: 1500 - the epoch length from session start for cycle detection if wholeSession == False
            detectionThreshold  : default: 0.035 - threshold for detecting the cycles from the decoder input, the default value is for 5% FA and 93% Hit for 5% noise and 20% Jitter (std) on the width of the cycles
            regionName          : default: 'V1' - the name of the region for the data for the generated figures
            showFigs            : default: True - to generate the figures to see the results'''

    # cycle scale to initiate the proper decoder
    if cycleName == 'highGamma':
        n = -8
    elif cycleName == 'gamma':
        n = -7
    elif cycleName == 'beta':
        n = -6
    elif cycleName == 'alpha':
        n = -5
    elif cycleName == 'theta':
        n = -4
    elif cycleName == 'delta':
        n = -3
    else:
        raise ValueError('enter a valid value for the target cycle, options: highGamma, gamma, beta, alpha, theta, delta')

    # load the decoder
    loadedDecoder, dt, binsBefore, binsAfter, cycleWidth = \
                        loadDecoder(decoderAdd=decoderAdd, modelEvaluation=False, n=n)

    # restricting the analysis to a portion of the session
    if wholeSession == False:
        spikeTimes = spikeTimes[spikeTimes<(cycleDetectionDur+0.1)]

    # generating the matrix with the right amount of history (past and future) for the decoder
    decoderInput_timePoints, decoderInput = \
            buildDecoderInput(spikeTimes, dt=dt,\
                        bins_before=binsBefore,\
                                bins_after=binsAfter, zScore=True)
    

    # applying the decoder on the data
    try:
        decoderOutput = \
                loadedDecoder.predict(decoderInput,verbose=1)
    except:
        decoderOutput = \
                loadedDecoder.predict(decoderInput)
        

    detectedCyclesTimes =\
         cycleTimeExtractionFromDecoderOutput(decoderOutput[:,0],\
             cycleWidth, dt=dt, detectionThreshold=detectionThreshold)
    
    detectedCyclesNo = len(detectedCyclesTimes)
    sessionLength = int(decoderInput_timePoints[-1])

    print('%(number)d %(string)s cycles detected during %(dur)d seconds in %(regionName)s' %{'string':cycleName,'number':detectedCyclesNo, 'regionName':regionName, 'dur':sessionLength})
    print('')

    # if showFigs:
            
    #     decoderOutputVisualization(\
    #         inputSignalToDecoder[binsBefore_general_template:-binsAfter_general_template],\
    #             decoderOutput, templateDetectedTimes, dt=dt, figTitle='detected cycles on ' + inputName,\
    #                 signalName=signalName)


    # if interCycleIntervalFig:
    #     fig = plt.figure()
    #     ax = fig.add_axes([0.15,0.1,0.8,0.8])
    #     fig.patch.set_alpha(0)
    #     ax.patch.set_alpha(0)

    #     plt.hist(templateDetectedTimes[1:]\
    #             -templateDetectedTimes[:-1],bins=50);
    #     plt.title('Inter-Cycle-Interval distribution - '+cycleName+' cycle '+inputName)
    #     plt.xlabel('second')

    #     fileName = 'interCycleIntervalDist_'+inputName+'_'+cycleName+'_general_Templates_'+sessionName+'.pickle'
    #     pickle.dump((fig,ax), open(fileName, 'wb'))
    
def decoderOutputVisualization(decoderInput, decoderOutput, \
            detectedTemplateTimes, dt=0.1, signalName='population FR',\
                figTitle = 'detected templates on the data'):


    # plt.style.use('dark_background')

    tVec = np.arange(0,len(decoderOutput)*dt,dt)

    if len(tVec) > len(decoderOutput):
        tVec = tVec[:len(decoderOutput)]

    figDecoderInput = plt.figure()
    axDecoderInput = figDecoderInput.add_axes([0.1,0.1,0.8,0.8])

    axDecoderInput.plot(tVec,decoderInput)

    
    detectedTemplateSamples = (detectedTemplateTimes/dt).astype('int')

    axDecoderInput.plot(detectedTemplateTimes,\
        decoderInput[detectedTemplateSamples],'r*')


    axDecoderInput.set_title(figTitle)
    axDecoderInput.legend([signalName,'detected templates'])
    axDecoderInput.set_xlabel('time (s)')


    figDecoderOutput = plt.figure()
    axDecoderOutput = figDecoderOutput.add_axes([0.1,0.1,0.8,0.8])
    plt.plot(tVec,decoderOutput[:,0])
    plt.title('decoder output')
    plt.xlabel('time (s)')


def cycleTimeExtractionFromDecoderOutput(decoderOutput, detectingTemplateLen,\
        dt=0.1, detectionThreshold=0.1):

    templateSamplesDetected = signal.find_peaks(decoderOutput, distance=detectingTemplateLen/dt/4,\
                                    height=detectionThreshold)[0]

    templateTimesDetected = templateSamplesDetected*dt

    return templateTimesDetected
    
# code changed from NeuralDecoding package https://github.com/KordingLab/Neural_Decoding
def buildDecoderInput(spikeTimes, dt, bins_before=30, bins_current=1, bins_after=30, zScore=True):

    histBinWidth_forDecoder = 10*dt
    decoderInput_timePoints, FR = \
            spike.firingRate(spikeTimes, windowSize=histBinWidth_forDecoder,\
                                                dt=dt, smooth=False, causalFR=False)

    inputLen=FR.shape[0]


    decoderInput=add_history(FR,bins_before,bins_after)


    #Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
    #This makes it so that the different sets don't include overlapping neural data
    validRange=np.arange(bins_before,inputLen-bins_after)
    
    #Get training data
    decoderInput=decoderInput[validRange,:,:]

    if zScore:
        X_mean=np.nanmean(decoderInput,axis=0)
        X_std=np.nanstd(decoderInput,axis=0)
        decoderInput=(decoderInput-X_mean)/X_std


    return decoderInput_timePoints, decoderInput

# code changed from Neural_Decoding package https://github.com/KordingLab/Neural_Decoding
def add_history(input,bins_before,bins_after,bins_current=1):

    num_examples=len(input) #Number of total time bins we have neural data for
    surrounding_bins=bins_before+bins_after+bins_current #Number of surrounding time bins used for prediction
    X=np.empty([num_examples,surrounding_bins,1]) #Initialize covariate matrix with NaNs
    X[:] = np.NaN
    #Loop through each time bin, and collect the spikes occurring in surrounding time bins
    #Note that the first "bins_before" and last "bins_after" rows of X will remain filled with NaNs, since they don't get filled in below.
    #This is because, for example, we cannot collect 10 time bins of spikes before time bin 8
    start_idx=0
    for i in range(num_examples-bins_before-bins_after): #The first bins_before and last bins_after bins don't get filled in
        end_idx=start_idx+surrounding_bins; #The bins of neural data we will be including are between start_idx and end_idx (which will have length "surrounding_bins")
        X[i+bins_before,:,0]=input[start_idx:end_idx] #Put neural data from surrounding bins in X, starting at row "bins_before"
        start_idx=start_idx+1;
    return X


def loadDecoder(decoderAdd, modelEvaluation=False, n=0):

    with open(decoderAdd,'rb') as f: #modelAdd[0] if choosing the file by pop up window
        modelData_loaded = pickle.load(f)

    dt = modelData_loaded['dt']*(2**(n))
    binsBefore = modelData_loaded['binsBefore']
    binsAfter = modelData_loaded['binsAfter']
    cycleWidth = 3*(2**(n))

    decoder = tf.keras.models.load_model(decoderAdd[:-4],compile=True)

    return decoder, dt, binsBefore, binsAfter, cycleWidth