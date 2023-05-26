
import pickle
import tensorflow as tf
import numpy as np
from scipy import signal
import matplotlib.pylab as plt
import ipywidgets
from ipywidgets import *
from matplotlib.widgets import Slider
import pkg_resources
import subprocess

from isoCycle import spike
from isoCycle import utility



def cycleDetection(spikeTimes, decoderAdd=None,\
        regionName='V1', cycleName='gamma', wholeSession=False, cycleDetectionDur=10, detectionThreshold=0.035,\
            inputName='populationSpiking', interCycleIntervalFig=True, spikeDistAroundCycles=True,\
                cycleNoForSpikeDisFig=10000, distWindowLenghtPerCycle=3, binsPerCycle=30,spikingDistShowSlider = False,\
                    limitedRAM=False, segmentLengthCoeff=50e3):
    
    '''
         detect cycles from population spike times recorded in a local network
         inputs:
            spikeTimes                 : the time of spikes from all neurons in seconds#
            decoderAdd                 : default: None - uses the default model included in the module
            cycleName                  : default: 'gamma' - options: 'highGamma', 'gamma', 'beta', 'alpha', 'theta' ,'delta'
            wholeSession               : default: False - if True the decoder is run on the entire session, if Fasle the cycles are detected in the first "cycleDetectionDur" seconds
            cycleDetectionDur          : default: 1500 - the epoch length from session start for cycle detection if wholeSession == False
            detectionThreshold         : default: 0.035 - threshold for detecting the cycles from the decoder input, the default value is for 5% FA and 93% Hit for 5% noise and 20% Jitter (std) on the width of the cycles
            regionName                 : default: 'V1' - the name of the region for the data for the generated figures
            interCycleIntervalFig      : default: True - to generate the inter-cycle-interval distribution of the detected cycles
            spikeDistAroundCycles      : default: True - to generate the distribution of spikes around the detected cycles
            cycleNoForSpikeDisFig      : default: 10000 - Number of the detected cycles used to generate the spike distribution around the cycles, 'All' to included all the detected cycles
            distWindowLenghtPerCycle   : default: 3 - the window length for the spike distribution figure based on the length of the aimed cycle 
            binsPerCycle               : default: 30 - the number of bins per cycle for the spike distribution figure
            spikingDistShowSlider      : default: False - show slider to change the bin size on the figure- can be run on googlecolab but very laggy
            limitedRAM                 : default: False - if True, the data is segmented to parts before decoding
            segmentLengthCoeff         : default: 50e3, the length of the segments in term of cycles to be detected if the limitedRAM==True

         output:
            detectedCyclesTimes        : one dimensional numpy array containing the time of the detected cycles 
    '''

    # check the input dimensionality
    try:
        spikeTimes = np.array(spikeTimes).squeeze()
    except:
        print('Spike Times should be in a one dimensional numpy array, use YOUR_ARRAY.ndim to check the dimensionality, you can use np.concatenate(YOUR_ARRAY) in case you have several clusters saved separately')
        return np.array([])
    
    if spikeTimes.ndim>1:
        print('Spike Times should be in a one dimensional numpy array, use YOUR_ARRAY.ndim to check the dimensionality, you can use np.concatenate(YOUR_ARRAY) in case you have several clusters saved separately')
        return np.array([])
    
    if np.max(spikeTimes)>1e6:
        print('I guess your numpy array contains the sample number for the spikes and not the spike times, devide the values with the sampling rate')
        return np.array([])

    # cycle scale to initiate the proper decoder
    if cycleName == 'highGamma':
        n = -8
        cycleColor = '#E8250C'
    elif cycleName == 'gamma':
        n = -7
        cycleColor = '#D98911'
    elif cycleName == 'beta':
        n = -6
        cycleColor = '#339172'
    elif cycleName == 'alpha':
        n = -5
        cycleColor = '#42daf5'
    elif cycleName == 'theta':
        n = -4
        cycleColor = '#2B4482'
    elif cycleName == 'delta':
        n = -3
        cycleColor = '#8C2B81'
    else:
        print('Enter a valid value for the target cycle, options: highGamma, gamma, beta, alpha, theta, delta')
        return np.array([])

    # load the decoder
    loadedDecoder, dt, binsBefore, binsAfter, cycleWidth = \
                        loadDecoder(decoderAdd=decoderAdd, modelEvaluation=False, n=n)
    

    # restricting the analysis to a portion of the session
    if wholeSession == False:
        spikeTimes = spikeTimes[spikeTimes<=(cycleDetectionDur+0.1)]

    
    if limitedRAM==True:
        segmentLength = segmentLengthCoeff*cycleWidth
        segmentNo = int(np.max(spikeTimes)/segmentLength) + 1
    else:
        segmentNo = 1
        segmentLength = np.max(spikeTimes)*1.1

    detectedCyclesTimes = []
    for segmentCounter in range(segmentNo):

        # generating the matrix with the right amount of history (past and future) for the decoder
        decoderInput_timePoints, decoderInput = \
                buildDecoderInput(spikeTimes[(spikeTimes>(segmentCounter*segmentLength)) &\
                                             (spikeTimes<((segmentCounter+1)*segmentLength))], dt=dt,\
                            bins_before=binsBefore,\
                                    bins_after=binsAfter, zScore=True)
        
        if limitedRAM==True:
            if segmentNo>1:
                print('Segment '+str(segmentCounter+1)+'/'+str(segmentNo))
        

        # applying the decoder on the data
        try:
            decoderOutput = \
                    loadedDecoder.predict(decoderInput,verbose=1)
        except:
            decoderOutput = \
                    loadedDecoder.predict(decoderInput)
            

        detectedCyclesTimes_segment =\
            cycleTimeExtractionFromDecoderOutput(decoderOutput[:,0],\
                cycleWidth, dt=dt, detectionThreshold=detectionThreshold)
        

        # correcting the detected cycle times for the buffer size of the decoder:

        # if inputType == 'spikeTime':
        detectedCyclesTimes_segment = detectedCyclesTimes_segment + decoderInput_timePoints[binsBefore] + dt

        detectedCyclesTimes.append(detectedCyclesTimes_segment)
            
    detectedCyclesTimes = np.concatenate(detectedCyclesTimes).squeeze()

    detectedCyclesNo = len(detectedCyclesTimes)
    sessionLength = int(decoderInput_timePoints[-1])



    print('%(number)d %(string)s cycles detected during %(dur)d seconds in %(regionName)s' %{'string':cycleName,'number':detectedCyclesNo, 'regionName':regionName, 'dur':sessionLength})
    print('')

    # if showFigs:
            
    #     decoderOutputVisualization(\
    #         inputSignalToDecoder[binsBefore_general_template:-binsAfter_general_template],\
    #             decoderOutput, templateDetectedTimes, dt=dt, figTitle='detected cycles on ' + inputName,\
    #                 signalName=signalName)

    if cycleNoForSpikeDisFig=='All':
        cycleNoForSpikeDisFig = len(detectedCyclesTimes)

    if spikeDistAroundCycles: 
            distWindowLenght = distWindowLenghtPerCycle*cycleWidth

            spikingDistRel2detectedCycles(spikeTimes, detectedCyclesTimes[:cycleNoForSpikeDisFig],\
                    distWindowLenght=distWindowLenght, histBinWidth=cycleWidth/binsPerCycle, cycleWdith=cycleWidth,\
                        cycleName=cycleName, cycleColor=cycleColor, showSlider=spikingDistShowSlider)
            print('\n')

    if interCycleIntervalFig:
        fig_ICI = plt.figure(figsize=(6,4))
        ax_ICI = fig_ICI.add_axes([0.2,0.15,0.75,0.75])
        # fig.patch.set_alpha(0)
        # ax.patch.set_alpha(0)

        histHighLim = 5*cycleWidth

        interCycleIntervals = detectedCyclesTimes[1:] - detectedCyclesTimes[:-1]
        interCycleIntervals = interCycleIntervals[interCycleIntervals<histHighLim]

        ax_ICI.hist(interCycleIntervals,bins=50,color=cycleColor);
        # ax_ICI.set_title('Inter-Cycle-Interval distribution - '+cycleName, fontsize=18)
        utility.color_title(['Inter-Cycle-Interval distribution - ',cycleName],\
                            ['k',cycleColor], ax=ax_ICI, textprops={'fontsize':16})
        ax_ICI.set_xlabel('second',fontsize=16)
        ax_ICI.set_ylabel('cycle #',fontsize=16)

        ax_ICI.spines['right'].set_visible(False)
        ax_ICI.spines['top'].set_visible(False)

        ax_ICI.spines['left'].set_position(('axes', -0.02))
        ax_ICI.spines['bottom'].set_position(('axes', -0.01))

        # yMax = np.round(ax_ICI.get_ylim()[1])
        # yMin = np.round(ax_ICI.get_ylim()[0])
        # ax_ICI.set_yticks([yMin,yMax])
        # ax_ICI.set_xticks([plotStart,plotEnd])
        ax_ICI.tick_params(axis='both', which='major', labelsize=12)

    

    #     fileName = 'interCycleIntervalDist_'+inputName+'_'+cycleName+'_general_Templates_'+sessionName+'.pickle'
    #     pickle.dump((fig_ICI,ax_ICI), open(fileName, 'wb'))
    return detectedCyclesTimes

def spikingDistRel2detectedCycles(spikeTimes, detectedCycleTimes, distWindowLenght,\
    histBinWidth, cycleWdith, saveFig=False, inputName='spiking', cycleName='e', sessionName=None, regionName='V1',\
        binsizeLowerBound=-3, defaultZoomDist=None, zeroLine=True, cycleColor='gray',returnFigHandle=False, showFig=True,\
            showSlider=True, normedHist=False, filterationFreqWidthCoeff=3, pValThreshold=1e-1,\
                peakHeightThresholdPercent=1, widthCoeff=1.4, figFolderAdd=''):

    # plt.style.use('dark_background')

    # print(len(detectedCycleTimes))

    if defaultZoomDist==None:
        defaultZoomDist = distWindowLenght
    # calculating the relative time of spikes and each detected template
    responseWindowStart = distWindowLenght
    responseWindowEnd = distWindowLenght

    triggeredSpikesTimes2detectedCycles = \
            spike.eventTriggeredSpiking(spikeTimes, \
                detectedCycleTimes,\
                    responseWindowEnd = responseWindowEnd, \
                        responseWindowStart = responseWindowStart)

    responseWindowDur = responseWindowStart + responseWindowEnd
    binNo = int(responseWindowDur/histBinWidth)

    fig = plt.figure(figsize=(8,6))
    # if showSlider:
    #     ax = fig.add_axes([0.15,0.2,0.8,0.75])
    # else:
    # ax = fig.add_axes([0.2,0.2,0.6,0.6])
    ax = fig.add_axes([0.15,0.2,0.8,0.75])

    counts, bins = np.histogram(np.concatenate(triggeredSpikesTimes2detectedCycles),bins=binNo,\
                                density=normedHist)

    if normedHist:
        plt.hist(bins[:-1], bins, weights=counts*np.ones(len(counts)),\
            align='right',color='grey')
    else:
        plt.hist(bins[:-1], bins, weights=counts*np.ones(len(counts))/len(detectedCycleTimes),\
            align='right',color='grey')

    # plt.axhline(chanceLevelPopulationFR,ls='--',color='w',alpha=0.6)
        
    plt.xlim([-responseWindowStart,responseWindowEnd])
    # plt.hist(np.concatenate(populationTriggeredSpikes_3sTemplates),bins=binNo);
    plt.xlabel('time from cycle center (s)', fontsize=14)
    
    if histBinWidth>0.1:
        plt.ylabel('avg spike count per bin [binsize: %(number)0.1f s]'%{'number':histBinWidth}, fontsize=14)
    elif histBinWidth>1e-3:
        plt.ylabel('avg spike count per bin [binsize: %(number)0.1f ms]'%{'number':histBinWidth*1e3}, fontsize=14)
    elif histBinWidth>1e-5:
        plt.ylabel('avg spike count per bin [binsize: %(number)0.2f ms]'%{'number':histBinWidth*1e3}, fontsize=14)


    # ax.set_title('distribution of '+ inputName +' aligned to detected '+cycleName+ \
    #                                     ' cycles in '+ regionName, fontsize=14)
    
    utility.color_title(['distribution of ', inputName,' aligned to detected ',cycleName, \
                                        ' cycles in ', regionName],\
                            ['k','k','k',cycleColor,'k','k'], ax=ax, textprops={'fontsize':15})

    ax.set_xbound([-defaultZoomDist,defaultZoomDist])

    
    if zeroLine:
        ax.axvline(0,ls='--',color=cycleColor)

    binsizeLowerBound = binsizeLowerBound
    binsizeHigherBound = np.floor(np.log10(responseWindowDur/3))

    if showSlider:

        axBinSize = plt.axes([0.15, 0.02, 0.75, 0.03])

        # binSizeUpdateSlider = Slider(axBinSize, 'bin size', binsizeLowerBound, binsizeHigherBound,\
        #     valinit=np.log10(histBinWidth), color='gray')
        
        binSizeUpdateSlider = Slider(axBinSize, 'bin size', histBinWidth/10, histBinWidth*7,\
            valinit=histBinWidth, color='gray')
        

        if histBinWidth>0.1:
            binSizeUpdateSlider.valtext.set_text('%(number)0.1f s'%{'number':histBinWidth})
        elif histBinWidth>1e-3:
            binSizeUpdateSlider.valtext.set_text('%(number)0.1f ms'%{'number':histBinWidth*1e3})
        elif histBinWidth>1e-5:
            binSizeUpdateSlider.valtext.set_text('%(number)0.2f ms'%{'number':histBinWidth*1e3})

    def update(val):
        histBinWidth = binSizeUpdateSlider.val#10**(binSizeUpdateSlider.val)
        binNoUpdated = int(responseWindowDur/histBinWidth)
        binSizeUpdateSlider.valtext.set_text('%(number)0.2f s'%{'number':histBinWidth})
        ax.clear()
        ax.set_xlim([-responseWindowStart,responseWindowEnd])
        counts, bins = np.histogram(np.concatenate(triggeredSpikesTimes2detectedCycles),\
                                    bins=binNoUpdated)
        ax.hist(bins[:-1], bins, weights=\
            counts*np.ones(len(counts))/len(detectedCycleTimes),\
        align='right',color='grey')
        ax.set_xlabel('time from cycle center (s)', fontsize=14)
        if histBinWidth>0.1:
            ax.set_ylabel('avg spike count per bin [binsize: %(number)0.1f s]'%{'number':histBinWidth}, fontsize=14)
            binSizeUpdateSlider.valtext.set_text('%(number)0.1f s'%{'number':histBinWidth})
        elif histBinWidth>1e-3:
            ax.set_ylabel('avg spike count per bin [binsize: %(number)0.1f ms]'%{'number':histBinWidth*1e3}, fontsize=14)
            binSizeUpdateSlider.valtext.set_text('%(number)0.1f ms'%{'number':histBinWidth*1e3})
        elif histBinWidth>1e-5:
            ax.set_ylabel('avg spike count per bin [binsize: %(number)0.2f ms]'%{'number':histBinWidth*1e3}, fontsize=14)
            binSizeUpdateSlider.valtext.set_text('%(number)0.2f ms'%{'number':histBinWidth*1e3})
        # ax.set_title('distribution of '+ inputName +' aligned to detected '+cycleName+ \
        #                                 ' cycles in '+ regionName, fontsize=14)
        
        utility.color_title(['distribution of ', inputName,' aligned to detected ',cycleName, \
                                        ' cycles in ', regionName],\
                            ['k','k','k',cycleColor,'k','k'], ax=ax, textprops={'fontsize':15})
        ax.patch.set_alpha(0)
        axBinSize.patch.set_alpha(0)
        fig.canvas.draw_idle()
        

        if zeroLine:
            ax.axvline(0,ls='--',color=cycleColor)
        

    if showSlider:
        binSizeUpdateSlider.on_changed(update)

    

    # quantifying the spiking triggered waveshape in the normalized distribution
    
#     if normedHist:
#         triggeredSpikes = np.concatenate(triggeredSpikesTimes2detectedCycles)

        

#         histHeights = counts*np.ones(len(counts))
#         timePoints = bins[1:]

#         histHeights_filtered = filters.butter_lowpass_filter(histHeights,\
#                                     filterationFreqWidthCoeff/cycleWdith,1/histBinWidth)

#         ax.plot(timePoints, histHeights_filtered)

#         # baseline = 1/(-np.min(timePoints)*2)
#         baseline = np.sum(counts)/len(counts)
#         ax.axhline(baseline,ls='--')

#         # boundries to look at the significance of the triggered spiking pattern and look for the 
#         # max and mins and calculating the width
        

#         leftBoundInd = np.where(timePoints>-((widthCoeff/2)*cycleWdith))[0][0]
#         rightBoundInd = np.where(timePoints<((widthCoeff/2)*cycleWdith))[0][-1]


#         leftBound = timePoints[leftBoundInd]
#         rightBound = timePoints[rightBoundInd]

#         timePoints_cut = timePoints[leftBoundInd:rightBoundInd]

#         histHeights_filtered_cut = histHeights_filtered[leftBoundInd:rightBoundInd]

#         # just checking in the signal around zero! 
#         pvalDist = stats.kstest(misc.normalizedBetween_0_and_1(\
#                     triggeredSpikes[(triggeredSpikes<((widthCoeff/2)*cycleWdith)) &\
#                                     (triggeredSpikes>-(widthCoeff/2)*cycleWdith)]),'uniform')[1]

        
#         #finding all the mins and maxes
#         signalDerivative = histHeights_filtered[1:] - histHeights_filtered[:-1]
#         signalDerivative_cut = histHeights_filtered_cut[1:] - histHeights_filtered_cut[:-1]
        
#         if pvalDist<pValThreshold: #if the distribution is significantly different from a straightline, uniform dist
        
#             maxInds = np.where((signalDerivative_cut<0) & (np.roll(signalDerivative_cut,1)>0))[0]
#             minInds = np.where((signalDerivative>0) & (np.roll(signalDerivative,1)<0))[0]
            
#             maxHeights = histHeights_filtered_cut[maxInds]
#             minHeights = histHeights_filtered[minInds]
            
#             maxTimes = timePoints_cut[maxInds]
#             minTimes = timePoints[minInds]
            
#             #heighest Max
#             heighestPeakInd = maxInds[np.argmax(maxHeights)]
            
#             peakHeight = histHeights_filtered_cut[heighestPeakInd]
#             peakTime = timePoints_cut[heighestPeakInd]
            
#             plt.axvline(peakTime,alpha=0.5)
            
#             peakHeight_rel2Baseline_Percent = 100*(peakHeight-baseline)/baseline

            
            
#             #if the peak is larger than the threshold look for the surrounding troughs 
#             if ((peakHeight_rel2Baseline_Percent>peakHeightThresholdPercent) and (np.abs(peakTime)<cycleWdith) and\
#                 (len(minTimes)>1)):
                
#     #             try:
#                 twoCloseMinTimes = timePoints[minInds[np.argsort(np.abs(minTimes-peakTime))[:2]]] #getting the time of the two closest mins to the max

#                 # check the mins to be in the two sides of the max and not being so far
#                 # if (np.max(np.abs(twoCloseMinTimes-peakTime))<cycleWdith) and (twoCloseMinTimes[0]*twoCloseMinTimes[1]<0):
#                 if ((twoCloseMinTimes[0]-peakTime)*(twoCloseMinTimes[1]-peakTime)<0):
#                     leftMinTime = np.min(twoCloseMinTimes)
#                     rightMinTime = np.max(twoCloseMinTimes)


#                     leftMinLevel = histHeights_filtered[minInds[np.argsort(np.abs(minTimes-peakTime))[0]]]
#                     rightMinLevel = histHeights_filtered[minInds[np.argsort(np.abs(minTimes-peakTime))[1]]]


#                 else:
#                     leftMinTime = np.NAN
#                     rightMinTime = np.NAN
#                     leftMinLevel = np.NAN 
#                     rightMinLevel = np.NAN

#                     plt.text(-cycleWdith*1.5,1.01*(baseline),'peak: %(peak)0.2f %%'%{'peak':\
#                                         peakHeight_rel2Baseline_Percent}, ha='center')

#             else:
#                 leftMinTime = np.NAN
#                 rightMinTime = np.NAN
#                 leftMinLevel = np.NAN 
#                 rightMinLevel = np.NAN

#                 plt.text(-cycleWdith*1.5,1.01*(baseline),'peak: %(peak)0.2f %%'%{'peak':\
#                                         peakHeight_rel2Baseline_Percent}, ha='center')
                    
# #             except:
# #                 leftMinTime = np.NAN
# #                 rightMinTime = np.NAN
#             estimatedWidth = rightMinTime - leftMinTime
#             peak2peakFromLeft = 100*(peakHeight-leftMinLevel)/baseline
#             peak2peakFromRight = 100*(peakHeight-rightMinLevel)/baseline
                
#         else:

#             plt.text(1.5*cycleWdith,1.01*(baseline),'pval: %(pval)0.1e'%{'pval':\
#                                         pvalDist}, ha='center')
            
#             peakHeight_rel2Baseline_Percent = np.NAN
#             peakTime = np.NAN
#             leftMinTime = np.NAN
#             rightMinTime = np.NAN
#             leftMinLevel = np.NAN
#             rightMinTime = np.NAN
#             estimatedWidth = np.NAN
#             peak2peakFromLeft = np.NAN
#             peak2peakFromRight = np.NAN

            
                

        
        
        
#         if ~np.isnan(leftMinTime):

#             plt.text(leftMinTime,peakHeight,'height: %(height)0.1f%%'%{'height':\
#                                         peak2peakFromLeft}, ha='center')


            


#             if (np.abs(peakTime))>1:
#                 plt.text(rightMinTime,peakHeight,'peak time: %(peaktime)0.1f s'%{'peaktime':\
#                                             peakTime}, ha='center')
#             else:
#                 plt.text(rightMinTime,peakHeight,'peak time: %(peaktime)0.0f ms'%{'peaktime':\
#                                             peakTime*1e3}, ha='center')

#             yMin = int((100*(np.min((leftMinLevel,rightMinLevel))-baseline)/baseline)-1)
#             yMax = int(peakHeight_rel2Baseline_Percent+1)
#             yMaxToSet = baseline + (yMax*baseline/100)
#             yMinToSet = baseline + (yMin*baseline/100)

#             ax.set_yticks([yMinToSet,baseline,yMaxToSet])
#             ax.set_yticklabels([yMin,0,yMax])
            
#             xZoomDist = 1.2*cycleWdith
#             yZoomDist = 1.05*yMaxToSet
#             ax.set_xbound([-xZoomDist,xZoomDist])
#             ax.set_ybound([2*baseline-yZoomDist,yZoomDist])

#         else:
#             ax.set_yticks([baseline])
#             ax.set_yticklabels([0])

#         plt.ylabel('% change relative to baseline')

#         if saveFig:
#             fileName = inputName+'_ProbabilityConditionedTo_'+cycleName+'_cycle_in_'+\
#                 regionName+'_'+sessionName+'.pickle'
#             pickle.dump((fig,ax), open(figFolderAdd+fileName, 'wb'))

#         if showFig==False:
#             plt.close()

#         if returnFigHandle:
#             return peakHeight_rel2Baseline_Percent, peakTime, pvalDist, histHeights, \
#                 peak2peakFromLeft, peak2peakFromRight, estimatedWidth, timePoints, fig


#         else:
#             return peakHeight_rel2Baseline_Percent, peakTime, pvalDist, histHeights, \
#                 peak2peakFromLeft, peak2peakFromRight, estimatedWidth, timePoints
    
#     else:
    # if saveFig:
    #     fileName = inputName+'_ProbabilityConditionedTo_'+cycleName+'_cycle_in_'+\
    #         regionName+'_'+sessionName+'.pickle'
    #     pickle.dump((fig,ax), open(figFolderAdd+fileName, 'wb'))
    
    if showFig==False:
        plt.close()

    return triggeredSpikesTimes2detectedCycles#, fig, ax

    
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
    decoderInput_timePoints, SP = \
            spike.spikingProbability(spikeTimes, windowSize=histBinWidth_forDecoder,\
                                                dt=dt, smooth=False, causal=False)

    inputLen=SP.shape[0]


    decoderInput=add_history(SP,bins_before,bins_after)


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

    if decoderAdd == None:
        decoderAdd = pkg_resources.resource_filename('isoCycle', "model/isoCycle_publishedModel_2022-06-28.h5")

    try:
        with open(decoderAdd[:-3]+'.pkl','rb') as f: #modelAdd[0] if choosing the file by pop up window
            modelData_loaded = pickle.load(f)
    except:
        print('model address is not valid, the default model is used\n')
        decoderAdd = pkg_resources.resource_filename('isoCycle', "model/isoCycle_publishedModel_2022-06-28.h5")
        with open(decoderAdd[:-3]+'.pkl','rb') as f: #modelAdd[0] if choosing the file by pop up window
            modelData_loaded = pickle.load(f)

    dt = modelData_loaded['dt']*(2**(n))
    binsBefore = modelData_loaded['binsBefore']
    binsAfter = modelData_loaded['binsAfter']
    cycleWidth = 3*(2**(n))

    decoder = tf.keras.models.load_model(decoderAdd,compile=True)

    return decoder, dt, binsBefore, binsAfter, cycleWidth


def run_exampleNotebook():
    exampleNotebook_path = pkg_resources.resource_filename('isoCycle', 'example/isoCycle_example.ipynb')

    subprocess.run(['jupyter', 'notebook', '--NotebookApp.token=""',exampleNotebook_path], check=True)


def isoCycleInput_build(spikesTimes=[] , spikeClusters=[],\
                         clusterLabel=[], clusterId=[], clustersToInclude='All', dataFileBaseFolder=None):
    
    '''
    - spikesTimes: all the spikes times detected in the recording session in a one dimensional vector (kilosort output format)
    - spikeClusters: the cluster Number of the recorded spikes as in the spikesTimes vector (kilosort output format)
    - clusterLabel: lables of the sorted clusters ('good','mua','noise') (kilosort output format)
    - clusterId: Ids of the sorted clusters in the order as in the clusterLabel vector (kilosort output format)

    options for the clustersToInclude:
    - 'All' : all the single and multi units as labeled by 'good' and 'mua' with phy/kilosort
    - 'SUA' : just to include the single unit clusters
    - a numpy array in which the number of including clusters are provided by the user
    '''

    if len(spikesTimes)!=len(spikeClusters):
        print('the spikeTiems and spikeClusters vectors are not valid')
        return np.array([])
    
    if len(clusterLabel)!=len(clusterId):
        print('the clusterLabel and clusterId vectors are not valid')
        return np.array([])
    
    # if no valid spikeTimes are passed to the function, it opens a window to get the spike_times.py (the output of the kilosort)
    if len(spikesTimes)==0:
        spikesTimes, spikeClusters, clusterLabel, clusterId =\
            spike.loadSpikesFromPhy(dataFileBaseFolder=dataFileBaseFolder)
        

    MUA_clusters = clusterId[np.where(np.array(clusterLabel)=='mua')[0]].astype('int')
    SUA_clusters = clusterId[np.where(np.array(clusterLabel)=='good')[0]].astype('int')


    if isinstance(clustersToInclude, np.ndarray):
        clusterNumbers = clustersToInclude

    elif clustersToInclude=='SUA':
        clusterNumbers = SUA_clusters

    elif clustersToInclude=='All':
        clusterNumbers = np.concatenate((MUA_clusters,\
                                         SUA_clusters))


    allValidSpikes = []

    for clusterCounter in range(len(clusterNumbers)):
        clusterNo = clusterNumbers[clusterCounter]
        
        clusterSpikeTimes = spikesTimes[np.where(spikeClusters==clusterNo)]
        
        allValidSpikes.append(clusterSpikeTimes)
        
    allValidSpikes = np.sort(np.concatenate(allValidSpikes)).squeeze()

    # with open('spikesTimes.pkl','wb') as f:
    #     pickle.dump(np.sort(allValidSpikes),f)

    return np.sort(allValidSpikes)
