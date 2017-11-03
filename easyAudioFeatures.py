# -*- coding: utf-8 -*-

"""
Some utility functions for processing and computing audio features using various audio processing libraries

------
Albin Andrew Correya

"""

import os, sys
import scipy
import librosa
import numpy as np
import matplotlib.pyplot as plt 



def getChroma(audio, fs=44100, n_fft=4098,hop_size=2045, display=True):
	"""
	Computes the chromagram using librosa
	specifies the parameters for fft and hop_size
	"""
	chroma = librosa.feature.chroma_stft(y=audio, sr=fs, tuning=0, norm=2, hop_length=hop_size, n_fft=n_fft)
	if display is True:
		plt.figure(figsize=(16, 8))
		plt.subplot(2,1,1)
		plt.title("Chroma representation of audio")
		librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='gray_r', hop_length=hop_size)   
	return chroma


def getBeatSyncChroma(audio_vector, fs, chromagram, display=True):
	"""
	Computes the beat-sync chromagram
	"""
	y_harmonic, y_percussive = librosa.effects.hpss(audio_vector)
	tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=fs)
	print ("Tempo ->",tempo)
	beat_chroma = librosa.util.sync(chromagram,beat_frames,aggregate=np.median)
	if display is True:
		librosa.display.specshow(beat_chroma,x_axis='time', y_axis='chroma', cmap='gray_r', hop_length=4098)
	return beat_chroma


def get2DfftMagnitudes(feature_vector):
	"""
	Computes 2d - fourier transform magnitude coefficiants of the input audio feature vector (numpy array)
	Usually fed by Constant-q transform or chroma feature vectors for cover detection tasks.

	"""  
	song_fft = np.fft.fft2(feature_vector)
	song_fft = np.abs(np.fft.fftshift(song_fft))
	if display is True:
		plt.figure(figsize=(8,6))
		plt.title('2D-Fourier transform magnitude coefficiants')
		librosa.display.specshow(song_fft, cmap='gray_r')
	return song_fft


def loadTrimmedAudio(fname):
	"""
	Trim silence parts of a input audio file
	"""
	audio, fs = librosa.load(fname)
	audio_trim, index = librosa.effects.trim(audio)
	return audio_trim


def essentiaStreamingHPCP(filename,  
				sampleRate=44100, 
				frameSize=4096, 
				hopSize=256, 
				windowType='blackmanharris62', 
				harmonicsPerPeak=8, 
				magnitudeThreshold=1e-05,
				maxPeaks=10000, 
				whitening=False, 
				referenceFrequency=440, 
				minFrequency=100, 
				maxFrequency=5000, 
				nonLinear=False, 
				numBins=24):
	'''

	Compute Harmonic Pitch Class Profiles (HPCP) for the input audio files using essentia audio processing library using
	the default parameters as mentioned in [1].

	Please refer to the following paper for detailed explanantion of the algorithm.

	[1]. Gómez, E. (2006). Tonal Description of Polyphonic Audio for Music Content Processing. INFORMS Journal on Computing, 18(3)


	Parameters for essentia HPCP function

		harmonicsPerPeak : (integer ∈ [0, ∞), default = 0) :
		number of harmonics for frequency contribution, 0 indicates exclusive fundamental frequency contribution

		maxFrequency : (real ∈ (0, ∞), default = 5000) :
		the maximum frequency that contributes to the HPCP [Hz] (the difference between the max and split frequencies must not be less than 200.0 Hz)
		
		minFrequency : (real ∈ (0, ∞), default = 40) :
		the minimum frequency that contributes to the HPCP [Hz] (the difference between the min and split frequencies must not be less than 200.0 Hz)
		
		nonLinear : (bool ∈ {true, false}, default = false) :
		apply non-linear post-processing to the output (use with normalized='unitMax'). Boosts values close to 1, decreases values close to 0.
		normalized (string ∈ {none, unitSum, unitMax}, default = unitMax) :
		whether to normalize the HPCP vector
		
		referenceFrequency : (real ∈ (0, ∞), default = 440) :
		the reference frequency for semitone index calculation, corresponding to A3 [Hz]
		
		sampleRate : (real ∈ (0, ∞), default = 44100) :
		the sampling rate of the audio signal [Hz]
		
		numBins : (integer ∈ [12, ∞), default = 12) :
		the size of the output HPCP (must be a positive nonzero multiple of 12)

	'''

	# import essentia library
	import essentia
	import essentia.streaming as streaming

	# Load audio file to the MonoLoader object
	loader = streaming.MonoLoader(filename=filename)

	# initiate FrameCutter class

	framecutter = streaming.FrameCutter(frameSize=frameSize, hopSize=hopSize)

	# initiate Windowing class with window type
	window = streaming.Windowing(type=windowType)

	# initiate Spectrum object
	spectrum = streaming.Spectrum()

	spectralPeaks = streaming.SpectralPeaks(magnitudeThreshold=0, 
											maxFrequency=maxFrequency, 
											minFrequency=minFrequency, 
											maxPeaks=maxPeaks,
											orderBy="frequency",
											sampleRate=sampleRate)

	if whitening is True:
		spectralWhitening = streaming.SpectralWhitening(maxFrequency= maxFrequency,
														sampleRate=sampleRate)
	
		hpcp = streaming.HPCP(sampleRate=sampleRate, 
								maxFrequency=maxFrequency, 
								minFrequency=minFrequency, 
								referenceFrequency=referenceFrequency, 
								nonLinear=nonLinear,
								harmonics=harmonicsPerPeak,
								size=numBins)


		pool = essentia.Pool()

		# connect algorithms together
		loader.audio >> framecutter.signal
		framecutter.frame >> window.frame >> spectrum.frame
		spectrum.spectrum >> spectralPeaks.spectrum
		spectrum.spectrum >> spectralWhitening.spectrum
		spectralPeaks.frequencies >> spectralWhitening.frequencies
		spectralPeaks.magnitudes >> spectralWhitening.magnitudes
		spectralPeaks.frequencies >> hpcp.frequencies
		spectralWhitening.magnitudes >> hpcp.magnitudes
		hpcp.hpcp >> (pool, 'tonal.hpcp')

	else:
		hpcp = streaming.HPCP(sampleRate=sampleRate, 
								maxFrequency=maxFrequency, 
								minFrequency=minFrequency, 
								referenceFrequency=referenceFrequency, 
								nonLinear=nonLinear,
								harmonics=harmonicsPerPeak,
								size=numBins)

		pool = essentia.Pool()

		# connect algorithms together
		loader.audio >> framecutter.signal
		framecutter.frame >> window.frame >> spectrum.frame
		spectrum.spectrum >> spectralPeaks.spectrum
		spectralPeaks.frequencies >> hpcp.frequencies
		spectralPeaks.magnitudes >> hpcp.magnitudes
		hpcp.hpcp >> (pool, 'tonal.hpcp')


	essentia.run(loader)

	return pool['tonal.hpcp'] 
	#except:
	#	raise ValueError('Failed to compute HPCP ! Please check your input parameters..')



def essentiaStandardHPCP(filename,  
				sampleRate=44100, 
				frameSize=4096, 
				hopSize=256, 
				windowType='blackmanharris62', 
				harmonicsPerPeak=8, 
				magnitudeThreshold=1e-05,
				maxPeaks=1000, 
				whitening=False, 
				referenceFrequency=440, 
				minFrequency=100, 
				maxFrequency=5000, 
				nonLinear=False, 
				numBins=24):
	'''

	Compute Harmonic Pitch Class Profiles (HPCP) for the input audio files using essentia standard mode using
	the default parameters as mentioned in [1].

	Please refer to the following paper for detailed explanantion of the algorithm.

	[1]. Gómez, E. (2006). Tonal Description of Polyphonic Audio for Music Content Processing. INFORMS Journal on Computing, 18(3)


	For full list of parameters of essentia standard mode HPCP please refer to http://essentia.upf.edu/documentation/reference/std_HPCP.html

	Parameters

		harmonicsPerPeak : (integer ∈ [0, ∞), default = 0) :
		number of harmonics for frequency contribution, 0 indicates exclusive fundamental frequency contribution

		maxFrequency : (real ∈ (0, ∞), default = 5000) :
		the maximum frequency that contributes to the HPCP [Hz] (the difference between the max and split frequencies must not be less than 200.0 Hz)
		
		minFrequency : (real ∈ (0, ∞), default = 40) :
		the minimum frequency that contributes to the HPCP [Hz] (the difference between the min and split frequencies must not be less than 200.0 Hz)
		
		nonLinear : (bool ∈ {true, false}, default = false) :
		apply non-linear post-processing to the output (use with normalized='unitMax'). Boosts values close to 1, decreases values close to 0.
		normalized (string ∈ {none, unitSum, unitMax}, default = unitMax) :
		whether to normalize the HPCP vector
		
		referenceFrequency : (real ∈ (0, ∞), default = 440) :
		the reference frequency for semitone index calculation, corresponding to A3 [Hz]
		
		sampleRate : (real ∈ (0, ∞), default = 44100) :
		the sampling rate of the audio signal [Hz]
		
		numBins : (integer ∈ [12, ∞), default = 12) :
		the size of the output HPCP (must be a positive nonzero multiple of 12)

		whitening : (boolean (True, False), default = False)
		Optional step of computing spectral whitening to the output from speakPeak magnitudes


	'''

	from essentia import Pool
	import essentia.standard as standard

	loader = standard.MonoLoader(filename=filename)

	audio = loader()

	frameGenerator = standard.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize)

	window = standard.Windowing(type=windowType)

	spectrum = standard.Spectrum()

	# Refer http://essentia.upf.edu/documentation/reference/std_SpectralPeaks.html
	spectralPeaks = standard.SpectralPeaks(magnitudeThreshold=0, 
											maxFrequency=maxFrequency, 
											minFrequency=minFrequency, 
											maxPeaks=maxPeaks,
											orderBy="frequency",
											sampleRate=sampleRate)

	# http://essentia.upf.edu/documentation/reference/std_SpectralWhitening.html
	spectralWhitening = standard.SpectralWhitening(maxFrequency= maxFrequency,
													sampleRate=sampleRate)

	# http://essentia.upf.edu/documentation/reference/std_HPCP.html
	hpcp = standard.HPCP(sampleRate=sampleRate, 
							maxFrequency=maxFrequency, 
							minFrequency=minFrequency, 
							referenceFrequency=referenceFrequency, 
							nonLinear=nonLinear,
							harmonics=harmonicsPerPeak,
							size=numBins)


	pool = Pool()

	#compute hpcp for each frame and add the results to the pool
	for frame in frameGenerator:
		spectrum_mag = spectrum(window(frame))
		frequencies, magnitudes = spectralPeaks(spectrum_mag)
		if whitening is True:
			w_magnitudes = spectralWhitening(spectrum_mag, 
											frequencies, 
											magnitudes)
			hpcp_vector = hpcp(frequencies, w_magnitudes)
		else:
			hpcp_vector = hpcp(frequencies, magnitudes)
		pool.add('tonal.hpcp',hpcp_vector)

	return pool['tonal.hpcp']



def displayHPCP(hpcp):
	'''
	function to display hpcp vector using matplotlib (librosa.display)
	Params 
		hpcp : hpcp vector
	'''
	from librosa.display import specshow
	s_hpcp = np.swapaxes(hpcp,0,1)
	plt.figure(figsize=(12,6))
	plt.title("HPCP")
	specshow(s_hpcp,x_axis='time',y_axis='chroma',cmap='gray_r')
	plt.show()

