# -*- coding: utf-8 -*-

'''
Some utility functions for processing audio and computing audio features useful audio cover detection task experiments using various audio processing libraries
------
Albin Andrew Correya
@2017
'''
from dzr_audio.signals import Signal
from essentia import Pool, array
import essentia.standard as estd
import numpy as np
import librosa


class ChromaFeatures:
    '''
    Class containing methods to compute various chroma features
    Methods :
                chroma_stft   : Computes chromagram using short fourier transform
                chroma_cqt    : Computes chromagram from constant-q transform of the audio signal
                chroma_cens   : Computes improved chromagram using CENS method as mentioned in
                chroma_hpcp   : Computes Harmonic pitch class profiles aka HPCP (improved chromagram)
    Example use :
                chroma = ChromaFeatures(deezer_sng_id=123456)
                #chroma cens with default parameters
                chroma.chroma_cens()
                #chroma stft with default parameters
                chroma.chroma_stft()
    '''

    def __init__(self, deezer_sng_id, mono=True):

        self.signal = Signal(deezer_sng_id, mono=True)
        self.sng_length = len(self.signal.data)
        self.audio_vector = self.signal.data[:,0]
        self.fs = 44100

        print "Audio vector loaded with shape", self.audio_vector.shape, "and sample rate", self.fs
        return

    def chroma_stft(self, frameSize=4096, hopSize=2048, display=False):
        """
        Computes the chromagram using librosa
        specifies the parameters for fft and hop_size
        """
        chroma = librosa.feature.chroma_stft(y=self.audio_vector,
                                            sr=self.fs,
                                            tuning=0,
                                            norm=2,
                                            hop_length=hopSize,
                                            n_fft=frameSize)
        if display:
            self.displayChroma(chroma, hopSize)
        return chroma

    def chroma_cqt(self, hopSize=2048, display=False):
        """
        """
        chroma = librosa.feature.chroma_cqt(y=self.audio_vector,
                                            sr=self.fs,
                                            hop_length=hopSize)
        if display:
            self.displayChroma(chroma, hopSize)
        return


    def chroma_cens(self, hopSize=2048, display=False):
        '''
        Computes CENS chroma vectors for the input audio signal (numpy array)
        Refer https://librosa.github.io/librosa/generated/librosa.feature.chroma_cens.html for more parameters
        '''
        chroma_cens = librosa.feature.chroma_cens(self.audio_vector,
                                                  self.fs,
                                                  hop_length=hopSize)
        if display:
            self.displayChroma(chroma_cens, hopSize)
        return chroma_cens

    def chroma_hpcp(self,
                frameSize=4096,
                hopSize=2048,
                windowType='blackmanharris62',
                harmonicsPerPeak=8,
                magnitudeThreshold=1e-05,
                maxPeaks=1000,
                whitening=True,
                referenceFrequency=440,
                minFrequency=100,
                maxFrequency=5000,
                nonLinear=False,
                numBins=12,
                display=False):
        '''
        Compute Harmonic Pitch Class Profiles (HPCP) for the input audio files using essentia standard mode using
        the default parameters as mentioned in [1].
        Please refer to the following paper for detailed explanantion of the algorithm.
        [1]. Gómez, E. (2006). Tonal Description of Polyphonic Audio for Music Content Processing.
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

        audio = array(self.audio_vector)

        print audio.shape

        frameGenerator = estd.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize)

        window = estd.Windowing(type=windowType)

        spectrum = estd.Spectrum()

        # Refer http://essentia.upf.edu/documentation/reference/std_SpectralPeaks.html
        spectralPeaks = estd.SpectralPeaks(magnitudeThreshold=0,
                                            maxFrequency=maxFrequency,
                                            minFrequency=minFrequency,
                                            maxPeaks=maxPeaks,
                                            orderBy="frequency",
                                            sampleRate=self.fs)

        # http://essentia.upf.edu/documentation/reference/std_SpectralWhitening.html
        spectralWhitening = estd.SpectralWhitening(maxFrequency= maxFrequency,
                                                    sampleRate=self.fs)

        # http://essentia.upf.edu/documentation/reference/std_HPCP.html
        hpcp = estd.HPCP(sampleRate=self.fs,
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
            if whitening:
                w_magnitudes = spectralWhitening(spectrum_mag,
                                                frequencies,
                                                magnitudes)
                hpcp_vector = hpcp(frequencies, w_magnitudes)
            else:
                hpcp_vector = hpcp(frequencies, magnitudes)
            pool.add('tonal.hpcp',hpcp_vector)

        if display:
            self.displayHPCP(pool['tonal.hpcp'])

        return pool['tonal.hpcp']

    def beatSyncChroma(self, chroma, display=False):
        """
        Computes the beat-sync chromagram
        """
        y_harmonic, y_percussive = librosa.effects.hpss(self.audio_vector)
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=self.fs)
        #print ("Tempo ->",tempo)
        beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        if display:
            self.displayChroma(beat_chroma)
        return beat_chroma


    def get2DfftMagnitudes(self, feature_vector, display=False):
        """
        Computes 2d - fourier transform magnitude coefficiants of the input feature vector (numpy array)
        Usually fed by Constant-q transform or chroma feature vectors for cover detection tasks.
        """
        # 2d fourier transform
        ndim_fft = np.fft.fft2(feature_vector)
        ndim_fft_mag = np.abs(np.fft.fftshift(ndim_fft))
        if display:
            from librosa.display import specshow
            plt.figure(figsize=(8,6))
            plt.title('2D-Fourier transform magnitude coefficiants')
            specshow(ndim_fft_mag, cmap='jet')
        return song_fft


    def displayChroma(self, chroma, hop_size=1024):
        '''
        Make plots for input chroma vector using matplotlib
        '''
        from librosa.display import specshow
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 8))
        plt.subplot(2,1,1)
        plt.title("Chroma representation of audio")
        specshow(chroma, x_axis='time', y_axis='chroma', cmap='gray_r', hop_length=hopSize)
        plt.show()
        return

    def displayHPCP(self, hpcp):
        '''
        function to display hpcp vector using matplotlib (librosa.display)
        Params
            hpcp : hpcp vector
        '''
        from librosa.display import specshow
        import matplotlib.pyplot as plt
        s_hpcp = np.swapaxes(hpcp,0,1) #swap the axis for specshow function
        plt.figure(figsize=(12,6))
        plt.title("HPCP")
        specshow(s_hpcp,x_axis='time',y_axis='chroma',cmap='gray_r')
        plt.show()
