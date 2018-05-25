# ChromaAudioFeatures

Set of functions and methods to compute various chroma and audio similarity measures particularly for the task of cover song identification.


## Dependencies

[TO DO SOON] : Add docker file with all the dependencies

All dependencies except Essentia can be installed via pip

```bash
$ pip install -r requirements.txt
```

For installing essentia check the documentation or you can easily set it up using the official Essentia docker image.



## Example use


```python
from chroma_features import ChromaFeatures

audio_path = "./test_audio.wav"

#Initiate the chroma class
chroma = ChromaFeatures(audio_file=audio_path, mono=True, sample_rate=44100)

# Now you can compute various chroma features and ther plots using the various methods of object chroma
chroma.chroma_stft()
chroma.chroma_cqt()

#You can specify custom params
chroma.chroma_hpcp(hopSize=2048, numBins=24)
chroma.chroma_cens(hopSize=1024)

```

For more detailed examples have a look on the ipython [notebook](examples.ipynb)



## Contribution






