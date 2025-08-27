# Wavecraft

Wavecraft is a Python-based tool for audio manipulation, segmentation and batch processing. It started as a unification of a lot of small bash and python tools I have made over time to make my life easier. However, it is slowly growing into a more comprehensive tool for audio processing. Wavecraft has a command-line interface for performing various operations on audio files. In its simplest form, it requires an operation name and an input file. The input can be a directory of files for batch processing. It can also take a text file as input for some operations. See the [usage](#usage) section for more details.

Wavecraft is a work in progress. I'll be updating and expanding it in the coming months, years and... decades? Pull the latest for updates and bug fixes. I'll be adding more operations and features if time permits. If you have any suggestions or feedback, please let me know.



## Dependencies

`Wavecraft` requires the following modules:

- `Python` 3.6 or higher
- `librosa`
- `soundfile`
- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`
- `pyloudnorm`
- `sounddevice`

Look at the [requirements.txt](requirements.txt) and [dependency installer](#python-dependency-installation-script) for details on how to use the bash script to install the dependencies. It will hopefully become a pip package in the future, but for now, you can use the bash script to install the dependencies.

## Usage

First, if you get a permission error, make sure the script is executable:

```shell
chmod +x wac.py
```

To perform operations on an audio file using `wac.py`, run the following command:

```shell
./wac.py operation [options] arg
```

Where `operation` is the operation to perform, `options` are the options for the operation, and `arg` is the path to the audio, metadata, or dataset file. 

Replace `operation_name` with the name of the operation you want to perform. The available operations so far are. All the operations can be done on a single file or batch process a directory of files:

- `segment`: Segment an audio file based on onsets or other methods.
- `extract`: Extract features from an audio file.
- `proxim`: Calculate proximity metrics for a dataset. It can find the most similar sounds in a dataset based on various metrics.
- `decompose`: Decompose an audio file using various methods.
- `beat`: Extract beats from an audio file.
- `onset`: Extract onsets from an audio file.
- `filter`: Apply filters to an audio files.
- `fade`: Apply fades to an audio file.
- `norm`: Normalise an audio file.
- `wmetadata`: Write metadata to an audio file.
- `trim`: Trim an audio file.
- `split`: Split an audio file into smaller files.
- `pan`: Pan an audio file or convert a multichannel file to mono.



For a detailed list of operations and their options, run:

```sh
./wac.py -h
```

## Examples

Here are some examples of how to use `Wavecraft`:

### Segment an audio file based on onsets / beats using different onset envelopes

```shell
./wac.py segment input_file.wav [options]
```
```shell
./wac.py segment input_file.wav -t 0.2 -oe mel -ts -70 -ff 50
```

This will segment the audio file into smaller samples based on onsets using a mel spectogram as the onset envelope. It will use a peak threshold of 0.2 and trims silence from both ends of the file if its below -70db and apply a high-pass filter with a cutoff frequency of 50Hz.

### Split an audio file into segments using a text file

To split an audio file into segments based on a text file, run:

```sh
./wac.py segment /path/to/audio/file.wav t /path/to/text/file.txt
```

This will split the audio file into segments based on the text file and save the segments to the output directory.

### Extract features from an audio file

To extract features from an audio file, run:

```sh
./wac.py extract /path/to/audio/file.wav -fdic True
```

This will extract all the features from the audio file and save them to a flattened dictionary in a JSON file.

### Find most similar sounds using proximity metrics

To calculate proximity metrics for a dataset, run:

```sh
./wac.py proxim /path/to/dataset -ns 5 -cls stats
```

This will calculate the proximity metrics for the dataset and retrieve the 5 most similar sounds.


### Decomposing an audio file

```shell
./wac.py decompose input_file.wav [options]
```


Note that some operations may require additional arguments. Use the `--help` option with `wac.py` and the operation name to see the available options for each operation.

# Python Dependency Installation Script

This repository contains a Bash script to automatically install Python dependencies from a `requirements.txt` file.

## Requirements

1. Python
2. Pip

## Usage

1. Ensure you have the `requirements.txt` file in your project directory with the necessary Python packages listed. It should be in the repo root directory.

It should look like this:

```
librosa>=0.10.1
numpy>=1.24.4
```

2. Make the script executable:
    ```shell
    chmod +x install_deps.sh
    ```
4. Run the script:
    ```shell
    ./install_deps.sh
    ```

## Troubleshooting

1. **pip not found**: Make sure `pip` is installed. You might need to install or update it.
2. **requirements.txt not found**: Ensure that the `requirements.txt` file exists in the root directory of `Wavecraft`.


## License

`Wavecraft` is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.

## Help Dump


Required arguments:
    operation
        Operation to perform. See below for details on each operation.
    input
        Path to the audio, metadata or dataset file. It can be a directory for
        batch processing. It is valid for all operations

Help:
    -h, --help
        show this help message and exit

I/O:
    -it, --input-text 
        The text file containing the segmentation data. Defaults to the
        nameofaudio.txt
    -o, --output-directory 
        Path to the output directory. Optional.
    -st, --save-txt
        Save segment times to a text file.

Audio Settings (low-level) - these apply to all operations where relevant:

    -sr, --sample-rate 
        Sample rate that the files will be loaded in for processing. Default is
        22050. Note that the default for exported sounds is the sound files'
        native sample rate
    --fmin 
        Minimum analysis frequency. Default is 30.
    --fmax 
        Maximum analysis frequency. Default is 11000
    --n-fft 
        FFT size. Default is 2048.
    --hop-size 
        Hop size. Default is 512.
    -spc, --spectogram 
        Spectogram to use when doing processes like decomposition among others.
        Default is None, in which case the appropiate spectogram will be used.
        Change this option only if you know what you are doing or if you want to
        experiment.
    -nra, --no-resolution-adjustment
        Disables the automatic adjusment of the analysis resolution and audio
        settings based on file duration. It is enabled by default.

Segmentation : splits the audio file into segments:
    operation -> segment

    -m, --segmentation-method 
        Segmentation method to use.
    -ml, --min-length 
        Minimum length of a segment in seconds. Default is 0.1s. anything
        shorter won't be used
    -t, --onset-threshold 
        Onset detection threshold. Default is 0.08.
    -oe, --onset-envelope 
        Onset envelope to use for onset detection. Default is mel (mel
        spectrogram). Choices are: mel (mel spectrogram), mfcc (Mel-frequency
        cepstral coefficients), cqt_chr (chroma constant-Q transform), rms
        (root-mean-square energy), zcr (zero-crossing rate), cens (chroma energy
        normalized statistics), tmpg (tempogram), ftmpg (fourier tempogram),
        tonnetz (tonal centroid features)
    -bl, --backtrack-length 
        Backtrack length in miliseconds. Backtracks the segments from the
        detected onsets. Default is 20ms.
    -a, --action {1,2,3}
        Action to perform: 1 for Render, 2 for Export, 3 for Exit. This is
        useful if you want just to automatically batch a large folder of files
        with the same outcome.

   generics:
    [-fi, --fade-in 2] [-fo, --fade-out 12] [-ct, --curve-type exp] [-ts,
    --trim-silence -65] [-ff, --filter-frequency 40] [-ft, --filter-type
    high] [-nl, --normalisation-level -3] [-nm, --normalisation-mode peak]

Feature extraction:
    operation -> extract

    -fex, --feature-extractor 
        Feature extractor to use. Default is all. Choices are: mel (mel
        spectrogram), cqt (constant-Q transform), stft (short-time Fourier
        transform), cqt_chr (chroma constant-Q transform), mfcc (Mel-frequency
        cepstral coefficients), rms (root-mean-square energy), zcr (zero-
        crossing rate), cens (chroma energy normalized statistics), tmpg
        (tempogram), ftmpg (fourier tempogram), tonnetz (tonal centroid
        features), pf (poly features).
    -fdic, --flatten-dictionary
        Flatten the dictionary of features. Default is False.

Distance metric learning - finds the most similar sounds based on a features dataset:
    operation -> proxim

    -ns, --n-similar 
        Number of similar sounds to retrieve
    -id, --identifier 
        Identifier to test, i.e., the name of sound file. If not provided, all
        sounds in the dataset will be tested against each other
    -cls, --class_to_analyse 
        Class to analyse. Default: stats. If not provided, all classes will be
        analysed. Note that this option can produce unexpected results if the
        dataset contains multiple classes with different dimensions
    -mt, --metric-to-analyze 
        Metric to analyze
    -tc, --test-condition 
        Test condition for the specified metric. A condition is a string
        enclosed in '', that can be used to filter the dataset. For example, -mt
        duration -tc '0.5-1.5' or -mt duration -tc '<0.5'. Default: None
    -ops
        Use opetions file to fine tune the metric learning
    -mn, --n-max 
        Max number of similar files to retrieve, Default: -1 (all)
    -mtr, --metric-range  [ ...]
        Range of values to test for a specific metric. Default: None

Decomposition - decomposes the audio file into harmonic, percussive or n components:
    operation -> decomp

    -nc, --n-components 
        Number of components to use for decomposition.
    -hpss, --source-separation 
        Decompose the signal into harmonic and percussive components, If used
        for segmentation, the choice of both is invalid.
    -sk, --sklearn
        Use sklearn for decomposition. Default is False.
    -nnf, --nn-filter
        Use nearest neighbor filtering for decomposition. Default is False.
        Produces a single stream, n_components and hpss are not valid

Beat detection - detects beats in the audio file:
    operation -> beat

    -k 
        Number of beat clusters to detect. Default is 5.

Filter - applies a high / low pass filter to the audio file:
    operation -> filter

    -ff, --filter-frequency 
        Frequency to use for the high-pass filter. Default is 40 Hz. Set to 0 to
        disable
    -ft, --filter-type 
        Type of filter to use. Default is high-pass.

Normalization - normalizes the audio file:
    operation -> norm

    -nl, --normalisation-level 
        Normalisation level, default is -3.
    -nm, --normalisation-mode 
        Normalisation mode; default is 'peak'.

Metadata - writes or reads metadata to/from the audio file:
    operations -> wmeta, rmeta

    --meta  [ ...]
        List of metadata or comments to write to the file. Default is None.
    -mf, --meta-file 
        Path to a JSON metadata file. Default is None.

Trim - trims the audio file. Either range or silence can be used. Defining -tr will disable silence trimming:
    operation -> trim

    -tr, --trim-range 
        Trim position range in seconds. It can be a single value or a range
        (e.g. 0.5-1.5) or condition (e.g. -0.5).
    -ts, --trim-silence 
        Trim silence from the beginning and end of the audio file. Default is
        -70 db.

Split - splits the audio file into multiple files:
    operation -> split

    -sp, --split-points  [ ...]
        Split points in seconds. It can be a single value or a list of split
        points (e.g. 0.5 0.2 3).

Fade - applies a fade in and/or fade out to the audio file. See audio settings for options:
    operation -> fade

    -fi, --fade-in 
        Duration in ms for fade in. Default is 12ms.
    -fo, --fade-out 
        Duration in ms for fade in. Default is 2ms.
    -ct, --curve-type 
        Type of curve to use for fades. Default is exponential.

Pan - pans the audio file:
    operation -> pan

    -pa, --pan-amount 
        Pan amount. Default is 0.
    -mo, --mono
        Converts the audio file to mono.
