#!/usr/bin/env python3
# Based on Chris Tralie's implementation of the paper "Let It Bee-Towards NMF-Inspired Audio Mosaicing" by Driedger et al.
import sys
import os
import numpy as np
import scipy.ndimage
import sounddevice as sd
import soundfile as sf
import librosa
from . import utils
from .debug import colors, Debug as debug

class Konkat:
    def __init__(self, args):
        self.args = args
        self.output = utils.get_output_path()
        
        self.args.source = np.ndarray
        if len(args.input) > 1:
            debug.log_info("Concatenating input files")
            self.source_name = os.path.dirname(self.args.input[0]).split('/')[-1]
            self.args.source = np.concatenate([librosa.load(file, sr=self.args.sample_rate)[0] 
                                               for file in self.args.input])
        else:
            self.args.input = args.input = args.input[0]
            self.source_name = os.path.basename(self.args.input).split('.')[0]
            self.args.source = librosa.load(self.args.input, sr=self.args.sample_rate)[0]
        
        self.target_name = os.path.basename(self.args.concat_target).split('.')[0]
        self.args.result = os.path.join(self.output, self.source_name+'_'+self.target_name+'_konkat.wav')
        # spec = importlib.util.spec_from_file_location("options", 'options.py')
        # self.ops = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(self.ops)
        # output = utils.get_output_path()
        # source_name = os.path.basename(self.ops.source).split('.')[0]
        # target_name = os.path.basename(self.ops.target).split('.')[0]
        # result = os.path.join(output, self.ops+'_'+target_name+'_konkat.wav')
        # self.ops.result = result
        # sr = int(ops.sr)
        # self.ops.y = librosa.load(self.ops.source, sr=self.args.sample_rate)[0]
        self.args.y_t = librosa.load(self.args.concat_target, sr=self.args.sample_rate)[0]
        
    def kl_divergence(self, S, approximation, epsilon=1e-10):
        """
        Compute the Kullback-Leibler divergence between S and its approximation.

        Parameters:
        - S: Target matrix.
        - approximation: Approximation of the target matrix.
        - epsilon: Small constant to avoid log(0) and division by zero.

        Returns:
        - KL divergence between S and the approximation.
        """
        approximation_safe = np.where(approximation == 0, 1, approximation)
        ratio = np.where(S / approximation_safe < epsilon, epsilon, S / approximation_safe)
        
        return np.sum(S * np.log(ratio) - S + approximation)


    def nmf_driedger(self, S, basis, n_iterations, rank=7, polyphony=10, continuity=3):
        """
        Implement the technique from "Let It Bee-Towards NMF-Inspired Audio Mosaicing".
        
        Parameters:
        - S: Target matrix of shape (n_frequencies, n_frames).
        - basis: An (n_frequencies, n_components) matrix of template sounds in some time order along the second axis.
        - n_iterations: Number of iterations.
        - rank: Width of the repeated activation filter.
        - polyphony: Degree of polyphony; i.e. number of values in each column of activations which should be un-shrunken.
        - continuity: Half length of time-continuous activation filter.

        Returns:
        - activations: Activation matrix.
        """
        n_frames = S.shape[1]
        n_components = basis.shape[1]
        
        activations = np.random.rand(n_components, n_frames)
            
        errors = np.zeros(n_iterations + 1)
        errors[0] = self.kl_divergence(S, np.dot(basis, activations))
        
        for iteration in range(n_iterations):
            debug.log_info(f"NMF iteration {iteration + 1} of {n_iterations}")
            
            # Avoid repeated activations
            max_filtered = scipy.ndimage.maximum_filter(activations, size=(1, rank))
            activations[activations < max_filtered] *= (1 - float(iteration + 1) / n_iterations)
            
            # Restrict number of simultaneous activations
            col_cutoff = -np.partition(-activations, polyphony, 0)[polyphony, :]
            activations[activations < col_cutoff[None, :]] *= (1 - float(iteration + 1) / n_iterations)
            
            # Support time-continuous activations
            if continuity > 0:
                for offset in range(-activations.shape[0] + 1, activations.shape[1]):
                    z = np.cumsum(np.concatenate((np.zeros(continuity), np.diag(activations, offset), np.zeros(continuity))))
                    x2 = z[2*continuity:]
                    
                    rows, cols = np.diag_indices(activations.shape[0])
                    valid_indices = (rows + offset < activations.shape[1]) & (rows + offset >= 0)
                    rows, cols = rows[valid_indices], cols[valid_indices] + offset
                    # epsilon = 1e-10
                    # activations[rows, cols] *= x2[:len(rows)] / (np.maximum(x2[:len(rows)], z[:-2*continuity][:len(rows)]) + epsilon)
                    # activations[rows, cols] *= x2
                    # activations[rows, cols] = x2
                    activations[rows, cols] *= x2[:len(rows)] * (np.maximum(x2[:len(rows)], z[:-2*continuity][:len(rows)]))
                    
            approximation = np.dot(basis, activations)
            approximation[approximation == 0] = 1
            S_over_approx = S / approximation
            basis_sum = np.sum(basis, 0)
            basis_sum[basis_sum == 0] = 1
            activations *= np.dot(basis.T, S_over_approx) / basis_sum[:, None]
            
            errors[iteration + 1] = self.kl_divergence(S, np.dot(basis, activations))
        
        return activations

    def stft(self, y, n_fft, hop_length, window = 'blackman'):
        return librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, window = window)

    def istft(self, stft, n_fft, hop_length, window = 'blackman'):
        return librosa.core.istft(stft, n_fft=n_fft, hop_length = hop_length, window = window)

    def pitch_shifted_spec(self, y, sr, window_size, hop_size, shift_range=6, gap_windows=10):
        """
        Concatenate pitch-shifted versions of the spectrograms of a sound.
        Parameters:
        - y: A mono audio array.
        - sr: Sample rate.
        - window_size: Window size.
        - hop_size: Hop size.
        - shift_range: The number of half-steps below and above to shift the sound.
        - gap_windows: Number of zeros to append between each shifted spectrogram.
        
        Returns:
        - concatenated_spec: The concatenated spectrogram.
        """
        librosa.effects.pitch_shift(y, sr=sr, n_steps=shift_range)
        spectrograms = [
            librosa.core.stft(
                y,
                n_fft=window_size, hop_length=hop_size, window='blackman'
            ) for shift in range(-shift_range, shift_range + 1)
        ]
        
        gaps = [np.zeros((spectrograms[0].shape[0], gap_windows), dtype=complex) for _ in spectrograms[:-1]]
        
        # Flatten and concatenate all spectrograms and gaps
        concatenated_spec = np.concatenate([item for sublist in zip(spectrograms, gaps) for item in sublist] + [spectrograms[-1]], axis=1)
        
        return concatenated_spec


    def griffin_lim_inverse(self, spectrogram, window_size, hop_size, n_iters=10, window_func=None):
        """
        Perform Griffin-Lim phase retrieval.

        Parameters:
        - spectrogram: An N_freqs x N_windows spectrogram.
        - window_size: Window size used in STFT.
        - hop_size: Hop length used in STFT.
        - n_iters: Number of iterations to go through (default 10).
        - window_func: A handle to a window function (None by default, uses halfsine).

        Returns:
        - An N x 1 real signal corresponding to phase retrieval.
        """
        debug.log_info("Performing phase retrieval")
        EPSILON = 2.2204e-16
        if not window_func:
            window_func = lambda W: np.sin(np.pi*np.arange(W)/float(W))
            
        magnitude_array = np.array(spectrogram, dtype=complex)
        
        for i in range(n_iters):
            # print(f"Iteration {i + 1} of {n_iters}")
            magnitude_array = self.stft(self.istft(magnitude_array, window_size, hop_size, window_func), window_size, hop_size, window_func)
            norm = np.sqrt(magnitude_array * np.conj(magnitude_array))
            norm[norm < EPSILON] = 1
            magnitude_array = np.abs(spectrogram) * (magnitude_array / norm)
        
        audio_signal = self.istft(magnitude_array, window_size, hop_size, window_func)
        
        return np.real(audio_signal)

    def konkat(self, source, target, sr=48000, window_size=2048, hop_size=1024, 
                        nmf_iterations=80, filter_width=7, polyphony_degree=10, activation_half_length=3, pitch_shift=8):
        """
        Apply audio mosaicing on source and target audio.

        Parameters:
        - source: ndarray of shape (n_samples,). Source audio.
        - target: ndarray of shape (n_samples,). Target audio.
        - result_path: Result audio file path.
        - window_size, hop_size, nmf_iterations, filter_width, polyphony_degree, activation_half_length: Parameters for NMF-based processing.

        Returns:
        None. The result is saved to the specified path.
        """
        if pitch_shift == 0:
            shifted_specs = self.stft(source, window_size, hop_size)
            magnitude_spec = np.abs(shifted_specs)
        else:
            shifted_specs = self.pitch_shifted_spec(y=source, sr=sr, window_size=window_size, hop_size=hop_size, shift_range=pitch_shift)
            magnitude_spec = np.abs(shifted_specs)
        
        target_spec = np.abs(self.stft(target, window_size, hop_size))

        H = self.nmf_driedger(target_spec, magnitude_spec, nmf_iterations, rank=filter_width, continuity=activation_half_length, polyphony=polyphony_degree)
        
        reconstructed_specs = shifted_specs @ H
        reconstructed_audio = self.griffin_lim_inverse(reconstructed_specs, window_size, hop_size, n_iters=10)
        # reconstructed_audio = librosa.core.griffinlim(reconstructed_specs, hop_length=hop_size, win_length=window_size)
        normalized_audio = reconstructed_audio / np.max(np.abs(reconstructed_audio))

        return normalized_audio

    def main(self):
        result = self.konkat(self.args.source, self.args.y_t, self.args.sample_rate, \
            window_size=int(self.args.win_size), hop_size=int(self.args.hop_size), \
            nmf_iterations=int(self.args.nmf_iterations), filter_width=int(self.args.filter_width), \
            polyphony_degree=int(self.args.poly_degree), \
            activation_half_length=int(self.args.half_length), pitch_shift=int(self.args.pitch_shift_range))
        # result = self.konkat(self.args.source, self.args.target, sr=self.args.sample_rate, window_size=self.args.n_fft, hop_size=self.args.hop_size, nmf_iterations=self.args.nmf_iterations, filter_width=self.args.filter_width, polyphony_degree=self.args.polyphony_degree, activation_half_length=self.args.activation_half_length)
        prev_result = np.copy(result)
        sd.play(prev_result, samplerate=self.args.sample_rate)
        sd.wait()
        while True:
            confirmation = input(f"\n{colors.GREEN}Do you want to render the results?{colors.ENDC}\n\n1) Render\n2) Replay preview\n3) Exit\n")
            if confirmation.lower() == '1':
                debug.log_info("Rendering")
                sf.write(self.args.result, result, self.args.sample_rate)
                debug.log_info("Done!")
                break
            elif confirmation.lower() == '2':
                debug.log_info("Replaying")
                sd.play(prev_result, samplerate=self.args.sample_rate)
                sd.wait()
            elif confirmation.lower() == '3':
                debug.log_warning("Aborting render")
                sys.exit(1)
            else:
                debug.log_error("\nInvalid input! Choose one of the options below.")
                continue



# if __name__ == '__main__':
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('--source', type=str, help="Path to audio file for source sounds")
#     # parser.add_argument('--target', type=str, help="Path to audio file for target sound")
#     # parser.add_argument('--sr', type=int, default=22050, help="Sample rate")
#     # parser.add_argument('--winSize', type=int, default=2048, help="Window Size in samples")
#     # parser.add_argument('--hopSize', type=int, default=512, help="Hop Size in samples")
#     # parser.add_argument('--NIters', type=int, default=60, help="Number of iterations of NMF")
#     # parser.add_argument('--r', type=int, default=7, help="Width of the repeated activation filter")
#     # parser.add_argument('--p', type=int, default=10, help="Degree of polyphony; i.e. number of values in each column of H which should be un-shrunken")
#     # parser.add_argument('--c', type=int, default=3, help="Half length of time-continuous activation filter")
#     # opt = parser.parse_args()
#     konkat = Konkat()
#     # if(opt.ops == ""):
#     #     konkat(opt.source, opt.target, opt.result, sr=opt.sr, winSize=opt.winSize, \
#     #             hopSize=opt.hopSize, NIters=opt.NIters, r=opt.r, p=opt.p, c=opt.c, \
#     #             savePlots=opt.saveplots)
#     #     sys.exit(1)
#     spec = importlib.util.spec_from_file_location("options", 'options.py')
#     ops = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(ops)
#     output = utils.get_output_path()
#     source_name = os.path.basename(ops.source).split('.')[0]
#     target_name = os.path.basename(ops.target).split('.')[0]
#     result = os.path.join(output, source_name+'_'+target_name+'_konkat.wav')
#     ops.result = result
#     # sr = int(ops.sr)
#     ops.y = librosa.load(ops.source, sr=ops.sr)[0]
#     ops.y_t = librosa.load(ops.target, sr=ops.sr)[0]
#     konkat.main(ops)
#     # shutil.copy(opt.ops, ops.RESULT)

