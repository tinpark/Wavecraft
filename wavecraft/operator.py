"""
Operator module for WaveCraft. This module contains the main function for the WaveCraft CLI.
"""
from calendar import c
import os
import sys
import asyncio
import soundfile as sf
import librosa

from .konkat import Konkat
from .segmentor import Segmentor
from .feature_extractor import Extractor
from .onset_detector import OnsetDetector
from .debug import Debug as debug, colors
from .beat_detector import BeatDetector
from .decomposer import Decomposer
from .proxi_metor import ProxiMetor
from . import utils
from . import metadata

def main(args, revert=None):
    """
    Main function that performs various operations on audio files based on the provided arguments.

    Args:
        args (argparse.Namespace): The command-line arguments.
        revert (Optional[str]): The revert option.

    Returns:
        None
    """
    utils.progress(args.operation)
    
    if args.operation == "proxim":
        debug.log_info('Calculating <proximity metric>')
        craft = ProxiMetor(args)
        craft.main()
        return

    dsp = args.operation not in ["wmeta", "info", "proxim"]
    process = args.operation not in ["segment", "extract", "onset", "beat",
                                           "decomp", "proxim", "rename", "concat"]
    # store these as they will be adjusted for short signals
    n_fft = args.n_fft
    hop_size = args.hop_size
    # for use with rythm features, otherwise window length = n_fft
    window_length = args.window_length = 384
    n_bins = args.n_bins = 84
    n_mels = args.n_mels = 128

    debug.log_info('Loading...')
    files = load_files(args)

    if args.operation == "concat":
        if args.concat_target is None:
            debug.log_error('Concat target not provided!')
            sys.exit(1)
        args.input = files
        args.sample_rate = sf.info(files[0]).samplerate
        craft = Konkat(args)
        craft.main()
        return

    warnings = {}
    errors = {}

    if process:
        batch = True
        if len(files) == 1:
            batch = False
        from .processor import Processor  
        processor = Processor(args, mode='render', batch=batch)

    count = 0
    for file in files:
        args.input = file
        
        
        if dsp:
            try:
                if process:
                    args.y, args.sample_rate = sf.read(file, dtype='float32')
                    args.meta_data = metadata.generate_metadata(file, args)
                    args.output = args.input
                else:
                    args.y=librosa.load(file, sr=args.sample_rate)[0]
            except RuntimeError:
                debug.log_error(f'Could not load {file}!')
                continue
            if not librosa.util.valid_audio(args.y):
                debug.log_error(f'{file} is not a valid audio file!')
                sys.exit()
            args.num_samples = args.y.shape[-1]
            args.duration = args.num_samples / args.sample_rate
            if args.no_resolution_adjustment is False:
                debug.log_info('Adjusting analysis resolution for short signal...')
                args.n_fft, args.hop_size, args.window_length, args.n_bins, args.n_mels = utils.adjust_anal_res(args)
            args.num_frames = int(args.num_samples / args.hop_size)

        if args.operation == "segment":
            debug.log_info(f'<Segmenting> {file}')
            craft = Segmentor(args)
            craft.main()
        elif args.operation == "extract":
            debug.log_info(f'<Extracting features> for {file}')
            craft = Extractor(args)
            errors, warnings = craft.main()
        elif args.operation == "onset":
            debug.log_info(f'Detecting <onsets> for {file}')
            craft = OnsetDetector(args)
            craft.main()
        elif args.operation == "beat":
            debug.log_info(f'Detecting <beats> for {file}')
            craft = BeatDetector(args)
            craft.main()
        elif args.operation == "decomp":
            debug.log_info(f'<Decomposing> {file}')
            craft = Decomposer(args, True)
            asyncio.run(craft.main())
        elif args.operation == "filter":
            debug.log_info(f'Applying <filter> to {file}')
            processor.filter(args.y, args.sample_rate, args.filter_frequency,
                             btype=args.filter_type)
        elif args.operation == "norm":
            debug.log_info(f'<Normalising> {file}')
            processor.normalise_audio(args.y, args.sample_rate, args.normalisation_level,
                                      args.normalisation_mode)
        elif args.operation == "fade":
            debug.log_info(f'Applying> <fade to {file}')
            processor.fade_io(args.y, args.sample_rate, args.fade_in,
                              args.fade_out, args.curve_type)
        elif args.operation == "trim":
            debug.log_info(f'<Trimming> {file}')
            processor.trim()
        elif args.operation == "pan":
            debug.log_info(f'<Panning> {file}')
            processor.pan(args.y, args.pan_amount, args.mono)
        elif args.operation == "split":
            debug.log_info(f'<Splitting> {file}')
            processor.split(args.y, args.sample_rate, args.split_points)
        elif args.operation == "rename":
            debug.log_info(f'<Renaming> {file}')
            utils.rename_file(file, args.rename, args.parent_folder, count+1)

        else:
            if args.operation == "wmeta":
                debug.log_info('Writing metadata')
                if args.meta_file:
                    args.meta = utils.load_json(args.meta_file)
                else:
                    debug.log_error('No metadata file provided!')
                    sys.exit()
                metadata.write_metadata(file, args.meta)
            if args.operation == "rmeta":
                debug.log_info('Extracting metadata')
                metadata.extract_metadata(file)

        utils.print_seperator()
        args.n_fft = n_fft
        args.hop_size = hop_size
        args.window_length = window_length
        args.n_bins = n_bins
        args.n_mels = n_mels

        count += 1

    utils.print_end()
    debug.log_done(f'<{args.operation}>')
    if len(warnings) > 0:
        debug.log_warning(f'Finished with <{len(warnings)} warning(s)>:')
        for k in warnings.keys():
            for w in warnings[k]:
                debug.log_warning(f'{k}: {w.message}. <Line {w.lineno} in file:> {w.filename}')
            
    if len(errors) > 0:
        debug.log_error(f'Finished with <{len(errors)} error(s)>:')
        for k in errors.keys():
            for e in errors[k]:
                debug.log_error(f'{k}: <{e}>')

def load_files(args):
    """
    Load audio files from a given input file or directory.

    Args:
        input_file (str): The path to the input file or directory.

    Returns:
        list: A list of valid audio file paths.

    Raises:
        SystemExit: If no valid files are found or if the input file or directory is invalid.
    """
    dsp = args.operation not in ["wmeta", "info", "proxim"]
    meta = args.operation in ["wmeta", "rmeta", "info", "proxim"]
    input_file = args.input
    files = []
    
    if input_file is None or input_file == '':
        debug.log_error('No input file or directory provided!')
        sys.exit()
    if input_file == '.':
        input_file = os.getcwd()
    # check if dir is home dir
    if input_file == os.path.expanduser('~'):
        debug.log_warning('You are about to process your home directory. Are you sure you want to continue?')
        user_input = input_file('\n1) Yes\n2) No\n')
        if user_input.lower() == '2':
            sys.exit(1)
    if os.path.isdir(input_file):
        input_dir = input_file
        # check to see if dir has subdirs
        if len(os.listdir(input_file)) > 0:
            for file in os.listdir(input_file):
                f = os.path.join(input_file, file)
                if os.path.isdir(f):
                    if not meta and os.path.basename(f) == 'wavecraft_data':
                        continue
                    debug.log_warning(f'Found subdirectory: <{file}>')
                    user_input = input(f'\n{colors.GREEN}Choose an action:{colors.ENDC}\
                        \n1) Process files in this subdirectory only \
                        \n2) Process files in all subdirectories \
                        \n3) Skip subdirectory \
                        \n4) Exit\n')
                    if user_input.lower() == '4':
                        debug.log_info('Exiting...')
                        sys.exit()
                    if user_input.lower() == '1':
                        for subfile in os.listdir(os.path.join(input_file, file)):
                            if utils.check_format(subfile, args.operation):
                                files.append(os.path.join(input_dir, file, subfile))
                    elif user_input.lower() == '2':
                        debug.log_info('Searching for files in all subdirectories...')
                        if args.operation == 'rename':
                            # only add the parent folder if renaming
                            files.append(os.path.join(input_dir, file))
                        for root, _, sub_files in os.walk(input_dir):
                            for file in sub_files:
                                if utils.check_format(file, args.operation):
                                    files.append(os.path.join(root, file))
                        break
                    else:
                        continue
                else:
                    if utils.check_format(file, args.operation):
                        files.append(os.path.join(input_dir, file))
        else:
            debug.log_error('No files found!')
            sys.exit()

    # single file
    else:
        if utils.check_format(input_file, dsp):
            files.append(input_file)
    if len(files) == 0:
        debug.log_error('No valid files found!')
        sys.exit()
    return files
