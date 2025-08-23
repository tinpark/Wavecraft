"""
This module contains functions for extracting, generating, and writing metadata to audio files.
"""
import os
import subprocess
import sys
import tempfile
import json
import time
from .debug import Debug as debug


#######################
# Metadata
#######################
def extract_metadata(input_file):
    """
    Extracts metadata from an audio file using ffprobe.
    """
    command = [
        'ffprobe', input_file, '-v', 'error', '-show_entries',
        'format_tags', '-of', 'json'
    ]
    output = subprocess.check_output(command, stderr=subprocess.DEVNULL, universal_newlines=True)
    if 'not found' in output:
        debug.log_error('ffmpeg is not installed. Please install it if you want to copy the metadata over.')
        return None

    metadata = json.loads(output)
    return metadata.get('format', {}).get('tags', {})  # Returns a dictionary of metadata tags


def generate_metadata(input_file, args):
    prev_metadata = extract_metadata(input_file)
    craft_data = _get_craft_metadata(args)

    # Ensure craft_data is a dictionary before combining
    final_metadata = _concat_metadata(prev_metadata, craft_data)

    return final_metadata


def write_metadata(input_file, metadata):
    """
    Writes metadata to an audio file.

    Args:
        input_file (str): The path to the input audio file.
        metadata (dict): A dictionary containing metadata fields and their values.

    Returns:
        None
    """
    if input_file.endswith('.json'):
        debug.log_warning('Cannot write metadata to a JSON file, skipping...')
        return

    # Build ffmpeg command arguments for each metadata field
    metadata_args = []
    for key, value in metadata.items():
        metadata_args.extend(['-metadata', f'{key}={value}'])

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        command = [
            'ffmpeg', '-v', 'quiet', '-y', '-i', input_file, '-codec', 'copy'
        ] + metadata_args + [tmp_file.name]
        subprocess.run(command)
        os.replace(tmp_file.name, input_file)


def export_metadata(data, output_path, operation, suffix='metadata'):
    """
    Export metadata to a JSON file.

    Args:
        data (dict): The metadata dictionary to be exported.
        output_path (str): The path where the JSON file will be saved.
        operation (str): The operation (for example, 'segment') being performed.
        suffix (str): The suffix to be appended to the JSON file name.

    Returns:
        None
    """
    output_path = os.path.realpath(output_path)
    meta_dir = os.path.join(os.path.dirname(output_path), 'wavecraft_data')

    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)

    output_file = f"{os.path.join(meta_dir, os.path.basename(output_path))}_{operation}_{suffix}.json"

    if os.path.exists(output_file):
        debug.log_warning(f'Overwriting JSON metadata {os.path.basename(output_file)}...')
    else:
        debug.log_info(f'Exporting JSON metadata {os.path.basename(output_file)}...')

    # Write dictionary directly to JSON
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)



#######################
# Private functions
#######################
def _get_craft_metadata(args):
    metadata = {}

    if args.operation == 'segment':
        metadata.update(_generate_segmentation_metadata(args))
    elif args.operation == 'extract':
        metadata.update(_generate_feature_extraction_metadata(args))
    elif args.operation == 'decompose':
        metadata.update(_generate_decomposition_metadata(args))
    elif args.operation == 'beat':
        metadata.update(_generate_beat_detection_metadata(args))
    elif args.operation == 'filter':
        metadata.update(_generate_filter_metadata(args))
    elif args.operation == 'norm':
        metadata.update(_generate_normalization_metadata(args))
    elif args.operation == 'fade':
        metadata.update(_generate_fade_metadata(args))
    elif args.operation == 'trim':
        metadata.update(_generate_trim_metadata(args))
    elif args.operation == 'pan':
        metadata.update(_generate_pan_metadata(args))
    elif args.operation == 'split':
        metadata.update(_generate_split_metadata(args))
    elif args.operation == 'proxim':
        metadata.update(_generate_proximity_metric_metadata(args))
    
    # Include audio settings in the metadata
    metadata.update(_generate_audio_settings_metadata(args))

    return metadata


def _stringify_dict(d, new_line=True):
    if new_line:
        return '\n'.join([f'{k}:{v}' for k, v in d.items()])
    else:
        return ', '.join([f'{k}:{v}' for k, v in d.items()])

def _concat_metadata(prev_metadata, craft_data):
    if prev_metadata is None:
        prev_metadata = {}

    # Update prev_metadata with craft_data, overwriting values in prev_metadata with values from craft_data
    for key, value in craft_data.items():
        
        if key in prev_metadata:
            # If there is a conflict in values, you could choose to resolve it:
            # perhaps by appending or choosing one value over the other
            prev_metadata[key] = f"{prev_metadata[key]}, {value}" if prev_metadata[key] != value else value
        else:
            prev_metadata[key] = value

    return prev_metadata


def _join_metadata(meta_list):
    meta_data = ''
    for meta in meta_list:
        meta_data+=str(meta)+'\n'
    return meta_data

# Function for each category to receive args and return the correct dictionary
def _generate_io_metadata(args):
    return {
        'input_text': args.input_text,
        'output_directory': args.output_directory,
        'save_txt': args.save_txt
    }

def _generate_audio_settings_metadata(args):
    return {
        'sample_rate': args.sample_rate,
        'fmin': args.fmin,
        'fmax': args.fmax,
        'n_fft': args.n_fft,
        'hop_size': args.hop_size,
        'spectogram': args.spectogram,
        'no_resolution_adjustment': args.no_resolution_adjustment
    }

def _generate_segmentation_metadata(args):
    return {
        'seg_method': args.segmentation_method,
        'seg_min_length': args.min_length,
        'seg_onset_threshold': args.onset_threshold,
        'seg_onset_envelope': args.onset_envelope,
        'seg_backtrack_length': args.backtrack_length
    }

def _generate_feature_extraction_metadata(args):
    return {
        'fex_extractor': args.feature_extractor,
        'fex_flatten_dict': args.flatten_dictionary
    }

def _generate_proximity_metric_metadata(args):
    return {
        'prox_n_similar': args.n_similar,
        'prox_identifier': args.identifier,
        'prox_class_to_analyse': args.class_to_analyse,
        'prox_metric_to_analyze': args.metric_to_analyze,
        'prox_test_condition': args.test_condition,
        'prox_ops': args.ops,
        'prox_n_max': args.n_max,
        'prox_metric_range': args.metric_range
    }

def _generate_decomposition_metadata(args):
    return {
        'decomp_n_components': args.n_components,
        'decomp_source_separation': args.source_separation,
        'decomp_sklearn': args.sklearn,
        'decomp_nn_filter': args.nn_filter
    }

def _generate_beat_detection_metadata(args):
    return {
        'beat_k': args.k
    }

def _generate_filter_metadata(args):
    return {
        'filter_frequency': args.filter_frequency,
        'filter_type': args.filter_type
    }

def _generate_normalization_metadata(args):
    return {
        'norm_level': args.normalisation_level,
        'norm_mode': args.normalisation_mode
    }

def _generate_metadata_metadata(args):
    return {
        'meta_file': args.meta_file
    }

def _generate_trim_metadata(args):
    return {
        'trim_range': args.trim_range,
        'trim_silence': args.trim_silence
    }

def _generate_split_metadata(args):
    return {
        'split_points': args.split_points
    }

def _generate_fade_metadata(args):
    return {
        'fade_in': args.fade_in,
        'fade_out': args.fade_out,
        'curve_type': args.curve_type
    }

def _generate_pan_metadata(args):
    return {
        'pan_amount': args.pan_amount,
        'mono': args.mono
    }
        
