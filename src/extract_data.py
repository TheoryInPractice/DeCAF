#!/usr/bin/env python3
"""
Extracts experiment summary data from the pkl files in the given directory.

Usage:
(extracting decomposition results)
src/extract_data.py data/finaldata > decomp.csv

(extracting kernelization results)
src/extract_data.py --type=kernel data/kernels > kernel.csv

(extracting preprosessing results)
src/extract_data.py --type=preprocess data/post_preprocessing > preprocess.csv
"""

__version__ = '0.0.1'

# imports standard libraries
import sys
import argparse
import pathlib
import pickle

# imports extrenal libraries
try:
    import pandas as pd
except ImportError:
    sys.stderr.writelines([
        'Failed to load modules.\n',
        'Please run: pip install pandas\n',
    ])
    sys.exit(1)


def get_parser():
    """Argument parser."""

    parser = argparse.ArgumentParser(description='{{ program_description }}')
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--type', choices=['decomp', 'kernel', 'preprocess'], default='decomp',
                        help='type of the output file (default: decomp)')
    parser.add_argument('target', nargs='+', help='target file or directory')
    return parser


def parse_dir_name(dir_name: str, output_type: str):
    if dir_name == 'post_preprocessing':
        return dir_name, None, None, None

    exp_type = dir_name.split('_', 1)[0]

    kernel_ordering = 'push_front'
    if '_arb' in dir_name:
        kernel_ordering = 'arbitrary'
    elif '_first' in dir_name:
        kernel_ordering = 'keep_first'
    elif '_last' in dir_name:
        kernel_ordering = 'keep_last'
    elif '_back' in dir_name:
        kernel_ordering = 'push_back'

    if output_type == 'decomp':
        kernel_version = 2 if '_kv2' in dir_name else 1
        perf_version = 3 if '_pv3' in dir_name else 2 if '_pv2' in dir_name else 1
    elif output_type == 'kernel':
        kernel_version = 2 if '_v2' in dir_name else 1
        perf_version = None
    return exp_type, kernel_version, kernel_ordering, perf_version


def parse_file_name(basename: str):
    """
    18_TF_maxcweight4_kdistinct19_seed0_witer3_ktotal24_n398.pkl
    10_LV_kdistinct11_ktotal78_threshold6_scalefac4_seed0_maxcweight11_n1184
    """
    tokens = basename.split('_')
    kdistinct_orig = [int(token.replace('kdistinct', '')) for token in tokens if token.startswith('kdistinct')][0]
    seed = [int(token.replace('seed', '')) for token in tokens if token.startswith('seed')][0]
    n_orig = int(tokens[-1].replace('n', ''))
    return seed, kdistinct_orig, n_orig


def extract_decomp_data(data: dict):
    return dict(
        kinput=data['kinput'],
        time_bsd=data['time_bsd'],
        passed=data['passed_bsd'],
        reconstructs=data['reconstructs'],
        num_lp_runs=data['num_lp_runs'],
    )


def extract_kernel_data(data: dict):
    return dict(
        kinput=data['kinput'],
        time_kernel=data['kernel_time'],
        passed=data['passed_kernel'],
        n_removed=len(data['removal_vertices']),
        n=data['n'],
        m=data['m'],
        num_total_blocks=data['num_total_blocks'],
        num_reduced_blocks=data['num_reduced_blocks'],
    )

def parse_file(path: pathlib.Path, output_type: str):
    fullpath = path.absolute()
    basename = path.stem
    seed, kdistinct_orig, n_orig = parse_file_name(basename)
    exp_type, kernel_version, kernel_ordering, perf_version = parse_dir_name(fullpath.parent.parent.name, output_type)

    if output_type == 'decomp' and exp_type != 'finaldata':
        raise ValueError(f'Unsupported experiment type: {exp_type}')
    elif output_type == 'kernel' and exp_type != 'kernels':
        raise ValueError(f'Unsupported experiment type: {exp_type}')
    elif output_type == 'preprocess' and exp_type != 'post_preprocessing':
        raise ValueError(f'Unsupported experiment type: {exp_type}')

    # load file
    with open(path, 'rb') as infile:
        data = pickle.load(infile)

    record = []
    if output_type == 'preprocess':
        preprocess_data = dict(
            basename=basename,
            seed=seed,
            n_orig=n_orig,
            kdistinct_orig=kdistinct_orig,
            kdistinct=data['post_preprocessing']['kdistinct'],
            time_preprocess=data['post_preprocessing']['preprocess_time'],
            n=data['post_preprocessing']['n'],
            m=data['post_preprocessing']['m'],
        )
        record += [preprocess_data]
    else:
        if output_type == 'decomp':
            if data['decomp_data']['bswd_dw_lp']['true_distinct'] is None:
                # we require to have the true-distinct expeirment data
                return []

            num_threads = data['num_threads']
            timeout = data['decomp_data']['timeout']
            kdistinct = data['decomp_data']['bswd_dw_lp']['true_distinct']['kinput']
            common_data = dict(
                basename=basename,
                kernel_version=kernel_version,
                kernel_ordering=kernel_ordering,
                perf_version=perf_version,
                num_threads=num_threads,
                timeout=timeout,
                seed=seed,
                n_orig=n_orig,
                kdistinct_orig=kdistinct_orig,
                kdistinct=kdistinct,
            )
        elif output_type == 'kernel':
            if data['true_distinct_kernel'] is None:
                # we require to have the true-distinct expeirment data
                return []

            kdistinct = data['true_distinct_kernel']['kinput']
            common_data = dict(
                basename=basename,
                kernel_version=kernel_version,
                kernel_ordering=kernel_ordering,
                seed=seed,
                n_orig=n_orig,
                kdistinct_orig=kdistinct_orig,
                kdistinct=kdistinct,
            )

        # input k = ground-truth k
        true_data = common_data.copy()
        true_data['feasible'] = True
        if output_type == 'decomp':
            true_data.update(**extract_decomp_data(data['decomp_data']['bswd_dw_lp']['true_distinct']))
        elif output_type == 'kernel':
            true_data.update(**extract_kernel_data(data['true_distinct_kernel']))

        record += [true_data]

        # varied input k
        if output_type == 'decomp':
            guesses = data['decomp_data']['bswd_dw_lp']['guesses']
        elif output_type == 'kernel':
            guesses = data['guess_kernels']

        for guess in guesses.values():
            if guess is None:
                continue

            guess_data = common_data.copy()
            if output_type == 'decomp':
                guess_data.update(**extract_decomp_data(guess))
            elif output_type == 'kernel':
                guess_data.update(**extract_kernel_data(guess))
            guess_data['feasible'] = kdistinct <= guess_data['kinput']
            record += [guess_data]

    return record


def main(args):
    """Entry point of the program. """

    # main logic
    records = []
    for target in args.target:
        # traverse all subdirectries
        for pkl_file_path in pathlib.Path(target).glob('**/*.pkl'):
            records += parse_file(pkl_file_path, args.type)

    df = pd.DataFrame.from_records(records)
    df.index.name = 'index'
    print(df.to_csv(index=False))


if __name__ == '__main__':
    main(get_parser().parse_args())
