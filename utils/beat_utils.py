import numpy as np


# https://github.com/craffel/mir_eval/blob/main/mir_eval/beat.py


def get_offbeats(beats):
    offbeats = []
    for i in range(len(beats) - 1):
        offbeat = (beats[i] + beats[i + 1]) / 2
        offbeats.append(offbeat)
    return offbeats


def get_double_metrical_variation(beats):
    variation = []
    for i in range(len(beats) - 1):
        variation.append(beats[i])
        variation.append((beats[i] + beats[i + 1]) / 2)
    return variation


def get_half_odd_metrical_variation(beats):
    return beats[::2]


def get_half_even_metrical_variation(beats):
    return beats[1::2]


def get_metrical_level_variations_names():
    return ['Ground truth annotations',
            'Off beats',
            'Double beats',
            'Half odd beats',
            'Half even beats']


def get_metrical_level_variations(reference_beats):
    """Return metric variations of the reference beats

    Parameters
    ----------
    reference_beats : np.ndarray
        beat locations in seconds

    Returns
    -------
    reference_beats : np.ndarray
        Original beat locations
    off_beat : np.ndarray
        180 degrees out of phase from the original beat locations
    double : np.ndarray
        Beats at 2x the original tempo
    half_odd : np.ndarray
        Half tempo, odd beats
    half_even : np.ndarray
        Half tempo, even beats

    """
    off_beats = get_offbeats(reference_beats)
    double_beats = get_double_metrical_variation(reference_beats)
    half_odd_beats = get_half_odd_metrical_variation(reference_beats)
    half_even_beats = get_half_even_metrical_variation(reference_beats)
    return (reference_beats,
            off_beats,
            double_beats,
            half_odd_beats,
            half_even_beats)
