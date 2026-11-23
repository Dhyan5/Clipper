import numpy as np

from clipforge.audio import energy_between, silence_intervals


def test_silence_intervals_finds_gap():
    times = np.arange(0, 10, 0.05)
    rms = np.full_like(times, 0.3)
    rms[(times >= 4.0) & (times < 6.0)] = 0.001
    runs = silence_intervals(times, rms, 0.0, 10.0, min_silence=0.35)
    assert any(abs(s - 4.0) < 0.2 and abs(e - 6.0) < 0.2 for s, e in runs)


def test_energy_between():
    times = np.arange(0, 10, 0.05)
    rms = np.zeros_like(times)
    rms[(times >= 2.0) & (times < 4.0)] = 0.5
    assert energy_between(times, rms, 2.0, 4.0) > 0.4
    assert energy_between(times, rms, 6.0, 8.0) == 0.0
