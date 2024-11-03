import pathlib

import numpy as np
import xarray as xr
from soundsig.sound import BioSound, WavFile


DATA_ROOT = pathlib.Path("./data-for-tests")
assert DATA_ROOT.exists()
OUTPUT_ROOT = pathlib.Path("./generated-test-data")


wav_paths = sorted(DATA_ROOT.glob("*wav"))

# adapted from https://github.com/theunissenlab/BioSoundTutorial/blob/master/BioSound1.ipynb
max_amp = 1.0

for wav_path in wav_paths:
    emitter = wav_path.name.split('_')[0]
    calltype = wav_path.name.split('-')[1]
    print(f"extracting features from file '{wav_path.name}',  emitter='{emitter}', call type='{calltype}'")

    soundwav = WavFile(wav_path, mono=True)
    biosound = BioSound(
        soundWave=soundwav.data.astype(float) / max_amp, fs=float(soundwav.sample_rate), 
        emitter=emitter, calltype=calltype
    )
    # we have to do spectroCalc first so that self.to is not None; it is used by fundest as the time vector
    biosound.spectroCalc(spec_sample_rate=1000, freq_spacing=50, min_freq=0, max_freq=10000)
    # calculate temporal envelope features
    biosound.ampenv(cutoff_freq=20, amp_sample_rate=1000)
    # calculate spectral envelope features
    biosound.spectrum(f_high=10000)
    # calculate fundamental frequency features
    biosound.fundest(maxFund=1500, minFund=300, lowFc=200, highFc=6000, 
                     minSaliency=0.5, debugFig=0, minFormantFreq=500, maxFormantBW=500,
                     windowFormant=0.1, method='Stack')

    features = {
        # tempora
        "mean_t": biosound.meantime,
        "std_t": biosound.stdtime,
        "skew_t": biosound.skewtime,
        "kurtosis_t": biosound.kurtosistime,
        "entropy_t": biosound.entropytime,
        "max_amp": biosound.maxAmp,
        # spectral
        "mean_s": biosound.meanspect,
        "std_s": biosound.stdspect,
        "skew_s": biosound.skewspect,
        "kurtosis_s": biosound.kurtosisspect,
        "entropy_s": biosound.entropyspect,
        "q1": biosound.q1,
        "q2": biosound.q2,
        "q3": biosound.q3,
        # fundamental
        "mean_f0": biosound.fund if not np.array_equal(biosound.fund, np.array([])) else np.nan,
        "mean_sal": biosound.meansal,
        "second_v": biosound.voice2percent,
        "pk2": biosound.fund2 if not np.array_equal(biosound.fund2, np.array([])) else np.nan,
        "max_fund": biosound.maxfund,
        "min_fund": biosound.minfund,
        "cv_fund": biosound.cvfund,
    }
    features = {
        ftr_name: np.array(ftr_val)[np.newaxis]
        for ftr_name, ftr_val in features.items()
    }

    channels = np.arange(1)  # because we set mono=True above we know there's just one channel
    data = xr.Dataset(
        {feature_name: (["channel"], feature_val) for feature_name, feature_val in features.items()},
        coords={"channel": channels},
    )
    data.to_netcdf(OUTPUT_ROOT / f"{wav_path.stem}.nc")
