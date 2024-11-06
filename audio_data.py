from pathlib import Path
from numpy.typing import NDArray
import numpy as np
import sounddevice as sd
import audiofile
import soundfile
from soundfile import _ffi_types, _ffi, _snd, _error_check, SEEK_SET
import threading


# mp3の保存圧縮率変更
def _cdata_io(self, action, data, ctype, frames):
    """Call one of libsndfile's read/write functions."""
    assert ctype in _ffi_types.values()
    self._check_if_closed()
    if self.seekable():
        curr = self.tell()
    if action == "write":
        # ビットレート設定
        # SFC_SET_COMPRESSION_LEVEL (0x1301)
        # https://github.com/libsndfile/libsndfile/blob/c81375f070f3c6764969a738eacded64f53a076e/docs/command.md#sfc_set_compression_level
        pointer_compression_level = _ffi.new("double *")
        pointer_compression_level[0] = 0  # 0:低圧縮率  1:高圧縮率
        _snd.sf_command(self._file, 0x1301, pointer_compression_level, _ffi.sizeof(pointer_compression_level))
    func = getattr(_snd, "sf_" + action + "f_" + ctype)
    frames = func(self._file, data, frames)
    _error_check(self._errorcode)
    if self.seekable():
        self.seek(curr + frames, SEEK_SET)  # Update read & write position
    return frames


soundfile.SoundFile._cdata_io = _cdata_io


def db_to_amp(db: float, eps: float = 2**-16) -> float:
    amp = 10 ** (-np.abs(db) / 20 + np.log10(1 + eps)) - eps
    return np.copysign(amp, db)


def amp_to_db(amp: float, eps: float = 2**-16) -> float:
    db = 20 * (np.log10(np.abs(amp) + eps) - np.log10(1 + eps))
    return np.copysign(db, amp)


def compressor(signal: NDArray, compress_db: float, ratio: float = 10, knee_db: float = 5) -> None:
    # 参考 https://dsp.stackexchange.com/questions/28548/differences-between-soft-knee-and-hard-knee-in-dynamic-range-compression-drc
    compress_db = np.abs(compress_db)
    if compress_db - knee_db / 2 < 0:
        knee_db = compress_db * 2
    knee_lower = db_to_amp(compress_db + knee_db / 2)
    knee_upper = db_to_amp(compress_db - knee_db / 2)
    abs_signal = np.abs(signal)
    target_knee = (knee_lower <= abs_signal) & (abs_signal <= knee_upper)
    target_compress = knee_upper < abs_signal
    target_knee_db = -amp_to_db(abs_signal[target_knee])
    signal[target_knee] = np.copysign(
        db_to_amp(
            target_knee_db + (1 / ratio - 1) * (target_knee_db + compress_db + knee_db / 2) ** 2 / (2 * knee_db)
        ),
        signal[target_knee],
    )
    signal[target_compress] = np.copysign(
        db_to_amp(-compress_db + (-amp_to_db(abs_signal[target_compress]) + compress_db) / ratio),
        signal[target_compress],
    )


def limitter(signal: NDArray, limit_db: float, knee_db: float = 3, max_db: float = 1) -> None:
    limit_db = np.abs(limit_db)
    if limit_db - knee_db / 2 < 0:
        knee_db = limit_db * 2
    knee_lower = db_to_amp(limit_db + knee_db / 2)
    knee_upper = db_to_amp(limit_db - knee_db / 2)
    abs_signal = np.abs(signal)
    target_knee = (knee_lower <= abs_signal) & (abs_signal <= knee_upper)
    target_limit = knee_upper < abs_signal
    target_knee_db = -amp_to_db(abs_signal[target_knee])
    signal[target_knee] = np.copysign(
        db_to_amp(target_knee_db + (-1) * (target_knee_db + limit_db + knee_db / 2) ** 2 / (2 * knee_db)),
        signal[target_knee],
    )
    limit = db_to_amp(limit_db)
    signal[target_limit] = np.copysign(
        limit,
        signal[target_limit],
    )
    if limit_db != max_db:
        signal *= db_to_amp(max_db) / limit


class AudioData:
    def __init__(self, path: Path):
        self._path = path
        signal, sr = soundfile.read(path, dtype=np.float32)
        self._signal = signal
        self._sr = sr
        self._times = np.linspace(0, (len(self._signal) - 1) / sr, len(self._signal), dtype=np.float32)
        self._fig = None
        self._playing = False
        self._lock = threading.Lock()

    @property
    def signal(self) -> NDArray:
        return self._signal

    @property
    def sr(self) -> int:
        return self._sr

    @property
    def times(self) -> NDArray:
        return self._times

    @property
    def end_time(self) -> float:
        return len(self._signal) / self._sr

    @property
    def playing(self) -> bool:
        return self._playing

    def play(self, seconds: float):
        frame = int(round(seconds * self.sr))
        with self._lock:
            sd.play(self._signal[frame:], self._sr)
            self._playing = True
        print("playing")

    def stop(self):
        with self._lock:
            sd.stop()
            self._playing = False
        print("stopeed")

    def save(
        self,
        dst_path: Path,
        start: float,
        end: float,
        l_compress_db: float,
        l_limit_db: float,
        r_compress_db: float,
        r_limit_db: float,
    ):
        ratio = 10
        knee_db = 5
        l_limit_db = l_compress_db + (l_compress_db - knee_db) / ratio
        r_limit_db = r_compress_db + (r_compress_db - knee_db) / ratio
        dst_signal = self._signal[int(round(start * self._sr)) : int(round((self.end_time + end) * self._sr))].copy()
        compressor(dst_signal[:, 0], l_compress_db, ratio, knee_db)
        compressor(dst_signal[:, 1], r_compress_db, ratio, knee_db)
        limitter(dst_signal[:, 0], l_limit_db, knee_db, max_db=1)
        limitter(dst_signal[:, 1], r_limit_db, knee_db, max_db=1)
        audiofile.write(str(dst_path), dst_signal.transpose((1, 0)), self._sr)
