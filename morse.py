"""
Dekoder CW.

Kluczowe cechy:
- Detektor progu per-sample (asymetryczne IIR peak/floor) - niezależny od
  rozmiaru bloku wejściowego.
- Stan (signal_on, samples_in_state) trwa między wywołaniami process().
- Klasyfikacja przerw ITU: 1d intra-char, 3d inter-char, 7d inter-word.
- Auto-WPM: 1D k-means na historii długości marków.
- Idle flush: cisza > 7d domyka bieżący znak bez czekania na kolejny mark.
"""
import numpy as np
from scipy import signal
from collections import deque


MORSE_TABLE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",", "..--..": "?", ".----.": "'",
    "-.-.--": "!", "-..-.": "/", "-.--.": "(", "-.--.-": ")",
    ".-...": "&", "---...": ":", "-.-.-.": ";", "-...-": "=",
    ".-.-.": "+", "-....-": "-", "..--.-": "_", ".-..-.": '"',
    "...-..-": "$", ".--.-.": "@",
}


class MorseDecoder:
    def __init__(self, fs, tone_freq=700.0, wpm_hint=20.0):
        self.fs = int(fs)
        self.tone_freq = float(tone_freq)
        self._design_filters()

        # detektor per-sample
        self._peak_state = 1e-4
        self._floor_state = 1e-6
        fs_f = float(self.fs)
        self._a_pa = 1.0 - np.exp(-1.0 / (fs_f * 0.001))
        self._a_pr = 1.0 - np.exp(-1.0 / (fs_f * 0.100))
        self._a_fu = 1.0 - np.exp(-1.0 / (fs_f * 0.100))
        self._a_fd = 1.0 - np.exp(-1.0 / (fs_f * 1.000))
        self._thr_ratio = 0.35

        self.min_dot = int(self.fs * 0.012)
        self.max_dot = int(self.fs * 0.300)
        self.dot_samples = int(self.fs * 1.2 / max(5.0, wpm_hint))
        self._recalc_thresholds()

        self.signal_on = False
        self.samples_in_state = 0
        self.mark_history = deque(maxlen=120)
        self.current_symbol = ""
        self.text = ""
        self.last_level = 0.0

    # -------------------------------------------------- filtry
    def _design_filters(self):
        fs = self.fs
        bw = 200.0
        low = max(50.0, self.tone_freq - bw / 2.0)
        high = min(fs / 2.0 - 200.0, self.tone_freq + bw / 2.0)
        self.bp_b, self.bp_a = signal.butter(
            4, [low / (fs / 2), high / (fs / 2)], btype="band")
        self.bp_zi = np.zeros(max(len(self.bp_a), len(self.bp_b)) - 1)
        self.env_b, self.env_a = signal.butter(
            2, 100.0 / (fs / 2), btype="low")
        self.env_zi = np.zeros(max(len(self.env_a), len(self.env_b)) - 1)

    def _recalc_thresholds(self):
        d = self.dot_samples
        self.mark_dot_max = int(d * 2.0)
        self.space_intra_max = int(d * 2.0)
        self.space_char_max = int(d * 5.0)
        self.idle_flush = int(d * 7.0)

    def set_tone_freq(self, freq):
        self.tone_freq = float(max(100.0, min(self.fs / 2.0 - 200.0, freq)))
        self._design_filters()

    def reset(self):
        self._peak_state = 1e-4
        self._floor_state = 1e-6
        self.signal_on = False
        self.samples_in_state = 0
        self.mark_history.clear()
        self.current_symbol = ""
        self.text = ""
        self.bp_zi[:] = 0
        self.env_zi[:] = 0
        self._recalc_thresholds()

    # -------------------------------------------------- detektor
    def _detect_thresholds(self, env):
        n = env.size
        thr = np.empty(n, dtype=np.float32)
        p = self._peak_state
        f = self._floor_state
        a_pa = self._a_pa
        a_pr = self._a_pr
        a_fu = self._a_fu
        a_fd = self._a_fd
        r = self._thr_ratio
        for i in range(n):
            e = float(env[i])
            if e > p:
                p += a_pa * (e - p)
            else:
                p += a_pr * (e - p)
            if e < f:
                f += a_fd * (e - f)
            else:
                f += a_fu * (e - f)
            thr[i] = f + r * (p - f)
        self._peak_state = p
        self._floor_state = f
        return thr

    # -------------------------------------------------- główna pętla
    def process(self, samples):
        x = np.asarray(samples, dtype=np.float32).ravel()
        if x.size == 0:
            return ""

        filtered, self.bp_zi = signal.lfilter(
            self.bp_b, self.bp_a, x, zi=self.bp_zi)
        env, self.env_zi = signal.lfilter(
            self.env_b, self.env_a, np.abs(filtered), zi=self.env_zi)

        thr = self._detect_thresholds(env)
        is_on_arr = (env > thr).astype(np.int8)
        n = is_on_arr.size

        extended = np.empty(n + 1, dtype=np.int8)
        extended[0] = 1 if self.signal_on else 0
        extended[1:] = is_on_arr

        transitions = np.flatnonzero(np.diff(extended))

        new_text = ""
        prev_logical = 0
        for t in transitions:
            logical_end = t + 1
            if prev_logical == 0:
                samples_in_block = max(0, logical_end - 1)
            else:
                samples_in_block = max(0, logical_end - prev_logical)
            total_dur = self.samples_in_state + samples_in_block
            if self.signal_on:
                self._end_mark(total_dur)
            else:
                new_text += self._end_space(total_dur)
            self.signal_on = not self.signal_on
            self.samples_in_state = 0
            prev_logical = logical_end

        if prev_logical == 0:
            samples_in_block = n
        else:
            samples_in_block = n + 1 - prev_logical
        self.samples_in_state += samples_in_block

        if (not self.signal_on and self.current_symbol
                and self.samples_in_state > self.idle_flush):
            new_text += self._end_space(self.samples_in_state)
            self.samples_in_state = 0

        self.last_level = float(env[-1])
        return new_text

    def flush(self):
        if not self.current_symbol:
            return ""
        return self._end_space(self.idle_flush + 1)

    # -------------------------------------------------- klasyfikacja
    def _end_mark(self, duration):
        if duration < self.min_dot * 0.5:
            return
        self.mark_history.append(duration)
        self._update_dot_estimate()
        if duration < self.mark_dot_max:
            self.current_symbol += "."
        else:
            self.current_symbol += "-"

    def _end_space(self, duration):
        if not self.current_symbol:
            return ""
        if duration < self.space_intra_max:
            return ""
        ch = MORSE_TABLE.get(self.current_symbol, "?")
        if duration < self.space_char_max:
            self.text += ch
            self.current_symbol = ""
            return ch
        self.text += ch + " "
        self.current_symbol = ""
        return ch + " "

    def _update_dot_estimate(self):
        if len(self.mark_history) < 6:
            return
        marks = np.sort(np.array(self.mark_history, dtype=float))
        n = marks.size
        best_i = None
        best_score = float("inf")
        for i in range(2, n - 1):
            c1 = marks[:i]
            c2 = marks[i:]
            score = c1.var() * len(c1) + c2.var() * len(c2)
            if score < best_score:
                best_score = score
                best_i = i
        if best_i is None:
            return
        dot_est = float(marks[:best_i].mean())
        dash_est = float(marks[best_i:].mean())
        if dash_est < dot_est * 1.4:
            return
        alpha = 0.4
        new_dot = int((1 - alpha) * self.dot_samples + alpha * dot_est)
        self.dot_samples = max(self.min_dot, min(self.max_dot, new_dot))
        self._recalc_thresholds()

    @property
    def wpm(self):
        if self.dot_samples <= 0:
            return 0.0
        return 1.2 / (self.dot_samples / self.fs)
