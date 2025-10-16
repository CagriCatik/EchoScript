import numpy as np

from audio_io import RingBuffer, SAMPLE_RATE


def test_ring_buffer_write_read():
    rb = RingBuffer(SAMPLE_RATE, 1)
    data = np.arange(0, SAMPLE_RATE, dtype=np.int16)
    rb.write(data)
    out = rb.read_last(1.0)
    assert out.shape == data.shape
    assert np.all(out == data)


def test_ring_buffer_overrun_drops_oldest():
    rb = RingBuffer(SAMPLE_RATE, 1)
    data = np.arange(0, SAMPLE_RATE * 2, dtype=np.int16)
    rb.write(data)
    out = rb.read_last(1.0)
    assert out[-10:].tolist() == data[-SAMPLE_RATE:][-10:].tolist()
    assert rb.overrun_count >= 1
