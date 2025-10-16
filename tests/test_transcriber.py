from transcribe import TranscriberWorker


def test_transcriber_params():
    worker = TranscriberWorker()
    worker.set_params(task="translate", language="es")
    assert worker._task == "translate"
    assert worker._language == "es"
