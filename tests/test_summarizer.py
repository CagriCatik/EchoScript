import pytest

from summarizer import LlmParams, OllamaUnavailable, Summarizer


class DummySession:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

    def post(self, url, json, timeout):
        if self.should_fail:
            raise RuntimeError("network down")
        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"response": "Summary", "eval_count": 12}

        return Response()

    def get(self, url, timeout):
        raise RuntimeError("no get")


def test_llm_params_payload():
    params = LlmParams(stop=["END"])
    payload = params.to_payload()
    assert payload["stop"] == ["END"]


def test_summarizer_retry_failure():
    summ = Summarizer(session=DummySession(should_fail=True))
    with pytest.raises(OllamaUnavailable):
        summ.summarize("text", LlmParams(), "model", "en", False)


def test_summarizer_success():
    summ = Summarizer(session=DummySession())
    result = summ.summarize("text", LlmParams(), "model", "en", False)
    assert result.summary_text == "Summary"
    assert result.tokens_out == 12
