from settings import SettingsRepo


def test_settings_clamp():
    repo = SettingsRepo()
    settings = repo.get_llm_settings()
    settings.temperature = 5.0
    settings.top_p = -1
    settings.top_k = -10
    settings.repeat_penalty = -1
    settings.num_predict = 1
    settings.num_ctx = 10
    repo.set_llm_settings(settings)
    new_settings = repo.get_llm_settings()
    assert new_settings.temperature <= 2.0
    assert new_settings.top_p >= 0.0
    assert new_settings.top_k >= 1
    assert new_settings.num_predict >= 16
    assert new_settings.num_ctx >= 128
