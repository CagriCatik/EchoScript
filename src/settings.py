from PySide6.QtCore import QSettings

ORG = "CATIK"
APP = "EchoScript"

KEY_MODEL = "model"
KEY_LANGUAGE = "language"
KEY_TRANSLATE = "translate_to_en"
KEY_MIC_INDEX = "mic_index"
KEY_SAVE_DIR = "save_dir"
KEY_OLLAMA_MODEL = "ollama_model"
KEY_AUTO_OPEN = "auto_open"
KEY_OVERLAY_ENABLED = "overlay_enabled"
KEY_OVERLAY_TEXT = "overlay_text"

def get_settings() -> QSettings:
    return QSettings(ORG, APP)
