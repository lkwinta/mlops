import pytest
from settings import Settings

def test_settings():
    settings = Settings()

    assert settings.APP_NAME == "MyApp-test"
    assert settings.ENVIRONMENT == "test"
    assert settings.GPT_API_KEY == "test_api_key"
