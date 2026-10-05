from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    ENVIRONMENT: str
    APP_NAME: str
    GPT_API_KEY: str

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, value):
        if value not in ["dev", "prod", "test"]:
            raise ValueError("ENVIRONMENT must be one of 'dev', 'prod', or 'test'")

        return value

    @field_validator("GPT_API_KEY")
    @classmethod
    def validate_gpt_api_key(cls, value):
        if not value:
            raise ValueError("GPT_API_KEY must be set in the environment variables")
        return value
