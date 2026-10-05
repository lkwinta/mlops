import os
import yaml
import argparse
from dotenv import load_dotenv
from settings import Settings


def export_envs(environment: str = "dev") -> None:
    file_name = f"config/.env.{environment}"
    load_dotenv(file_name)


def load_secrets(secret_file: str = "secrets.yaml") -> None:
    if os.path.exists(secret_file):
        with open(secret_file, "r") as file:
            secrets = yaml.safe_load(file)
            for key, value in secrets.items():
                os.environ[key] = value
    else:
        raise FileNotFoundError(f"Secrets file '{secret_file}' not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load environment variables from specified.env file."
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="dev",
        choices=["dev", "test", "prod"],
        help="The environment to load (dev, test, prod)",
    )
    parser.add_argument(
        "--secret", type=str, default="secrets.yaml", help="The secrets file to load"
    )
    args = parser.parse_args()

    export_envs(args.environment)
    load_secrets(args.secret)

    settings = Settings()

    print("APP_NAME: ", settings.APP_NAME)
    print("ENVIRONMENT: ", settings.ENVIRONMENT)
    print("GPT_API_KEY: ", settings.GPT_API_KEY)
