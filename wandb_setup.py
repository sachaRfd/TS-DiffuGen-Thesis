import os

"""

This script should be used to setup your WandB environment
for training.

Please insert your WandB API Key inside the "".
A WandB account can be created through the following link:

https://wandb.ai/

The key should be available within your account settings.

"""


def setup_wandb():
    os.environ["WANDB_API_KEY"] = "Insert_API_Key_here"


if __name__ == "__main__":
    setup_wandb()
