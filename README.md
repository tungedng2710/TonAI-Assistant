<p align="center">
  <img src="stuffs/splash.png" alt="Description" style="width: 20%;">
</p>

# TonAI-Assistant
The virtual assistant is in the progress of development, you can try it with limited features

## Installation
### Step 1: Install Anaconda
#### Ubuntu
1. Download the Anaconda installer script:
    ```bash
    wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
    ```
2. Run the installer script:
    ```bash
    bash Anaconda3-2023.03-Linux-x86_64.sh
    ```
3. Follow the prompts and restart your terminal or run:
    ```bash
    source ~/.bashrc
    ```
#### macOS
1. Download the Anaconda installer:
    - Visit [Anaconda Downloads](https://www.anaconda.com/products/distribution#download-section) and download the macOS installer.
2. Run the installer:
    ```bash
    bash ~/Downloads/Anaconda3-2023.03-MacOSX-x86_64.sh
    ```
3. Follow the prompts and restart your terminal or run:
    ```bash
    source ~/.bash_profile
    ```
#### Windows
1. Download the Anaconda installer:
    - Visit [Anaconda Downloads](https://www.anaconda.com/products/distribution#download-section) and download the Windows installer.
2. Run the installer and follow the prompts to complete the installation.
3. Open the Anaconda Prompt from the Start menu.

### Step 2: Create and Activate Conda Environment

1. Open your terminal (Anaconda Prompt on Windows).
2. Create a new conda environment named 'tonai' with Python 3.10:
    ```bash
    conda create -n tonai python=3.10
    ```
3. Activate the environment:
    ```bash
    conda activate tonai
    ```
### Step 3: Install Required Packages

1. Ensure you have `requirement.txt` file in your working directory.
2. Install the required packages using pip:
    ```bash
    pip install -r requirement.txt
    ```

## Run your Virtual Assistant
Create the json file ```utils/bot_info.json```. Refer the format of this [JSON_FILE](https://github.com/tungedng2710/TonAI-Assistant/blob/main/utils/bot_info_example.json)
To run the Telegram bot from your Terminal:
```
python3 main_bot.py
```

To run the virtual assistant (command line), from your Terminal:
```
python3 assistant_cli.py
```

## More from TonAI Research:
* [TonAI Creative](https://github.com/tungedng2710/TonAI-Creative)
* [TonAI Telegram Bot](https://github.com/tungedng2710/TonAI-Telegram)
