Create a Cobra environment Pyton=3.8 from Scratch, not from base. If you are on base, conda deactivate
and from there create a new environment: conda create --name NameOfTheEnvironment python=3.8

Follow the steps:

1. First of all, upgrade the pip
    pip install --upgrade pip

2. Install PyInterval, beacuse it only works if there is no module installed
    pip install pyinterval

3. Install Pytorch
    https://pytorch.org/
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

4. Install al the VADs one at a time:
    Speechbrain
        Get all the speechbrain methods, it is mandatory: git clone https://github.com/speechbrain/speechbrain.git
        pip install speechbrain
        # pip install --editable .

    Picovoice
        sudo pip3 install pvcobrademo

    Webratc
        pip install webrtcvad

    InaSpeechSegmenter
        sudo apt-get install ffmpeg
        pip install inaSpeechSegmenter

5. Install the library responsible with the graphs text
    AdjustText
        conda install -c conda-forge adjusttext

6. Install all the requirements
    pip install -r requirements.txt

7. (optional, if not already beeing installed)
    Install pvcobra
        pip install pcvobra

Explinations of the functionality of the code:
    Run python3 vad_evaluation.py and give the absolut path of the input_data folder

    You will see in the input_data subdirectories, ex test1 and test2. In these 
    you can just stock data, but the data that will be evaluate should be online in
    input_data
    ex:
        folder_path = "/home/alex/Documents/PY/evaluarea_vad/input_data" - is ok
        folder_path = "/home/alex/Documents/PY/evaluarea_vad/input_data/test1" - is wrong

