import os
from xml.sax.handler import all_properties
import numpy as np
import glob
import warnings

# import the libraries coresponding the models
from vad.libraries.picovoice_model import picovoice
from vad.libraries.webrtc_model import webrtc
from vad.libraries.speechbrain_model import speechbrain
from vad.libraries.inaspeechsegmenter_model import (
    inaspeechsegmenter,
)

# ignore all warnings caused by inaspeechsegmenter
warnings.filterwarnings("ignore")
warnings.warn("DelftStack")
warnings.warn("Do not show this message")


def findfile(name, path):
    for dirpath, dirname, filename in os.walk(path):
        if name in filename:
            return os.path.join(dirpath, name)


def report_directory(folder_path):
    # Leaf directory
    directory = "report_files"

    # Parent Directories
    parent_dir = folder_path

    # Path
    path = os.path.join(parent_dir, directory)

    # Create the directory
    os.makedirs(path)

    return path


def main_course_detection(folder_path, VAD_type, trained_models_path_speechbrain):

    # create a directory for report files if there is none
    # first, check whether the specified
    # path exists or not
    isExist = os.path.exists(folder_path + "/report_files")
    if isExist == False:
        report_path = report_directory(folder_path)
    else:
        report_path = folder_path + "/report_files"

    # select the directory that you want to work
    os.chdir(folder_path)

    # select all the wav files
    my_files = glob.glob("*.wav")

    # for different thresholds given in a range 0.1 - 1.0 analise for each of the two VAD types
    # costruct two arrays: precision and recall after the evaluation and plot them
    # we will reuse the two arrays for the next VAD type
    for threshold_number in np.arange(0.0, 1.0, 0.1):
        vad_results = []

        # going through all the wav files from the directory
        for wav in my_files:
            # get the path to the wav file
            audio_path = folder_path + "/" + wav

            # define the aggressiveness for WebRTC VAD
            aggressiveness = -1

            if VAD_type == "picovoice":
                time_picovoice = picovoice(
                    audio_path, threshold_number, wav, report_path
                )
            elif VAD_type.find("speechbrain") != -1:
                time_speechbrain = speechbrain(
                    audio_path, threshold_number, wav, report_path, VAD_type, trained_models_path_speechbrain
                )
            elif VAD_type == "inaspeechsegmenter":
                time_inaspeechsegmenter = inaspeechsegmenter(
                    audio_path, threshold_number, wav, report_path, VAD_type
                )
            elif VAD_type == "webrtc":
                if 0.0 <= threshold_number and threshold_number < 0.31:
                    aggressiveness = 0
                elif 0.3 <= threshold_number and threshold_number < 0.61:
                    aggressiveness = 1
                elif 0.6 <= threshold_number and threshold_number < 0.81:
                    aggressiveness = 2
                elif 0.8 <= threshold_number and threshold_number < 1.1:
                    aggressiveness = 3
                # This is a more clever way, but I have a bug, so leaves it for now
                # switcher = {
                #     0.0 <= threshold_number and threshold_number < 0.31: 0,
                #     0.3 <= threshold_number and threshold_number < 0.61: 1,
                #     0.6 <= threshold_number and threshold_number < 0.81: 2,
                #     0.8 <= threshold_number and threshold_number < 1.1: 3
                # }

                # aggressiveness = switcher.get(True)

                time_webratc = webrtc(
                    audio_path, aggressiveness, wav, report_path)

            # concatenate the new folder that was made in a more generic way
            if aggressiveness != -1:
                # if there was a webRTC VAD type, so the value changed
                vad_results.append(
                    report_path
                    + "/"
                    + wav
                    + "_report_file_"
                    + VAD_type
                    + "_"
                    + "%0.1f" % aggressiveness
                    + ".txt"
                )
            else:
                vad_results.append(
                    report_path
                    + "/"
                    + wav
                    + "_report_file_"
                    + VAD_type
                    + "_"
                    + "%0.2f" % threshold_number
                    + ".txt"
                )

    return report_path
