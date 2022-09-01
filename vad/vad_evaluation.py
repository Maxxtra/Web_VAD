import os
from xml.sax.handler import all_properties
import matplotlib.pyplot as plt
import numpy as np
import glob
import warnings
from matplotlib.figure import Figure
from adjustText import adjust_text

# import the library which makes the evaluation
from vad.libraries.evaluation import evaluation_process_multifile

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


def plotting(recall, precision, VAD_type, ax):
    ax.plot(
        recall,
        precision,
        linestyle="dashed",
        linewidth=3,
        marker="o",
        markersize=7,
        label=VAD_type,
    )

    # make a list with all the thresholds
    threshold_list = np.arange(0.0, 1.1, 0.1)
    threshold_list = ["%.1f" % elem for elem in threshold_list]

    # a list of texts that will annotate points
    texts = []

    # add labels to line plots
    for x, y, threshold_number in zip(recall, precision, threshold_list):
        if VAD_type == "webrtc":
            if 0.0 <= float(threshold_number) and float(threshold_number) < 0.31:
                texts.append(ax.text(x, y, 0.0))
            elif 0.3 <= float(threshold_number) and float(threshold_number) < 0.61:
                texts.append(ax.text(x, y, 0.1))
            elif 0.6 <= float(threshold_number) and float(threshold_number) < 0.81:
                texts.append(ax.text(x, y, 0.2))
            elif 0.8 <= float(threshold_number) and float(threshold_number) < 1.1:
                texts.append(ax.text(x, y, 0.3))
        else:
            texts.append(ax.text(x, y, threshold_number))

    # adjust the annotations
    # autoalign is xy to make the decision by itself how to align them
    adjust_text(
        texts,
        x=recall,
        y=precision,
        autoalign="xy",
        only_move={"points": "y", "text": "y"},
        force_points=0.15,
        arrowprops=dict(arrowstyle="->", color="r", lw=0.15),
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("VAD evaluation multifile")
    ax.legend()


def main_course(folder_path, VADs, trained_models_path_speechbrain):

    # create a directory for report files if there is none
    # first, check whether the specified
    # path exists or not
    isExist = os.path.exists(folder_path + "/report_files")
    if isExist == False:
        report_path = report_directory(folder_path)
    else:
        report_path = folder_path + "/report_files"

    # declare the figure that we will interpretate
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)

    # select the directory that you want to work
    os.chdir(folder_path)

    # select all the wav files
    my_files = glob.glob("*.wav")

    # declare the times for each VAD type
    # the time is calculated without printing the results into report files
    time_picovoice = time_speechbrain = time_inaspeechsegmenter = time_webratc = 0

    # declare two lists that gets all the precisions and recalls
    total_precision_recall = [
        [[], []],
        [[], []],
        [[], []],
        [[], []],
        [[], []],
        [[], []],
    ]

    for VAD_type in VADs:
        # declare two empty arrays
        precision, recall = ([] for i in range(2))

        # for different thresholds given in a range 0.1 - 1.0 analise for each of the two VAD types
        # costruct two arrays: precision and recall after the evaluation and plot them
        # we will reuse the two arrays for the next VAD type
        for threshold_number in np.arange(0.0, 1.0, 0.1):
            vad_results, gt_result = ([] for i in range(2))

            # going through all the wav files from the directory
            for wav in my_files:
                # get the path to the wav file
                audio_path = folder_path + "/" + wav

                # get the GT file which needs to have the same name with the wav file
                txt = wav.replace(".wav", ".txt")
                GT_path = folder_path + "/" + txt

                gt_result.append(GT_path)

                # check if there is a GT file
                if findfile(txt, folder_path) == None:
                    print("No GT file for the " + wav + ". Stop")
                    break

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

            # make a 2D array that has the name of the folders with VAD results and the GT inputs
            list_result = evaluation_process_multifile(vad_results, gt_result)

            recall.append(list_result[0])
            precision.append(list_result[1])

        # plotting(recall, precision, VAD_type, ax)

        # # convet each element into a string
        # [str(i) for i in recall]
        # [str(i) for i in precision]
        if VAD_type == "inaspeechsegmenter":
            total_precision_recall[0][0] = recall
            total_precision_recall[0][1] = precision
        elif VAD_type == "picovoice":
            total_precision_recall[1][0] = recall
            total_precision_recall[1][1] = precision
        elif VAD_type == "speechbrain_3":
            total_precision_recall[2][0] = recall
            total_precision_recall[2][1] = precision
        elif VAD_type == "speechbrain_10":
            total_precision_recall[3][0] = recall
            total_precision_recall[3][1] = precision
        elif VAD_type == "speechbrain_100":
            total_precision_recall[4][0] = recall
            total_precision_recall[4][1] = precision
        elif VAD_type == "webrtc":
            total_precision_recall[5][0] = recall
            total_precision_recall[5][1] = precision

    # # Plotting the time for each VAD

    # # create datasets
    # all_times = [
    #     time_picovoice,
    #     time_webratc,
    #     time_speechbrain,
    #     time_inaspeechsegmenter,
    # ]
    # bars_vadds = ("picovoice", "webrtc", "speechbrain", "inaspeechsegmenter")
    # x_pos = np.arange(len(bars_vadds))

    # plot2 = plt.figure(2)

    # # Create bars
    # plt.bar(
    #     x_pos, all_times, width=0.8, color=["black", "red", "green", "blue", "cyan"]
    # )

    # # Create names on the x-axis
    # plt.xticks(x_pos, bars_vadds)

    # return fig
    return total_precision_recall
