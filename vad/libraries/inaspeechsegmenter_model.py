import time
from vad.inaSpeechSegmenter.inaSpeechSegmenter import Segmenter


def inaspeechsegmenter(audio_path, threshold_number, wav, report_path, VAD_type):

    # start measure the time
    start = time.time()

    # create an instance of speech segmenter
    # this loads neural networks and may last few seconds
    #  Warnings have no incidence on the results
    seg = Segmenter()

    # segmentation is performed using the __call__ method of the segmenter instance
    segmentation = seg(audio_path, threshold_number=threshold_number)

    #  the result is a list of tuples
    # each tuple contains:
    # * label in 'male', 'female', 'music', 'noEnergy'
    # * start time of the segment
    # # * end time of the segment
    # print(segmentation)

    for x in segmentation:
        for y in x:
            if y == "male" or y == "female":
                y = "voice"
            else:
                y = "noise"

    save_path = (
        report_path
        + "/"
        + wav
        + "_report_file_"
        + VAD_type
        + "_"
        + "%0.2f" % threshold_number
        + ".txt"
    )

    # the vad is done, stop the time
    end = time.time()

    # find the total vad time
    total_time = end - start

    f = open(save_path, mode="w", encoding="utf-8")

    for index, tuple in enumerate(segmentation):
        audio_type = tuple[0]
        begin_value = tuple[1]
        end_value = tuple[2]
        f.writelines("%0.2f " % begin_value + " %0.2f" %
                     end_value + "   voice\n")

    return total_time
