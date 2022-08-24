import os
import time
from speechbrain.pretrained import VAD
import torchaudio
import re


def _get_audio_info(audio_file):
    """
        Returns the sample rate and the length of the input audio file.

        Arguments
    ---------
        audio_file: path
            Path of the audio file containing the recording.
        threshold_number: float
            A float type number that will be used to get the voice intervals.
    """

    # Getting the total size of the input file
    metadata = torchaudio.info(audio_file)
    sample_rate = metadata.sample_rate
    audio_len = metadata.num_frames
    return sample_rate, audio_len


def save_boundaries(boundaries, save_path, audio_file):
    """Saves the boundaries on a file (and/or prints them)  in a readable format.

    Arguments
    ---------
    boundaries: torch.tensor
        Tensor containing the speech boundaries. It can be derived using the
        get_boundaries method.
    save_path: path
        When to store the text file containing the speech/non-speech intervals.
    audio_file: path
        Path of the audio file containing the recording. The file is read
        with torchaudio. It is used here to detect the length of the
        signal.
    """
    # Create a new file if needed
    if save_path is not None:
        f = open(save_path, mode="w", encoding="utf-8")

    # Getting the total size of the input file
    if audio_file is not None:
        sample_rate, audio_len = _get_audio_info(audio_file)
        audio_len = audio_len / sample_rate

    # Printing speech and non-speech intervals
    for i in range(boundaries.shape[0]):
        begin_value = boundaries[i, 0]
        end_value = boundaries[i, 1]

        if save_path is not None:
            f.writelines("%0.2f\t" % begin_value + "%0.2f\t" %
                         end_value + "voice\n")

    if save_path is not None:
        f.close()


def speechbrain(audio_file, threshold, wav, report_path, speechbrain_type, trained_models_path_speechbrain):
    """Fetch and load the speechbrain model, based on the epochs training
        From the speecbrain_type we extract the number of epochs
        Afterwards, we "declare" the class based on the folder with the name:
                no_epochs + epochs
        In this file should be hyperparams.yaml + model.ckpt + normalizer.ckpt + optimizer.ckpt

        The hyperparams file should contain a "modules" key, which is a
        dictionary of torch modules used for computation.

        The hyperparams file should contain a "pretrainer" key, which is a
        speechbrain.utils.parameter_transfer.Pretrainer

        Arguments
        ---------
        audio_file : str
            The location to use for finding the audio file
        threshold : float
            The threshold
        wav : str
            The name of the wav file
        report_path : str
            A pathto the generated report file where to save
        speechbrain_type : str
            The name of the speechbrain model that contains the no of epochs in it
            ex: speechbrain_3
                this is a speechbrain model trained on 3 epochs
        """
    # extract the number of epochs that the speechbrain model has
    temp = re.findall(r"\d+", speechbrain_type)
    # res is a list with only one element
    res = list(map(int, temp))

    # start measure the time
    start = time.time()

    # 0- Instantiating the class, baed on the pretrained model
    speechbrain_model_type = VAD.from_hparams(
        source=trained_models_path_speechbrain + str(res[0]) + "_epochs",
        savedir="pretrained_models/" + str(res[0]) + "_epochs",
    )

    # 1- Let's compute frame-level posteriors first
    prob_chunks = speechbrain_model_type.get_speech_prob_file(audio_file)

    # 2- Let's apply a threshold on top of the posteriors
    prob_th = speechbrain_model_type.apply_threshold(
        prob_chunks, threshold, threshold - threshold / 100
    )

    # 3- Let's now derive the candidate speech segments
    boundaries = speechbrain_model_type.get_boundaries(prob_th)

    # 4- Apply energy VAD within each candidate speech segment (optional)
    boundaries = speechbrain_model_type.energy_VAD(
        audio_file, boundaries, threshold, threshold - threshold / 100
    )

    # 5- Merge segments that are too close
    boundaries = speechbrain_model_type.merge_close_segments(
        boundaries, close_th=0.033)

    # 6- Remove segments that are too short
    boundaries = speechbrain_model_type.remove_short_segments(
        boundaries, len_th=0.1)

    # 7- Double-check speech segments (optional).
    boundaries = speechbrain_model_type.double_check_speech_segments(
        boundaries, audio_file, threshold
    )

    # the vad is done, stop the time
    end = time.time()

    # find the total vad time
    total_time = end - start

    # Print the output
    save_boundaries(
        boundaries,
        report_path
        + "/"
        + wav
        + "_report_file_"
        + speechbrain_type
        + "_"
        + "%0.2f" % threshold
        + ".txt",
        audio_file,
    )

    return total_time
