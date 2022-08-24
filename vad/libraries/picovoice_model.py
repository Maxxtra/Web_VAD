import struct
import wave
import pvcobra
import wave
import time


def read_file(file_name, sample_rate):
    wav_file = wave.open(file_name, mode="rb")
    channels = wav_file.getnchannels()
    num_frames = wav_file.getnframes()

    if wav_file.getframerate() != sample_rate:
        raise ValueError(
            "Audio file should have a sample rate of %d. got %d"
            % (sample_rate, wav_file.getframerate())
        )

    samples = wav_file.readframes(num_frames)
    wav_file.close()

    frames = struct.unpack("h" * num_frames * channels, samples)

    if channels == 2:
        print(
            "Picovoice processes single-channel audio but stereo file is provided. Processing left channel only."
        )

    return frames[::channels]


def picovoice(audio_path, threshold, wav, report_path):
    """"
        Reads the file and prints in an output file the intervals of the voice activity.
        Based on the threshold it extracts the begining and the end of the voice activity.

         Arguments
        ---------
        audio_file: path
            Path of the audio file containing the recording.
        threshold_number: float
            A float type number that will be used to get the voice intervals.
    """
    cobra = pvcobra.create(
        library_path=pvcobra.LIBRARY_PATH,
        access_key="RIlWhK2OeAU3Co5J5fxr+Vohx8P+ZNvX6vi6maLjdEY314Tk6suMKQ==",
    )

    # start measure the time
    start = time.time()

    audio = read_file(audio_path, cobra.sample_rate)

    num_frames = len(audio) // cobra.frame_length

    seconds, intervals = ([] for i in range(2))

    for i in range(num_frames):
        frame = audio[i * cobra.frame_length : (i + 1) * cobra.frame_length]
        result = cobra.process(frame)
        if result >= threshold:
            seconds.append(float(i * cobra.frame_length) / float(cobra.sample_rate))

    # the vad is done, stop the time
    end = time.time()

    # find the total vad time
    total_time = end - start

    with open(
        report_path
        + "/"
        + wav
        + "_report_file_picovoice_"
        + "%0.2f" % threshold
        + ".txt",
        "w",
    ) as rf:
        # check if there is no voice detected
        if seconds == []:
            return total_time

        intervals.append(seconds[0])

        for i in range(1, len(seconds) - 1):
            if (seconds[i] - seconds[i - 1] <= 0.033) and (
                seconds[i + 1] - seconds[i] > 0.1
            ):
                intervals.append(seconds[i])
                intervals.append(seconds[i + 1])

        intervals.append(seconds[-1])

        for i in range(0, len(intervals) - 1, 2):
            if intervals[i] != intervals[i + 1]:
                rf.writelines(
                    "%0.2f " % intervals[i] + " %0.2f" % intervals[i + 1] + "   voice\n"
                )
                # print(
                #     f" <{intervals[i]:0.2f} - {intervals[i+1]:0.2f}>  thresh: {threshold:0.2f}")

    return total_time
