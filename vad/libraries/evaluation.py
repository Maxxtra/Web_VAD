from interval import interval


def load_label(filename):
    """
    Incarca un fisier label intr-o structura de intervale

    """
    valid_labels = ["speech", "voice", "male", "female"]

    with open(filename, "r") as labfile:
        txt_lines = labfile.readlines()
    interv = interval()
    for line in txt_lines:
        tokens = line.split()
        if tokens[2] in valid_labels:
            interv = interv | interval([float(tokens[0]), float(tokens[1])])
    return interval(interv)


def intervals_sum(interv_list):
    """
    Insumeaza toate intervalele dintr-un sir de intervale
    """
    sum = 0
    for interv in interv_list:
        sum += interv[1] - interv[0]
    return sum


def evaluation_process_multifile(report_file, GT_path):
    sum_vad = 0
    sum_tp = 0
    sum_gt = 0

    for file, gt in zip(report_file, GT_path):
        # print(file)
        # GT file
        sum_gt = sum_gt + intervals_sum(load_label(gt))

        # check if the report_file is emtpy, so return [0, 0]
        # if(os.stat(file).st_size == 0):
        #     return [0, 0]

        vad_interv = load_label(file)

        sum_vad_file = intervals_sum(vad_interv)

        # calculeaza TP
        tp = load_label(gt) & vad_interv
        sum_tp_file = intervals_sum(tp)

        sum_vad = sum_vad + sum_vad_file
        sum_tp = sum_tp + sum_tp_file

    recall = 100 * sum_tp / sum_gt

    if sum_vad == 0:
        precision = 0
    else:
        precision = 100 * sum_tp / sum_vad

    return [recall, precision]
