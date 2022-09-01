from io import BytesIO
from flask import send_file
import zipfile
from contextlib import redirect_stderr
from crypt import methods
from urllib import request
from flask import Flask, render_template, request, Response, redirect, url_for
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import os
import shutil
import time
from vad.vad_evaluation import main_course
from vad.vad_detection import main_course_detection

app = Flask(__name__)

vad_types = []
vad_type = ""
parent_dir = os.getcwd()


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")


@app.route("/evaluation", methods=["GET", "POST"])
def upload_files():
    if request.method == "POST":
        global vad_types

        # get the list of vad types selected by the user
        hidden_skills = request.form.get("hidden_skills")
        if hidden_skills is not None:
            vad_types = hidden_skills.split(",", 6)

        # make the directory where the files should save
        directory = "uploaded_files"
        global parent_dir
        path = os.path.join(parent_dir, directory)

        isExist = os.path.exists(path)
        if isExist == False:
            # Create the directory
            os.mkdir(path)

        app.config["UPLOAD_PATH"] = path

        # go through all the files
        for folder in request.files.getlist("folder_path"):
            folder.save(os.path.join(
                app.config["UPLOAD_PATH"], folder.filename))

        # fig = main_course(path, vad_types)

        # delete the files
        # shutil.rmtree(path)

        # output = io.BytesIO()
        # FigureCanvas(fig).print_png(output)
        # return Response(output.getvalue(), mimetype='image/png')

        # get the path to the trained_models for speechbrain
        trained_models_path_speechbrain = parent_dir + \
            "/vad/trained_models/"

        total_precision_recall = main_course(
            path, vad_types, trained_models_path_speechbrain)

        return render_template(
            "graph.html",
            labels_inaspeech=total_precision_recall[0][0], values_inaspeech=total_precision_recall[0][1],
            labels_picovoice=total_precision_recall[1][0], values_picovoice=total_precision_recall[1][1],
            labels_speechbrain_3=total_precision_recall[2][
                0], values_speechbrain_3=total_precision_recall[2][1],
            labels_speechbrain_10=total_precision_recall[3][
                0], values_speechbrain_10=total_precision_recall[3][1],
            labels_speechbrain_100=total_precision_recall[4][
                0], values_speechbrain_100=total_precision_recall[4][1],
            labels_webrtc=total_precision_recall[5][0], values_webrtc=total_precision_recall[5][1],
        )

    return render_template("evaluation.html")


@app.route("/detection", methods=["GET", "POST"])
def detection():
    if request.method == "POST":
        global vad_type

        selection = request.form.get('comp_select')
        if selection is not None:
            vad_type = str(selection)

        # make the directory where the files should save
        global parent_dir
        parent = parent_dir + "/archives/"

        directory = vad_type
        path = os.path.join(parent, directory)

        isExist = os.path.exists(path)
        if isExist == False and len(parent) == 0:
            # Create the directory
            os.mkdir(path)

        if len(parent) != 0:
            # Iterate directory
            for patH in os.listdir(parent):
                # check if current path is a file
                if os.path.isfile(os.path.join(parent, patH)):
                    vad_type = patH
                    path = path + vad_type

        app.config["UPLOAD_PATH"] = path

        # go through all the files
        for folder in request.files.getlist("folder_path"):
            folder.save(os.path.join(
                app.config["UPLOAD_PATH"], folder.filename))

        # get the path to the trained_models for speechbrain
        trained_models_path_speechbrain = parent_dir + \
            "/vad/trained_models/"

        detection_path = main_course_detection(
            path, vad_type, trained_models_path_speechbrain)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        fileName = vad_type + "_{}.zip".format(timestr)
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(detection_path):
                for file in files:
                    zipf.write(os.path.join(root, file))

        memory_file.seek(0)
        return send_file(memory_file,
                         attachment_filename=fileName,
                         as_attachment=True)

    return render_template("detection.html",
                           data=[{'name': 'InaSpeechSegmenter'}, {'name': 'Picovoice'}, {'name': 'Speechbrain'}, {'name': 'WebRTC'}])


# @app.route("/inaspeechsegmenter", methods=["GET", "POST"])
# def page_inaspeechsegmenter():
#     return render_template("inaspeechsegmenter.html")


# @app.route("/picovoice", methods=["GET", "POST"])
# def page_picovoice():
#     return render_template("picovoice.html")


# @app.route("/speechbrain", methods=["GET", "POST"])
# def page_speechbrain():
#     return render_template("speechbrain.html")


# @app.route("/webrtc", methods=["GET", "POST"])
# def page_webrtc():
#     return render_template("webrtc.html")

if __name__ == "__main__":
    app.run(port=6454, debug=True)
