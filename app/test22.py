import torch
import numpy as np
import cv2
import time
from pyzbar.pyzbar import decode
import sqlite3

# создание бд
db = sqlite3.connect('server.db')
sql = db.cursor()

sql.execute('''CREATE TABLE IF NOT EXISTS data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state TEXT,
    time TEXT,
    frames int,
    xmin TEXT,
    ymin TEXT,
    xmax TEXT,
    ymax TEXT,
    name TEXT
    )''')

db.commit()

class MugDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, capture_index, model_name):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """

        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name,
                                   force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(
                    row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        done_barcodes = []
        is_bar_end = []
        # states
        waiting_stream = 'waiting_stream'
        stream_on = 'stream_on'
        waiting_barcode = 'waiting_barcode'
        decoded = 'decoded'
        state = waiting_stream
        colvo_frames = 0
        # дописать sql request и добавить в них время

        # Get video
        cap = self.get_video_capture()

        # for infinity decode frames
        while True:
            colvo_frames += 1
            # удалить
            print(state)
            # Check stream yes or no

            # Stream on
            if state == waiting_stream and cap.isOpened():
                state = stream_on
                sql.execute(f''' INSERT INTO data (state, time, frames, xmin, ymin, xmax, 
                ymax, name)
                                VALUES(?,?,?,?,?,?,?,?)''',
                            (state, time.time(), None, None, None, None, None, None))
                db.commit()
                ## sql request

            # Stream off
            if state == waiting_stream and cap.isOpened() == False:
                print('No camera 1 ')
                break
                ## sql request

            # should be state == stream_on,
            # Info about stream
            # stream on
            if state == stream_on:
                state = waiting_barcode
                sql.execute(f''' INSERT INTO data (state, time, frames, xmin, ymin, xmax, 
                                ymax, name)
                                                VALUES(?,?,?,?,?,?,?,?)''',
                            (state, time.time(), None, None, None, None, None, None))
                db.commit()
                ## sql request

            # stream was on but turn off
            if state == stream_on and cap.isOpened() == False:
                state = waiting_stream
                sql.execute(f''' INSERT INTO data (state, time, frames, xmin, ymin, xmax, 
                                ymax, name)
                                                VALUES(?,?,?,?,?,?,?,?)''',
                            (state, time.time(), None, None, None, None, None, None))
                db.commit()
                ## sql request

            if state != stream_on and cap.isOpened() == False:
                print('No camera')
                state = waiting_stream
                ## sql request

            # should be state == waiting_barcode, cap.isOpened() == True

            # Get frame
            ret, frame = cap.read()
            # if video end
            try:
                assert ret
            except AssertionError:
                print('No more frames to decode')
                break

            # #this statement start work after detected barcode
            if state == decoded:
                y = decode(frame)
                # Get barcode
                if y is None and 'N' not in is_bar_end:
                    is_bar_end.append(y)
                if y is not None and 'N' in is_bar_end:
                    done_barcodes.append(y)
                    is_bar_end.clear()
                    x = None
                    state = waiting_barcode
                    # sql request
                # d_b = [1,2]
                # Check barcode in frame
            if state == waiting_barcode:
                x = decode(frame)
                # Get barcode
                if x is not None:
                    if x in done_barcodes:
                        pass
                    else:
                        is_bar_end.append(x)
                        state = decoded

            # should be state == decode, cap.isOpened() == True, x != None

            list_of_records = ''
            time_rec = ''
            if state == decoded:
                # output frame
                frame = cv2.resize(frame, (1280, 720))
                results = self.score_frame(frame)
                frame = self.plot_boxes(results, frame)
                cv2.imshow('YOLOv5 Detection', frame)

                # prepare data for .txt
                im = frame
                results1 = self.model(im)
                record = results1.pandas().xyxy
                time_rec = time.time()
                list_of_records = record

                # load data in .txt
                file_of_records = open('records.txt', 'w')  # поменять на w
                file_of_records.write(str(time_rec))
                file_of_records.write('\n')
                file_of_records.write(str(list_of_records))
                file_of_records.write('\n')
                file_of_records.close()

                # чистим выходной файл убираю путсые строки
                infile = "records.txt"
                outfile = "cleaned_records.txt"
                with open(infile) as fin, open(outfile, "w+") as fout:
                    for line in fin:
                        x = line[0]
                        if x == '[' or x == 'C' or x == 'I':
                            line = line.replace(line, '')
                        fout.write(line)

                # составляем массив из cleaned_records
                file1 = open("cleaned_records.txt", "r")
                s = []
                x = []

                while True:
                    # считываем строку
                    line = file1.readline()
                    # прерываем цикл, если строка пустая
                    if not line:
                        break
                    # выводим строку
                    s = line.split()
                    if len(s) == 8:
                        s[7] = 'Rose'
                        x.append(s)
                    if len(s) == 1:
                        x.append(s)

                # убираем лишнее время и делаем строки массивами
                a = []
                d = ''
                for i in range(len(x) - 1):
                    if len(x[i]) == len(x[i + 1]):
                        pass
                    else:
                        a.append(x[i])
                    if len(x[i]) == len(x[i + 1]) and len(x[i]) == 8:
                        a.append(x[i])

                # записываем массивы в файл
                file_of_records = open('Very_clear_records.txt', 'w')
                for i in a:
                    file_of_records.write(str(i))
                    file_of_records.write('\n')
                file_of_records.close()
                # закрываем файл
                file1.close()

                #
                # объединение времен с координатами
                appended_records = []
                for i in range(len(a)):
                    if len(a[i]) == 1:
                        tme = a[i]
                        i += 1
                        try:
                            while len(a[i]) != 1:
                                ap_r = tme + a[i]
                                appended_records.append(ap_r)
                                i += 1
                        except IndexError:
                            pass


                # sql request
                for i in range(len(appended_records)):
                    time_ = appended_records[i][0]
                    xmin = appended_records[i][2]
                    ymin = appended_records[i][3]
                    xmax = appended_records[i][4]
                    ymax = appended_records[i][5]
                    name = appended_records[i][8]
                    sql.execute(f''' INSERT INTO data (state, time, frames, xmin, ymin, 
                    xmax, ymax, name)
                                                    VALUES(?,?,?,?,?,?,?,?)''',
                                (state, time_, colvo_frames, xmin, ymin, xmax, ymax, name))
                db.commit()

                # Esc is closed window
                if cv2.waitKey(5) & 0xFF == 27:
                    cv2.destroyAllWindows()
        cap.release()


# Create a new object and execute.
detector = MugDetection(capture_index='vid.mp4', model_name='best.pt')
detector()
