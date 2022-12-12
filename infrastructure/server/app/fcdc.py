#!usr/bin/python
# -*- coding: latin-1 -*-

import torch
import numpy as np
import cv2
import time
from pyzbar.pyzbar import decode
import mysql.connector as mariadb
# connect db
mariadb_connection = mariadb.connect(user='fcdc', password='fcdc', host='db',
                                     port='3306', database='fcdc')
cursor = mariadb_connection.cursor()

# create table
cursor.execute('''CREATE TABLE IF NOT EXISTS meta_data (
    id int AUTO_INCREMENT PRIMARY KEY,
    state1 varchar(255),
    `time` varchar(255),
    frames int,
    xmin varchar(255),
    ymin varchar(255),
    xmax varchar(255),
    ymax varchar(255),
    name varchar(255),
    barcode varchar(255)
    )''')

mariadb_connection.commit()

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
# read video
    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """

        return cv2.VideoCapture(self.capture_index)
# load prepared model
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
# convert frame to array
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
# class to label
    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]
# draw boxes
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
#main func
    def __call__(self):
        done_barcodes = []
        is_bar_end = []
        # states
        waiting_stream = 'waiting_stream'
        stream_on = 'stream_on'
        waiting_barcode = 'waiting_barcode'
        decoded = 'decoded'
        state = waiting_stream
        amount_of_frames = 0

        # Get video
        cap = self.get_video_capture()

        # for infinity decode frames
        while True:
            amount_of_frames += 1

            # print(state)


            # Stream on
            if state == waiting_stream and cap.isOpened():
                state = stream_on
            # sql request
                sql_st = ''' INSERT INTO meta_data (state1, time) VALUES(%s, %s)'''
                val = (state, time.time())
                cursor.execute(sql_st, val)
                mariadb_connection.commit()


            #if Stream off
            if state == waiting_stream and cap.isOpened() == False:
                print('No camera 1 ')
                break
                # mb need sql request

            # should be state == stream_on,

            # Info about stream

            #if stream on
            if state == stream_on:
                state = waiting_barcode
                sql_st = ''' INSERT INTO meta_data (state1, time) VALUES(%s, %s)'''
                val = (state, time.time())
                cursor.execute(sql_st, val)
                mariadb_connection.commit()
                #mb need sql request

            # stream was on but turn off
            if state == stream_on and cap.isOpened() == False:
                state = waiting_stream
                sql_st = ''' INSERT INTO meta_data (state1, time) VALUES(%s, %s)'''
                val = (state, time.time())
                cursor.execute(sql_st, val)
                mariadb_connection.commit()
                # mb need sql request

            # if stream was on but video break
            if state != stream_on and cap.isOpened() == False:
                print('No camera')
                state = waiting_stream
            #sql request
                sql_st = ''' INSERT INTO meta_data (state1, time) VALUES(%s, %s)'''
                val = (state, time.time())
                cursor.execute(sql_st, val)
                mariadb_connection.commit()


            # should be state == waiting_barcode, cap.isOpened() == True

            # read frames from video
            # if ret == False no more frames 
            ret, frame = cap.read()
            try:
                assert ret
            except AssertionError:
                print('No more frames to decode')
                break

            # Check barcode in frame
            if state == waiting_barcode:
                x = decode(frame)
            # Get barcode
                if x is not None:
                    x = int(x[0].data.decode('utf-8'))
                    
            # if order already done
                    if x in done_barcodes:
                        pass
            # if not
                    else:
                        is_bar_end.append(x)
                        state = decoded
                        sql_st = ''' INSERT INTO meta_data (state1, time, barcode) VALUES(%s,%s,%s)'''
                        val = ('start',time.time(),x)
                        cursor.execute(sql_st,val)
                        mariadb_connection.commit()
        
            #read stop barcode
            if state == decoded:
                 y = decode(frame)
            #if y==[] start barcode end
                 if y==[] and len(is_bar_end)!=2:
                     is_bar_end.append('N')
            #if y!=[] catch stop barcode
                 if y!=[] and len(is_bar_end)==2:
                     print(int(y[0].data.decode('utf-8')))
                     done_barcodes.append(int(y[0].data.decode('utf-8')))
                     state = waiting_barcode
                     is_bar_end.clear()
                     sql_st = ''' INSERT INTO meta_data (state1, time, barcode) VALUES(%s,%s,%s)'''
                     val = ('stop',time.time(),int(y[0].data.decode('utf-8')))
                     cursor.execute(sql_st,val)
                     mariadb_connection.commit()
                     x = None

            # should be state == decode, cap.isOpened() == True, x != None

            list_of_records = ''
            time_rec = ''
            if state == decoded:
            # output frame useless in work
            # frame = cv2.resize(frame, (1280, 720))
            # results = self.score_frame(frame)
            # frame = self.plot_boxes(results, frame)
            # cv2.imshow('YOLOv5 Detection', frame)

            # prepare data for .txt
                im = frame
                results1 = self.model(im)
                record = results1.pandas().xyxy
                time_rec = time.time()
                list_of_records = record

            # load data in .txt
                file_of_records = open('records.txt', 'w')  
                file_of_records.write(str(time_rec))
                file_of_records.write('\n')
                file_of_records.write(str(list_of_records))
                file_of_records.write('\n')
                file_of_records.close()

                # clear records.txt del empty array
                infile = "records.txt"
                outfile = "cleaned_records.txt"
                with open(infile) as fin, open(outfile, "w+") as fout:
                    for line in fin:
                        x = line[0]
                        if x == '[' or x == 'C' or x == 'I':
                            line = line.replace(line, '')
                        fout.write(line)

                # make array from clenead records
                file1 = open("cleaned_records.txt", "r")
                s = []
                x = []

                while True:
                    #read lines
                    line = file1.readline()
                    # if line empty skip 
                    if not line:
                        break
                    # output line 
                    s = line.split()
                    if len(s) == 8:
                        s[7] = 'Rose'
                        x.append(s)
                    if len(s) == 1:
                        x.append(s)

                # del  extra time lines
                a = []
                d = ''
                for i in range(len(x) - 1):
                    if len(x[i]) == len(x[i + 1]):
                        pass
                    else:
                        a.append(x[i])
                    if len(x[i]) == len(x[i + 1]) and len(x[i]) == 8:
                        a.append(x[i])

                # write arrays to txt
                file_of_records = open('Very_clear_records.txt', 'w')
                for i in a:
                    file_of_records.write(str(i))
                    file_of_records.write('\n')
                file_of_records.close()
                file1.close()

                
                # unite time and coords
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

                #prepare data to sql request  
                for i in range(len(appended_records)):
                    secs = appended_records[i][0]
                    xmin = appended_records[i][2]
                    ymin = appended_records[i][3]
                    xmax = appended_records[i][4]
                    ymax = appended_records[i][5]
                    name = appended_records[i][8]
                    # sql request
                    sql_st = f''' INSERT INTO meta_data (state1, time, frames, xmin, ymin, xmax, 
                                                ymax, name) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)'''
                    val = (state,secs,amount_of_frames,xmin,ymin,xmax,ymax,name)
                    cursor.execute(sql_st,val)
                mariadb_connection.commit()

                # Esc is closed window
                if cv2.waitKey(5) & 0xFF == 27:
                    cv2.destroyAllWindows()
        cap.release()


# Create a new object and execute.
detector = MugDetection(capture_index='/mnt/input/video2.mp4', model_name='/mnt/input/model.pt')
detector()
