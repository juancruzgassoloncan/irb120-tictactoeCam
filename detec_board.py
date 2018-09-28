#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Copyright (C) 2018 FI-UNER Robotic Group
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
@mail: robotica@ingenieria.uner.edu.ar
"""
import cv2
import numpy as np
import helper as H
import dill as pickle

# import socket
# import select
# import time




class calibration:

    def __init__(self, camera):
        self.win_names = ('Raw', 'Canny', 'Dilation', 'Calibration')
        self.win_size = (640, 480)
        self.camera = camera
        self.cam_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.maxT = 100
        self.minT = 30
        self.k_size = 3
        self.d_iter = 3
        self.e_iter = 2
        self.roi = None
        self.cells = None
        class calibData:
            roi = None
            cells = None
            minCannyTh = None
            maxCannyTh = None
            kernelSize = None
            dilationIter = None
            erodeIter = None
        self.calibData = calibData()

    def transl(self, x, y):
        self.T1 = np.float32([[1, 0, x],
                               [0, 1, y]])

    def create_windows(self):
        win_size = self.win_size
        for name in self.win_names:
            cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)
            cv2.resizeWindow(name, win_size[0], win_size[1])

    def checkCamera(self):
        if not self.camera.isOpened():
            print("Error opening video stream or file")
            self.camera.release()
            cv2.destroyAllWindows()
            return False
        else:
            return True

    def show(self):
        while True:
            _, frame = self.camera.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def get_roi(self):
        cv2.createTrackbar('max_canny', self.win_names[1], self.maxT, 255, lambda *args: None)
        cv2.createTrackbar('min_canny', self.win_names[1], self.minT, 255, lambda *args: None)
        cv2.createTrackbar('kernel_size', self.win_names[2], 2, 5, lambda *args: None)
        cv2.createTrackbar('dilate_iter', self.win_names[2], 3, 5, lambda *args: None)
        cv2.createTrackbar('erode_iter', self.win_names[2], 2, 5, lambda *args: None)

        cv2.createTrackbar('transl_x', self.win_names[3], 0, 5, lambda *args: None)
        cv2.createTrackbar('transl_y', self.win_names[3], 0, 5, lambda *args: None)
        cv2.setTrackbarMin('transl_x', self.win_names[3], -5)
        cv2.setTrackbarMin('transl_y', self.win_names[3], -5)

        while True:
            self.maxT = cv2.getTrackbarPos('max_canny', self.win_names[1])
            self.minT = cv2.getTrackbarPos('min_canny', self.win_names[1])
            self.k_size = cv2.getTrackbarPos('kernel_size', self.win_names[2])
            self.d_iter = cv2.getTrackbarPos('dilate_iter', self.win_names[2])
            self.e_iter = cv2.getTrackbarPos('erode_iter', self.win_names[2])
            x = cv2.getTrackbarPos('transl_x', self.win_names[3])
            y = cv2.getTrackbarPos('transl_y', self.win_names[3])
            self.transl(x, y)

            ret, frame = self.camera.read()

            if not ret:
                self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            raw_frame = frame.copy()
            src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            src_gray = cv2.blur(src_gray, (3, 3))
            canny_output = cv2.Canny(src_gray, self.minT, self.maxT)

            if self.k_size == 0:
                wrapped = canny_output
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.k_size, self.k_size))
                dilation = cv2.dilate(canny_output, kernel, iterations=self.d_iter)
                closed = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
                erode = cv2.erode(closed, kernel, iterations=self.e_iter)
                wrapped = cv2.warpAffine(erode, self.T1, (self.cam_width, self.cam_height))
            board = H.findBiggestContour(wrapped)

            if board is False:
                cv2.imshow(self.win_names[1], wrapped)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                continue

            cv2.drawContours(frame, board, -1, (0, 0, 255), 1)
            rect = cv2.boundingRect(board[0])
            x, y, w, h = H.wrap_digit(rect, 5, False)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.roi = np.array((y, y + h, x, x + w))
            cv2.imshow(self.win_names[0], raw_frame)
            cv2.imshow(self.win_names[1], canny_output)
            cv2.imshow(self.win_names[2], wrapped)
            cv2.imshow(self.win_names[3], frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # self.camera.release()
        cv2.destroyAllWindows()
        return self.roi

    def set_roi(self, new_roi=()):
        if len(new_roi) == 0:
            self.roi = np.floor([self.cam_height / 5, self.cam_height * 4 / 5,
                                 self.cam_width / 5, self.cam_width * 4 / 5]).astype(int)
        elif len(new_roi) == 4:
            self.roi = new_roi
        else:
            raise Exception("roi need to be length 4")

    def apply_roi(self, img):
        copy = img.copy()
        copy = copy[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        return copy

    def view_by_roi(self):

        while True:
            _, frame = self.camera.read()
            frame = self.apply_roi(frame)
            cv2.imshow('view roi', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def get_cells(self):
        source_window = 'cells'
        while True:
            ret, frame = self.camera.read()

            if not ret:
                continue

            roi_frame = self.apply_roi(frame)
            src_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            src_gray = cv2.blur(src_gray, (3, 3))
            canny_output = cv2.Canny(src_gray, self.minT, self.maxT)

            if self.k_size == 0:
                wrapped = canny_output
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.k_size, self.k_size))
                dilation = cv2.dilate(canny_output, kernel, iterations=self.d_iter)
                closed = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
                erode = cv2.erode(closed, kernel, iterations=self.e_iter)
                wrapped = cv2.warpAffine(erode, self.T1, (self.cam_width, self.cam_height))

            cells = H.findCells(wrapped)
            if cells is False:
                cv2.imshow(source_window, roi_frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                continue

            cells, bbox = H.sort_cells(cells)
            # Display the resulting frame
            cnt = []
            for i in range(len(cells)):
                cv2.drawContours(roi_frame, cells, i, (100, 255, 100), 1)
                cv2.putText(roi_frame, str(i), bbox[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 50), 2, cv2.LINE_AA)
            #            cv2.waitKey()

            cv2.imshow(source_window, roi_frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        # self.camera.release()
        cv2.destroyAllWindows()
        self.cells = np.array(cells)
        # cells = np.transpose(cells.reshape((3, 3))).flatten()
        return self.cells

    def run(self):
        self.create_windows()
        self.checkCamera()
        self.get_roi()
        self.get_cells()
        self.calibData.roi = self.roi
        self.calibData.cells = self.cells
        self.calibData.minCannyTh = self.minT
        self.calibData.maxCannyTh = self.maxT
        self.calibData.kernelSize = self.k_size
        self.calibData.dilationIter = self.d_iter
        self.calibData.erodeIter = self.e_iter

        return self.calibData


class brain(calibration):

    def __init__(self, cam, calibData):
        self.camera = cam
        self.win_size = (800,600)
        self.win_names = ['detector']
        self.create_windows()
        self.calibData = calibData
        self.set_roi(calibData.roi)

    def prosses(self, frame):
        # src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # src_gray = cv2.blur(src_gray, (3, 3))
        canny_output = cv2.Canny(frame, self.calibData.minCannyTh, self.calibData.maxCannyTh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.calibData.kernelSize,
                                                            self.calibData.kernelSize))
        dilation = cv2.dilate(canny_output, kernel, iterations=self.calibData.dilationIter)
        closed = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        erode = cv2.erode(closed, kernel, iterations=self.calibData.erodeIter)
        return erode

    def detect(self, img):
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        src_gray = cv2.blur(src_gray, (3, 3))
        frame = self.prosses(src_gray)
        cnt = H.findContour(frame)
        circles = cv2.HoughCircles(src_gray, cv2.HOUGH_GRADIENT, 1.5, 50),#param1=100,param2=100,minRadius=10,maxRadius=50)
        print (len(circles.shape))
        if len(circles.shape) == 3:
            a, b, c = circles.shape
            for i in range(b):
                cv2.circle(img, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, cv2.LINE_AA)
            return frame, 2
        patternX = cv2.cvtColor(cv2.imread('./img/X.png'), cv2.COLOR_BGR2GRAY)
        patternX = cv2.resize(patternX, (90, 90))
        patternX = H.findBiggestContour2(patternX)
        threshold = 1.5
        for c in cnt:
            res = cv2.matchShapes(patternX, c, 1, 0.0)
            if res <= threshold:
                rect = cv2.boundingRect(c)
                x, y, w, h = H.wrap_digit(rect, 1, False)
                cv2.rectangle(img, (x, y), (x + w, y + h), (50, 180, 200), 2)
                return img, 1
        return img, 0

    def cut_cells(self):
        cc = []
        for i in self.calibData.cells:
            cell = cv2.boundingRect(i)
            x, y, w, h = H.wrap_digit(cell, -4, False)
            cc.append((y, y + h, x, x + w))
        while True:
            ret, frame = self.camera.read()

            if not ret:
                self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # frame = cv2.warpAffine(frame, self.T1, (self.cam_width, self.cam_height))
            roi_frame = self.apply_roi(frame)

            a1 = self.detect(roi_frame[cc[0][0]:cc[0][1], cc[0][2]:cc[0][3]])
            a2 = self.detect(roi_frame[cc[1][0]:cc[1][1], cc[1][2]:cc[1][3]])
            a3 = self.detect(roi_frame[cc[2][0]:cc[2][1], cc[2][2]:cc[2][3]])

            b1 = self.detect(roi_frame[cc[3][0]:cc[3][1], cc[3][2]:cc[3][3]])
            b2 = self.detect(roi_frame[cc[4][0]:cc[4][1], cc[4][2]:cc[4][3]])
            b3 = self.detect(roi_frame[cc[5][0]:cc[5][1], cc[5][2]:cc[5][3]])

            c1 = self.detect(roi_frame[cc[6][0]:cc[6][1], cc[6][2]:cc[6][3]])
            c2 = self.detect(roi_frame[cc[7][0]:cc[7][1], cc[7][2]:cc[7][3]])
            c3 = self.detect(roi_frame[cc[8][0]:cc[8][1], cc[8][2]:cc[8][3]])

            row1 = np.array([a1, a2, a3])
            row2 = np.array([b1, b2, b3])
            row3 = np.array([c1, c2, c3])

            detected = np.concatenate((row1[:, 1], row2[:, 1], row3[:, 1]), axis=0)
            detected = np.reshape(detected, (3, 3))

            row1 = [cv2.resize(x[0], (90, 90)) for x in row1]
            row1 = np.concatenate(row1, axis=1)

            row2 = [cv2.resize(x[0], (90, 90)) for x in row2]
            row2 = np.concatenate(row2, axis=1)

            row3 = [cv2.resize(x[0], (90, 90)) for x in row3]
            row3 = np.concatenate(row3, axis=1)

            complete = np.concatenate((row1, row2, row3), axis=0)
            cv2.imshow(self.win_names[0], complete)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break



#    res = cv2.matchTemplate(dilation,template,cv2.TM_CCOEFF_NORMED)

#
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        return pickle.load(input)
#    while True:
#        cv2.imshow('dilate', dilation)
#        if cv2.waitKey(5) & 0xFF == ord('q'):
#            break


# Break the loop
# When everything done, release the video capture object
if __name__ == '__main__':
    camera = H.open_camera_device('./img/video_sample.mp4')
    # camera = H.open_camera_device(1)

    calibrator = calibration(camera)
    calibData = calibrator.run()
    # save_object(calibData,'calibration.bin')
    # calibData = load_object('calibration.bin')
    detector = brain(camera, calibData)
    detector.cut_cells()
#    HOST = ''                 # Symbolic name meaning all available interfaces
#    PORT = 50007              # Arbitrary non-privileged port
##    socket.setdefaulttimeout(0.5)
#    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#    s.bind((HOST, PORT))
#    s.listen(1)
#    conn, addr = s.accept()
#    conn.settimeout(0.5)
#    print( 'Connected by', addr)
#    cutCells(pitch, Cells, conn, cam)
#%%
#    cv2.destroyAllWindows()
