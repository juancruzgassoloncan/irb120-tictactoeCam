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


# Jetson onboard camera
def open_jetson_camera():
    camera = cv2.VideoCapture(
        "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    return camera


# Open an external usb camera /dev/videoX
def open_camera_device(device_number):
    camera = cv2.VideoCapture(device_number)
    return camera


def wrap_digit(rect, padding=3, center=True):
    x, y, w, h = rect
    #    padding = 5q
    if center:
        hcenter = int(x + w / 2)
        vcenter = int(y + h / 2)
        if (h > w):
            w = h
            x = hcenter - int(w / 2)
        else:
            h = w
            y = vcenter - int(h / 2)
    return (x - padding, y - padding, w + 2 * padding, h + 2 * padding)


def findBiggestContour2(mask):
    temp_bigger = []
    img1, cont, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cont) == 0:
        return False
    for cnt in cont:
        temp_bigger.append(cv2.contourArea(cnt))
    greatest = max(temp_bigger)
    index_big = temp_bigger.index(greatest)
    key = 0
    for cnt in cont:
        if key == index_big:
            return cnt
            break
        key += 1


def findBiggestContour(mask):
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    board = []
    if hierarchy is None: return False
    for cnt, hie in zip(contours, hierarchy[0, :, :]):
        if hie[3] == -1 and hie[2] != -1:
            board.append(cnt)
    if not board:
        return False
    else:
        return board


def findContour(mask):
    img1, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def findCells(mask):
    ext, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for cnt, hie in zip(contours, hierarchy[0, :, :]):
        if hie[3] == 0:
            cells.append(cnt)
    if not cells:
        return False
    else:
        return cells


def sort_cells(cnts):
    centroids = []
    for c in cnts:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroids.append((cX, cY))

    [cnts, centroids] = zip(*sorted(zip(cnts, centroids), key=lambda b: b[1][1], reverse=True))
    reverse = False
    row1 = sorted(zip(cnts[:3], centroids[:3]), key=lambda b: b[1][0], reverse=reverse)
    row2 = sorted(zip(cnts[3:6], centroids[3:6]), key=lambda b: b[1][0], reverse=reverse)
    row3 = sorted(zip(cnts[6:], centroids[6:]), key=lambda b: b[1][0], reverse=reverse)
    cc = (*[a for a, _ in row1], *[a for a, _ in row2], *[a for a, _ in row3])
    bb = (*[a for _, a in row1], *[a for _, a in row2], *[a for _, a in row3])
    return cc, bb
