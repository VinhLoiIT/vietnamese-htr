import sys
from PyQt5.QtCore import Qt, QPoint, QRect, QByteArray, QBuffer, QIODevice, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QRubberBand, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea
from PyQt5.QtGui import QPixmap, QPainter, QPen, QPalette, QBrush, QIcon, QImage

from typing import List, Optional
import numpy as np
import requests
import copy
import base64
import json
from PIL import Image
from io import BytesIO

from scipy.cluster.vq import kmeans2
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage import median_filter

import cv2

class ImageViewer(QWidget):
    
    def __init__(self, image_path: Optional[str] = None):
        super().__init__()
        self.lastPoint = QPoint()

        self.current_rubberBand = None
        self.bboxes: List[QRubberBand] = []
        
        self.initUI()

        if image_path:
            self.loadImage(image_path)

    def initUI(self):
        layout = QVBoxLayout()

        self.label_image = QLabel()
        self.label_image.setBackgroundRole(QPalette.Dark)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.label_image)
        layout.addWidget(self.scrollArea)

        self.setLayout(layout)

    def loadImage(self, image_path):
        image = QPixmap(image_path)
        if image.hasAlphaChannel():
            image = QPixmap.fromImage(image.toImage().convertToFormat(QImage.Format_RGB32))
        self.label_image.setPixmap(image)
        self.label_image.resize(image.width(), image.height())
        return True

    def mousePressEvent(self, event):
        new_pos = self._convertMousePosToLocal(event.pos())

        self.lastPoint = new_pos
        if not self.current_rubberBand:
            self.current_rubberBand = self._create_rubber_band()
        self.current_rubberBand.setGeometry(QRect(self.lastPoint, self.lastPoint))
        self.current_rubberBand.show()

    def _create_rubber_band(self):
        rubber_band = QRubberBand(QRubberBand.Rectangle, self.label_image)
        pal = QPalette()
        pal.setBrush(QPalette.Highlight, QBrush(Qt.red))
        rubber_band.setPalette(pal)
        return rubber_band

    def mouseMoveEvent(self, event):
        new_pos = self._convertMousePosToLocal(event.pos())

        last_pos = self.lastPoint

        if new_pos.x() >= last_pos.x():
            if new_pos.y() >= last_pos.y():
                self.current_rubberBand.setGeometry(QRect(last_pos, new_pos))
            else:
                self.current_rubberBand.setGeometry(last_pos.x(), new_pos.y(), new_pos.x() - last_pos.x(), last_pos.y() - new_pos.y())
        else:
            if new_pos.y() >= last_pos.y():
                self.current_rubberBand.setGeometry(new_pos.x(), last_pos.y(), last_pos.x() - new_pos.x(), new_pos.y() - last_pos.y())
            else:
                self.current_rubberBand.setGeometry(QRect(new_pos, last_pos))

    def mouseReleaseEvent(self, event):
        self.bboxes.append(self.current_rubberBand)
        self.current_rubberBand = None

    def _convertMousePosToLocal(self, mousePos: QPoint):
        new_pos = self.label_image.mapFromGlobal(self.mapToGlobal(mousePos))
        new_pos.setX(max(0, new_pos.x()))
        new_pos.setX(min(new_pos.x(), self.label_image.size().width()))

        new_pos.setY(max(0, new_pos.y()))
        new_pos.setY(min(new_pos.y(), self.label_image.size().height()))
        return new_pos

    def getBoxes(self) -> List[QRect]:
        box: QRubberBand
        boxes = [box.geometry() for box in self.bboxes]
        return boxes

    def pixmap(self) -> QPixmap:
        return self.label_image.pixmap()

    @pyqtSlot()
    def autoSegmentation(self):
        print('Segmentation')
        pil_image = Image.fromqpixmap(self.label_image.pixmap())
        qrects = segmentation(pil_image)

        self.clearBoxes()
        for qrect in qrects:
            rubber_band = self._create_rubber_band()
            rubber_band.setGeometry(qrect)
            rubber_band.show()
            self.bboxes.append(rubber_band)


    @pyqtSlot()
    def clearBoxes(self):
        print('Clear')
        for box in self.bboxes:
            box.hide()
        self.bboxes.clear()

    @pyqtSlot()
    def rotateImage(self, degree):
        print('Rotate {}'.format(degree))


def binarize(pil_image: Image.Image):
    data = np.array(pil_image, dtype=np.float)
    H, W, C = data.shape
    data = data.reshape(H*W, 3)
    centroid, label = kmeans2(data, k=np.array([[0., 0., 0.],               # black
                                                [255., 255., 255.]]))       # white
    label = label * 255.
    label = label.reshape(H, W)
    return Image.fromarray(label)


def segmentation(pil_image: Image.Image):
    '''
    Return list of bounding box (x, y, w, h)
    '''
    pil_image = binarize(pil_image)

    data: np.ndarray = np.array(pil_image, dtype=np.uint8)
    counts = np.count_nonzero(data == 0, axis=0)

    # norm
    counts = counts / data.shape[0]

    # smooth
    counts = gaussian_filter1d(counts, 2)
    # plt.plot(counts)

    break_points = np.abs(counts) < 0.02
    # plt.plot(break_points)

    # plt.savefig('fig.jpg')

    data[:, break_points] = np.array([0])

    contours, _ = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = [cv2.boundingRect(contour) for contour in contours] # x, y, w, h
    qrects = [QRect(rect[0], rect[1], rect[2], rect[3]) for rect in rects]

    results = []
    for i in range(0, len(qrects)):
        flag = True
        for j in range(0, len(qrects)):
            r1, r2 = qrects[i], qrects[j]
            if i != j and r2.contains(r1):
                flag = False
                break
        if flag:
            results.append(r1)

    # sort by x
    results = sorted(results, key=lambda x: x.x())
    
    return results