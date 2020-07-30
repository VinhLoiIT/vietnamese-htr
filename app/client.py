import sys
from PyQt5.QtCore import Qt, QPoint, QRect, QByteArray, QBuffer, QIODevice
from PyQt5.QtWidgets import QMainWindow, QApplication, QRubberBand, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QPen, QPalette, QBrush, QIcon, QImage

from typing import List
import requests
import argparse
import copy
import base64
import json
from image_viewer import ImageViewer
from PIL import Image
from io import BytesIO


class Main(QMainWindow):

    def __init__(self, port: int):
        super().__init__()
        self.initUI()
        self.initToolbar()
        self.statusBar().showMessage('Ready')

        self.API_URL = f'http://localhost:{port}/api/recognize'

        self.show()

    def initUI(self):
        self.image_viewer = ImageViewer()

        self.setCentralWidget(self.image_viewer)
        self.resize(800, 600)

    def initToolbar(self):
        toolbar = self.addToolBar('Toolbar')
        toolbar.setMovable(False)

        load_image_button = QPushButton(text='LoadImage')
        load_image_button.clicked.connect(self.onLoadImageClick)
        toolbar.addWidget(load_image_button)
        
        segment_button = QPushButton(text='Word Segment')
        segment_button.clicked.connect(self.image_viewer.autoSegmentation)
        toolbar.addWidget(segment_button)

        clear_bbox_button = QPushButton(text='Clear')
        clear_bbox_button.clicked.connect(self.image_viewer.clearBoxes)
        toolbar.addWidget(clear_bbox_button)
        
        predict_button = QPushButton(text='Predict')
        predict_button.clicked.connect(self.onPredict)
        toolbar.addWidget(predict_button)

    def onLoadImageClick(self):
        file_path, file_ext = QFileDialog.getOpenFileName(self,
                                                          'Select image file',
                                                          '.',
                                                          'PNG(*.png);; JPEG (*.jpg *.jpeg);; TIFF (*.tif);; All files (*.*)',
                                                          'JPEG (*.jpg *.jpeg)')
        if file_path:
            self.loadImage(file_path)

    def onRotateLeftClick(self):
        print('Rotate L')
        
    def onRotateRightClick(self):
        print('Rotate R')

    def onPredict(self):
        print('Predict')

        bboxes = self.image_viewer.getBoxes()

        if len(bboxes) == 0:
            print('Empty boxes')
            return

        request_predicts = {
            'images': []
        }

        # box: QRubberBand
        for i, box in enumerate(bboxes):
            cropped: QPixmap = self.image_viewer.pixmap().copy(box)
            image: Image.Image = Image.fromqpixmap(cropped)
            # image.save('images/{}.jpg'.format(i))

            buffer = BytesIO()
            image.save(buffer, 'jpeg')
            buffer.seek(0)

            request_predicts['images'].append({
                'id': i,
                'data': base64.b64encode(buffer.getvalue()).decode(),
            })
        
        result = requests.post(self.API_URL, json=request_predicts)
        if result.ok:
            print(result.json())
        else:
            print(result.reason)


    def loadImage(self, image_path):
        ok = self.image_viewer.loadImage(image_path)
        if ok:
            image: QPixmap = self.image_viewer.pixmap()
            message = f'{image_path} | {image.width()} x {image.height()}'
            self.statusBar().showMessage(message)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=5050)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    mainMenu = Main(args.port)
    sys.exit(app.exec_())