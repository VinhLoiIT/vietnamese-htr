import argparse
import base64
import json
from io import BytesIO

import torch
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
from PIL import Image

from model import ModelTF


class CERecognize(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('images', type=list, location='json')
        args = parse.parse_args()

        images = []
        ids = []
        for image in args['images']:
            image_id = image['id']
            ids.append(image_id)

            image_data = image['data']
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            images.append(image)

        text, weight = model.inference(images, max_length=15, beam_width=1, output_weights=False)

        results = {
            'results': [],
        }
        for t, i in zip(text, ids):
            # result = {
            #     'status': 0, # No error
            #     'id': image_id,
            #     'text': text,
            #     'weights': weight,
            # }
            result = {
                'status': 0, # No error
                'id': i,
                'text': t,
            }
            results['results'].append(result)

        return jsonify(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('-p', '--port', type=int, default=5000)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # checkpoint_path = '/home/aioz-intern-1/tf_exp/exp_tf_nlayers/tf_2de/checkpoints/epoch=48.ckpt'
    model = ModelTF.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()

    app = Flask(__name__)
    api = Api(app)

    api.add_resource(CERecognize, '/api/recognize')
    app.run(host='0.0.0.0', port=args.port, debug=False)
