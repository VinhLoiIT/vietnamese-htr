from flask import Flask, request
import numpy as np
from PIL import Image
import argparse
from system import CEInference, CTCInference
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['ce','ctc'], help='Use CESystem or CTCSystem')
    parser.add_argument('checkpoint', type=str, help='Checkpoint of System')
    parser.add_argument('--batch-size', type=int, default=8)
    # parser.add_argument('--batch-size', type=int, default=8)
    # parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    app = Flask(__name__)
    if args.type == 'ce':
        inference = CEInference(args.checkpoint, device)
    else:
        inference = CTCInference(args.checkpoint, device)

    @app.route('/api/recognize', methods=['POST'])
    def recognize():
        data = request.json
        images = data['images']
        results = []
        for image in images:
            image_data = np.array(image['data'], np.uint8)
            image_data = Image.fromarray(image_data)
            texts = inference.inference([image_data])
            results.append({'id': image['id'], 'text': ''.join(texts[0])})
        # batches = [images[i:i + args.batch_size] for i in range(0, len(images), args.batch_size)]
        # for batch in batches:
        #     batch = [(image['id'], Image.fromarray(np.array(image['data']))) for image in batch]
        #     image_ids: List[str]
        #     image_data: List[List]
        #     image_ids, image_data = list(zip(**batch))
        #     inference.inference(image_data)
        return {'results': results}

    app.run(host='127.0.0.1', port=8080, debug=True)


