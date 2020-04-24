from flask import Flask, request
from typing import List
import numpy as np
from PIL import Image, ImageOps
import argparse
from system import CEInference, CTCInference
import torch
from mnist import Net
from torchvision import transforms
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class FitBox(object):

#     def fit_box(self, image: np.ndarray):
#         hist_row = np.count_nonzero(binary_roi, axis=0) / binary_roi.shape[0]
#         hist_col = np.count_nonzero(binary_roi, axis=1) / binary_roi.shape[1]

#         left = first_nonzero(hist_row > 0.1, 0)
#         right = last_nonzero(hist_row > 0.1, 0) - binary_roi.shape[1] - 1
#         top = first_nonzero(hist_col > 0.01, 0)
#         bot = last_nonzero(hist_col > 0.01, 0) - binary_roi.shape[0] - 1

#         return (left,top,right,bot)

#     def __call__(self, pil_image: Image.Image) -> Image.Image:
#         nparray = np.array(pil_image)
#         nparray = fit_box(nparray)
#         image = Image.fromarray(nparray)
#         return image

def MNISTTransform():
    transform=transforms.Compose([
        ImageOps.invert,
        # FitBox(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return transform

def setupMNIST(checkpoint: str):
    net = Net()
    checkpoint = torch.load(checkpoint, map_location='cpu')
    net.load_state_dict(checkpoint)
    net.eval()
    for param in net.parameters():
        param.requires_grad_ = False
    return net

i = 0

@torch.no_grad()
def mnist_inference(net, tf, images: List[Image.Image]) -> List[str]:
    global i
    images = list(map(tf, images))
    transforms.ToPILImage()(images[0]).save(f'{i}.jpg')
    i += 1
    images = torch.stack(images) # [B,C,H,W]
    output = net(images) # [B,10]
    output = output.argmax(-1) # [B]
    output = list(map(str, output.cpu().tolist()))
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['ce','ctc'], help='Use CESystem or CTCSystem')
    parser.add_argument('checkpoint', type=str, help='Checkpoint of System')
    parser.add_argument('--mnist-checkpoint', type=str, default='./mnist_cnn.pt')
    parser.add_argument('--batch-size', type=int, default=8)
    # parser.add_argument('--batch-size', type=int, default=8)
    # parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    app = Flask(__name__)
    if args.type == 'ce':
        inference = CEInference(args.checkpoint, device)
    else:
        inference = CTCInference(args.checkpoint, device)

    mnist_net = setupMNIST(args.mnist_checkpoint)
    mnist_tf = MNISTTransform()

    @app.route('/api/recognize', methods=['POST'])
    def recognize():
        data = request.json
        images = data['images']
        results = []
        for image in images:
            image_data = np.array(image['data'], np.uint8)
            image_type = image.get('type', 'text')
            image_data = Image.fromarray(image_data)
            if image_type == 'text':
                text = inference.inference([image_data])[0]
            elif image_type == 'digit':
                text = mnist_inference(mnist_net, mnist_tf, [image_data])[0]
            results.append({'id': image['id'], 'text': ''.join(text)})
        # batches = [images[i:i + args.batch_size] for i in range(0, len(images), args.batch_size)]
        # for batch in batches:
        #     batch = [(image['id'], Image.fromarray(np.array(image['data']))) for image in batch]
        #     image_ids: List[str]
        #     image_data: List[List]
        #     image_ids, image_data = list(zip(**batch))
        #     inference.inference(image_data)
        return {'results': results}

    app.run(host='127.0.0.1', port=8080, debug=True)


