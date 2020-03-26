import time
import numpy as np

import cv2
import imutils
from PIL import Image

import torch
from torchvision import transforms

from style_transfer.models import TransformerNet
from style_transfer.func import test_image_transform

#torch.set_num_threads(4)
net = TransformerNet(alpha=0.3)
net.load_state_dict(torch.load("trained_models/starry_night_small.pth",
                               map_location='cpu'))
net.eval()

cap = cv2.VideoCapture(0)
t0 = time.time()
frame_cnt = 0
fps = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #frame = imutils.resize(frame, width=256)
    frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    with torch.no_grad():
        x = test_image_transform(img)
        x = x.unsqueeze(0)
        gen = net(x)
    gen = gen.squeeze().permute([1, 2, 0])
    gen = (gen.detach().numpy() * 255).astype(np.uint8)
    gen = cv2.cvtColor(gen, cv2.COLOR_RGB2BGR)

    cv2.putText(gen, f"{fps} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)
    cv2.imshow('style transfer', gen)
    if time.time() - t0 > 1:
        fps = int(frame_cnt / (time.time() - t0))
        frame_cnt = 0
        t0 = time.time()
    frame_cnt += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
