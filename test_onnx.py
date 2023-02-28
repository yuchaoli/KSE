import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from easy_inference.utils.pad_frames import pad_frames

# for amr_detection: [640, 480] = [w, h]
# for amr_pose:      [640, 512]

# input = np.random.rand(5, 3, 480, 640).astype(np.float32)

use_pose = False

img = Image.open('test.jpg')
if use_pose:
    img = img.resize((640, 512))
else:
    img = img.resize((640, 480))
img = img.convert('RGB')
img_t = np.array(img).astype('float32')
img_t = np.tile(img_t, (5,1,1,1))
img_t = img_t.transpose(0,3,1,2)
input = img_t / 255.

plt.imshow(input[0,:,:,:].transpose(1,2,0))
plt.show()

ort_sess = ort.InferenceSession('yolov7-tiny.onnx', providers=['CPUExecutionProvider']) # 'CUDAExecutionProvider'
output = ort_sess.run(None, {'images': input})[0]

n_obj = int(len(output)/5)
obj_ids = []
for x in output:
    if x[0] == 0:
        obj_ids.append(int(x[5]))

print('Found {} objects: {}'.format(n_obj, obj_ids))