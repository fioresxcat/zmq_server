import tensorflow as tf
import numpy as np
import cv2
import time

MODEL_PATH = 'mbnv2_128x64'
IMG_SIZE = (128, 64)
MAX_BATCH_SIZE = 4

class FeatureExtractor:
    def __init__(self, model_path=MODEL_PATH, img_size = IMG_SIZE, max_batch_size=MAX_BATCH_SIZE):
        print('loading tensorflow model')
        self.model = tf.saved_model.load(model_path)
        print('loaded model')

        self.img_size = img_size
        self.max_batch_size = max_batch_size
        img = np.random.randint(0, 255, ((self.max_batch_size,)+self.img_size+(3,)), dtype = np.uint8)
        img = list(img)
        img = self.preprocess(img)

        print('warming up tensorflow model...')
        for _ in range(50):
            pred = self.model.signatures['serving_default'](img)['output_1']
    
    def preprocess(self, data):
        data = [cv2.resize(img, self.img_size[::-1]) for img in data]
        # data = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(data))
        return tf.cast(data, tf.float32)

    def batch_padding(self, data):
        num_rois = len(data)
        num_batch = num_rois // self.max_batch_size
        remaining = num_rois - num_batch * self.max_batch_size
        data = self.preprocess(data)
        if remaining != 0:
            pad = tf.random.normal(((self.max_batch_size-remaining, ) + self.img_size + (3,)))
            data = tf.concat([data, pad], axis=0)
            num_batch += 1
        return data, num_batch, remaining

    def inference(self, data):
        data, num_batch, remaining = self.batch_padding(data)
        preds = None
        for i in range(num_batch):
            pred = self.model.signatures['serving_default'](data[i*self.max_batch_size:(i+1)*self.max_batch_size])['output_1']
            if i == 0:
                preds = pred
            else:
                preds = tf.concat([preds, pred], axis=0)
        if remaining != 0:
            preds = preds[:-(self.max_batch_size - remaining)]
        
        if preds is not None:
            preds = np.array(preds)

        return preds
        

if __name__ == '__main__':
    model = FeatureExtractor()
    img = np.random.randint(0, 255, IMG_SIZE+(3,), dtype = np.uint8)
    print(img.shape)

    ls_time = []
    for num in [3, 8, 10]:
        ls_roi = [img] * num
        start = time.perf_counter()
        out = model.inference(ls_roi)
        ls_time.append(time.perf_counter()-start)
        print('out: ', out.shape)
    
    print('mean infer time: ', np.mean(ls_time))
    
