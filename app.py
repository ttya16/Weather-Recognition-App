from keras.models import model_from_json
from keras.preprocessing import image
import keras
import tensorflow as tf

import numpy as np
import os
import shutil
from datetime import datetime

from PIL import Image

from flask import Flask, render_template, request, redirect, url_for, send_from_directory

app = Flask(__name__)

# 画像ファイル保存先
SAVE_DIR = "./images"

# モデルのパス
MODEL_PATH = './models/model.json'
WEIGHT_PATH = './models/weight_fine_tuning_2classes.hdf5'

model = model_from_json(open(MODEL_PATH).read())
model.load_weights(WEIGHT_PATH)

#tensorflow graphの初期化
#これをしておかないとmodel.predictでエラーになる。。
global graph
graph = tf.get_default_graph()

#Image size
IMG_H, IMG_W = 224, 224


@app.route('/')
def index():
    # アップロード画像はトップ画面に戻ると自動的に削除されるようにする。
    if os.path.isdir(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)

    return render_template('index.html')

@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory(SAVE_DIR, path)

@app.route('/result', methods=['POST'])
def upload():
    # アップロード画像を保存するディレクトリを用意
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    if request.files['image']:
        # 画像読み込み
        img_file = request.files['image']

        #stream = request.files['image'].stream
        #stream_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        #img = cv2.imdecode(stream_array, 1)
        # モデルに読み込ませるための処理
        #img = cv2.resize(img, (IMG_H, IMG_W))
        img_arr = image.load_img(img_file, target_size=(IMG_H, IMG_W))  
        img_arr = image.img_to_array(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = keras.applications.vgg16.preprocess_input(img_arr)


        # モデル予測
        with graph.as_default():
            pred = model.predict(img_arr)[0]


        if pred[0] > 0.5:
            if pred[0] > 0.9:
                result = '天気が悪そうですね'
            else:
                result = '悪い天気ではなさそうですが、雲が多いように見えます。'

        else:
            result = '良い天気のように見えます。'

        # アップロードされた画像はresult画面で表示させたいので、一旦保存
        dt_now = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_")
        save_path = os.path.join(SAVE_DIR, dt_now+".png")
        print(save_path)
        image_data = Image.open(img_file)
        image_data.save(save_path, 'png')

        #print(result)

        return render_template('result.html', path = save_path, result = result)


if __name__ == '__main__':
    app.run()


