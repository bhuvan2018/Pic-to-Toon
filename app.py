import os
import io
import uuid
import sys
import yaml
import traceback
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from flask import Flask, render_template, request, flash
from PIL import Image

with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)

sys.path.insert(0, './white_box_cartoonizer/')
from white_box_cartoonizer.cartoonize import WB_Cartoonize
if not opts['run_local']:
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        from gcloud_utils import upload_blob, generate_signed_url, delete_blob, download_video
    else:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS not set in environment variables")
    from video_api import api_request
app = Flask(__name__)
app.secret_key = 'my_super_secret_key_123!'
app.config['UPLOAD_FOLDER_IMAGES'] = 'static/uploaded_images'
app.config['UPLOAD_FOLDER_VIDEOS'] = 'static/uploaded_videos'
app.config['CARTOONIZED_FOLDER'] = 'static/cartoonized_images'
app.config['OPTS'] = opts
wb_cartoonizer = WB_Cartoonize(os.path.abspath("white_box_cartoonizer/saved_models/"), opts['gpu'])
for folder in [app.config['UPLOAD_FOLDER_IMAGES'], app.config['UPLOAD_FOLDER_VIDEOS'], app.config['CARTOONIZED_FOLDER']]:
    os.makedirs(folder, exist_ok=True)
def convert_bytes_to_image(img_bytes):
    """Convert image bytes to a NumPy array."""
    pil_image = Image.open(io.BytesIO(img_bytes))
    if pil_image.mode == "RGBA":
        image = Image.new("RGB", pil_image.size, (255, 255, 255))
        image.paste(pil_image, mask=pil_image.split()[3])
    else:
        image = pil_image.convert('RGB')
    return np.array(image)

def generate_histogram(image, img_name, mode="original"):
    """Generate and save a histogram for RGB channels."""
    plt.figure(figsize=(8, 4))
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title(f"{mode.capitalize()} Image Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    hist_path = os.path.join(app.config['CARTOONIZED_FOLDER'], f"{img_name}_{mode}_hist.jpg")
    plt.savefig(hist_path)
    plt.close()
    return hist_path

def generate_pie_chart(image, img_name, mode="original"):
    """Generate and save a pie chart showing the average color distribution."""
    avg_colors = np.mean(image, axis=(0, 1))
    labels = ['Red', 'Green', 'Blue']
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(5, 5))
    plt.pie(avg_colors, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f"{mode.capitalize()} Image Color Distribution")
    pie_chart_path = os.path.join(app.config['CARTOONIZED_FOLDER'], f"{img_name}_{mode}_pie.jpg")
    plt.savefig(pie_chart_path)
    plt.close()
    return pie_chart_path

def generate_bar_graph(image, img_name, mode="original"):
    """Generate and save a bar graph of the average RGB intensities."""
    avg_colors = np.mean(image, axis=(0, 1))
    labels = ['Red', 'Green', 'Blue']
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=avg_colors, palette=colors)
    plt.ylim(0, 255)
    plt.title(f"{mode.capitalize()} Image Average RGB Intensity")
    plt.xlabel("Color Channel")
    plt.ylabel("Intensity")
    bar_graph_path = os.path.join(app.config['CARTOONIZED_FOLDER'], f"{img_name}_{mode}_bar.jpg")
    plt.savefig(bar_graph_path)
    plt.close()
    return bar_graph_path

def process_video_locally(video_path):
    cartoonized_video_path = video_path.replace("uploaded_videos", "cartoonized_videos")
    os.makedirs(os.path.dirname(cartoonized_video_path), exist_ok=True)
    os.rename(video_path, cartoonized_video_path)
    return cartoonized_video_path

@app.route('/')
@app.route('/cartoonize', methods=["POST", "GET"])
def cartoonize():
    opts = app.config['OPTS']
    if request.method == 'POST':
        try:
            if request.files.get('image'):
                img = request.files["image"].read()
                image = convert_bytes_to_image(img)
                img_name = str(uuid.uuid4())
                orig_hist_path = generate_histogram(image, img_name, "original")
                orig_pie_path = generate_pie_chart(image, img_name, "original")
                orig_bar_path = generate_bar_graph(image, img_name, "original")
                cartoon_image = wb_cartoonizer.infer(image)
                cartoonized_img_name = os.path.join(app.config['CARTOONIZED_FOLDER'], img_name + ".jpg")
                cv2.imwrite(cartoonized_img_name, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))
                cartoon_hist_path = generate_histogram(cartoon_image, img_name, "cartoonized")
                cartoon_pie_path = generate_pie_chart(cartoon_image, img_name, "cartoonized")
                cartoon_bar_path = generate_bar_graph(cartoon_image, img_name, "cartoonized")
                return render_template("index_cartoonized.html", 
                                       cartoonized_image=cartoonized_img_name,
                                       orig_histogram=orig_hist_path,
                                       orig_pie_chart=orig_pie_path,
                                       orig_bar_graph=orig_bar_path,
                                       cartoon_histogram=cartoon_hist_path,
                                       cartoon_pie_chart=cartoon_pie_path,
                                       cartoon_bar_graph=cartoon_bar_path)
            elif request.files.get('video'):
                video = request.files["video"]
                video_filename = str(uuid.uuid4()) + ".mp4"
                video_path = os.path.join(app.config['UPLOAD_FOLDER_VIDEOS'], video_filename)
                video.save(video_path)
                print(f"📂 Video uploaded: {video_path}")
                try:
                    if opts.get('run_local', False):
                        cartoonized_video_path = process_video_locally(video_path)
                        cartoonized_video_url = f"/static/cartoonized/{video_filename}"
                    else:
                        video_url = upload_blob(video_path, "cartoonized_videos/" + video_filename)
                        response = api_request(video_url)
                        cartoonized_video_url = response.get("output_uri")
                    if not cartoonized_video_url:
                        flash("Error processing video. Please try again.")
                        return render_template("index_cartoonized.html")
                    return render_template("index_cartoonized.html", 
                                           original_video=video_url if not opts.get('run_local', False) else video_path, 
                                           cartoonized_video=cartoonized_video_url)
                except Exception:
                    traceback.print_exc()
                    flash("Error processing the image or video. Please try again.")
                    return render_template("index_cartoonized.html")
        except Exception:
            traceback.print_exc()
            flash("Error processing the request. Please try again.")
            return render_template("index_cartoonized.html")
    else:
        return render_template("index_cartoonized.html")
if __name__ == "__main__":
    if opts['colab-mode']:
        app.run()
    else:
        app.run(debug=True, host='127.0.0.1', port=8080)