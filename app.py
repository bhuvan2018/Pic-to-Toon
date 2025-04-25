import os
import io
import uuid
import sys
import flet as ft
import yaml
import traceback
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

# Load configuration
with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)

# Import cartoonizer
sys.path.insert(0, './white_box_cartoonizer/')
from white_box_cartoonizer.cartoonize import WB_Cartoonize

# Configure cloud services if needed
if not opts['run_local']:
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        from gcloud_utils import upload_blob, generate_signed_url, delete_blob, download_video
    else:
        print("GOOGLE_APPLICATION_CREDENTIALS not set in environment variables")
    from video_api import api_request

# Initialize cartoonizer
wb_cartoonizer = WB_Cartoonize(os.path.abspath("white_box_cartoonizer/saved_models/"), opts['gpu'])

# Create directories
UPLOAD_FOLDER_IMAGES = 'static/uploaded_images'
UPLOAD_FOLDER_VIDEOS = 'static/uploaded_videos'
CARTOONIZED_FOLDER = 'static/cartoonized_images'

for folder in [UPLOAD_FOLDER_IMAGES, UPLOAD_FOLDER_VIDEOS, CARTOONIZED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Helper functions
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
    hist_path = os.path.join(CARTOONIZED_FOLDER, f"{img_name}_{mode}_hist.jpg")
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
    pie_chart_path = os.path.join(CARTOONIZED_FOLDER, f"{img_name}_{mode}_pie.jpg")
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
    bar_graph_path = os.path.join(CARTOONIZED_FOLDER, f"{img_name}_{mode}_bar.jpg")
    plt.savefig(bar_graph_path)
    plt.close()
    return bar_graph_path

def process_video_locally(video_path):
    cartoonized_video_path = video_path.replace("uploaded_videos", "cartoonized_videos")
    os.makedirs(os.path.dirname(cartoonized_video_path), exist_ok=True)
    os.rename(video_path, cartoonized_video_path)
    return cartoonized_video_path

# Main app function
def main(page: ft.Page):
    page.title = "Pic-to-Toon"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = "#1D85A2"  # The same background color as in the HTML
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO
    
    # Status message display
    status_message = ft.Text(
        size=18,
        color=ft.colors.WHITE,
        text_align=ft.TextAlign.CENTER,
        visible=False
    )
    
    # Loading indicator
    loading = ft.ProgressRing(
        width=40, 
        height=40, 
        stroke_width=4, 
        color=ft.colors.WHITE,
        visible=False
    )
    
    # Results containers
    cartoonized_result = ft.Container(visible=False)
    visualization_buttons = ft.Container(visible=False)
    
    # Graph containers
    histogram_container = ft.Container(visible=False)
    pie_chart_container = ft.Container(visible=False)
    bar_graph_container = ft.Container(visible=False)
    
    # Create header
    header = ft.Column([
        ft.Text(
            "Pic-to-Toon Cartoonify Your Memories",
            size=36,
            weight=ft.FontWeight.BOLD,
            color=ft.colors.WHITE,
            text_align=ft.TextAlign.CENTER,
        ),
        ft.Text(
            "Transform Your Favourite Images",
            size=24,
            color="#E3F2FD",
            text_align=ft.TextAlign.CENTER,
        ),
    ], alignment=ft.MainAxisAlignment.CENTER)
    
    # Function to show histogram
    def show_histogram(e):
        histogram_container.visible = True
        pie_chart_container.visible = False
        bar_graph_container.visible = False
        page.update()
    
    # Function to show pie chart
    def show_pie_chart(e):
        histogram_container.visible = False
        pie_chart_container.visible = True
        bar_graph_container.visible = False
        page.update()
    
    # Function to show bar graph
    def show_bar_graph(e):
        histogram_container.visible = False
        pie_chart_container.visible = False
        bar_graph_container.visible = True
        page.update()
    
    # Function to handle image processing
    def process_image(e: ft.FilePickerResultEvent):
        if not e.files or len(e.files) == 0:
            return
        
        try:
            # Show loading
            loading.visible = True
            status_message.value = "Processing your image..."
            status_message.visible = True
            page.update()
            
            # Reset all visualization containers
            histogram_container.visible = False
            pie_chart_container.visible = False
            bar_graph_container.visible = False
            page.update()
            
            # Read the file
            file_path = e.files[0].path
            with open(file_path, "rb") as f:
                img_bytes = f.read()
                
            # Process the image
            image = convert_bytes_to_image(img_bytes)
            img_name = str(uuid.uuid4())
            
            # Generate original analysis
            orig_hist_path = generate_histogram(image, img_name, "original")
            orig_pie_path = generate_pie_chart(image, img_name, "original")
            orig_bar_path = generate_bar_graph(image, img_name, "original")
            
            # Cartoonize the image
            cartoon_image = wb_cartoonizer.infer(image)
            cartoonized_img_name = os.path.join(CARTOONIZED_FOLDER, img_name + ".jpg")
            cv2.imwrite(cartoonized_img_name, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))
            
            # Generate cartoonized analysis
            cartoon_hist_path = generate_histogram(cartoon_image, img_name, "cartoonized")
            cartoon_pie_path = generate_pie_chart(cartoon_image, img_name, "cartoonized")
            cartoon_bar_path = generate_bar_graph(cartoon_image, img_name, "cartoonized")
            
            # Update the UI with the results
            cartoonized_result.content = ft.Column([
                ft.Image(
                    src=cartoonized_img_name,
                    width=500,
                    fit=ft.ImageFit.CONTAIN,
                    border_radius=10
                ),
                ft.FilledButton(
                    text="Download",
                    icon=ft.icons.DOWNLOAD,
                    on_click=lambda _: page.launch_url(cartoonized_img_name)
                ),
                ft.Text(
                    "(Valid for 5 minutes only!)",
                    size=16,
                    weight=ft.FontWeight.BOLD,
                    color="#7300FF",
                    text_align=ft.TextAlign.CENTER,
                )
            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
            cartoonized_result.visible = True
            
            # Setup visualization buttons
            visualization_buttons.content = ft.Column([
                ft.Text(
                    "Color Distribution & Intensity Analysis",
                    size=26,
                    weight=ft.FontWeight.BOLD,
                    color=ft.colors.WHITE,
                    text_align=ft.TextAlign.CENTER
                ),
                ft.Row([
                    ft.ElevatedButton(
                        text="Histogram",
                        icon=ft.icons.BAR_CHART,
                        style=ft.ButtonStyle(
                            color=ft.colors.WHITE,
                            bgcolor=ft.colors.BLUE,
                            shape=ft.RoundedRectangleBorder(radius=8)
                        ),
                        on_click=show_histogram
                    ),
                    ft.ElevatedButton(
                        text="Pie Chart",
                        icon=ft.icons.PIE_CHART,
                        style=ft.ButtonStyle(
                            color=ft.colors.WHITE,
                            bgcolor=ft.colors.GREEN,
                            shape=ft.RoundedRectangleBorder(radius=8)
                        ),
                        on_click=show_pie_chart
                    ),
                    ft.ElevatedButton(
                        text="Bar Graph",
                        icon=ft.icons.ANALYTICS,
                        style=ft.ButtonStyle(
                            color=ft.colors.WHITE,
                            bgcolor=ft.colors.ORANGE,
                            shape=ft.RoundedRectangleBorder(radius=8)
                        ),
                        on_click=show_bar_graph
                    )
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=10)
            ], alignment=ft.MainAxisAlignment.CENTER)
            visualization_buttons.visible = True
            
            # Setup the graph containers
            histogram_container.content = ft.ResponsiveRow([
                ft.Column([
                    ft.Text("Original Image", size=18, color=ft.colors.WHITE),
                    ft.Image(src=orig_hist_path, width=350, fit=ft.ImageFit.CONTAIN, border_radius=10)
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, col={"sm": 12, "md": 6}),
                ft.Column([
                    ft.Text("Cartoonized Image", size=18, color=ft.colors.WHITE),
                    ft.Image(src=cartoon_hist_path, width=350, fit=ft.ImageFit.CONTAIN, border_radius=10)
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, col={"sm": 12, "md": 6})
            ])
            
            pie_chart_container.content = ft.ResponsiveRow([
                ft.Column([
                    ft.Text("Original Image", size=18, color=ft.colors.WHITE),
                    ft.Image(src=orig_pie_path, width=350, fit=ft.ImageFit.CONTAIN, border_radius=10)
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, col={"sm": 12, "md": 6}),
                ft.Column([
                    ft.Text("Cartoonized Image", size=18, color=ft.colors.WHITE),
                    ft.Image(src=cartoon_pie_path, width=350, fit=ft.ImageFit.CONTAIN, border_radius=10)
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, col={"sm": 12, "md": 6})
            ])
            
            bar_graph_container.content = ft.ResponsiveRow([
                ft.Column([
                    ft.Text("Original Image", size=18, color=ft.colors.WHITE),
                    ft.Image(src=orig_bar_path, width=350, fit=ft.ImageFit.CONTAIN, border_radius=10)
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, col={"sm": 12, "md": 6}),
                ft.Column([
                    ft.Text("Cartoonized Image", size=18, color=ft.colors.WHITE),
                    ft.Image(src=cartoon_bar_path, width=350, fit=ft.ImageFit.CONTAIN, border_radius=10)
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, col={"sm": 12, "md": 6})
            ])
            
            # Hide loading and update message
            loading.visible = False
            status_message.value = "Your cartoon is ready!"
            page.update()
            
        except Exception as e:
            # Handle errors
            loading.visible = False
            status_message.value = f"Error: {str(e)}"
            status_message.color = ft.colors.RED_ACCENT
            status_message.visible = True
            print(traceback.format_exc())
            page.update()
    
    # Function to handle video processing
    def process_video(e: ft.FilePickerResultEvent):
        if not e.files or len(e.files) == 0:
            return
        
        try:
            # Show loading
            loading.visible = True
            status_message.value = "Processing your video..."
            status_message.visible = True
            page.update()
            
            # Hide any visualization containers
            visualization_buttons.visible = False
            histogram_container.visible = False
            pie_chart_container.visible = False
            bar_graph_container.visible = False
            page.update()
            
            # Read the file
            file_path = e.files[0].path
            file_size = os.path.getsize(file_path) / 1024  # KB
            max_file_size = 30720  # 30MB in KB
            
            if file_size >= max_file_size:
                loading.visible = False
                status_message.value = "File too Big, please select a file less than 30MB"
                status_message.color = ft.colors.RED_ACCENT
                status_message.visible = True
                page.update()
                return
            
            # Process the video
            video_filename = str(uuid.uuid4()) + ".mp4"
            video_path = os.path.join(UPLOAD_FOLDER_VIDEOS, video_filename)
            
            # Copy the file to our upload folder
            with open(file_path, "rb") as src, open(video_path, "wb") as dst:
                dst.write(src.read())
            
            if opts.get('run_local', False):
                cartoonized_video_path = process_video_locally(video_path)
                cartoonized_video_url = f"/static/cartoonized/{video_filename}"
            else:
                video_url = upload_blob(video_path, "cartoonized_videos/" + video_filename)
                response = api_request(video_url)
                cartoonized_video_url = response.get("output_uri")
            
            if not cartoonized_video_url:
                loading.visible = False
                status_message.value = "Error processing video. Please try again."
                status_message.color = ft.colors.RED_ACCENT
                status_message.visible = True
                page.update()
                return
            
            # Update the UI with results
            cartoonized_result.content = ft.Column([
                ft.Text("Cartoonized Video", size=22, color=ft.colors.WHITE),
                ft.Container(
                    content=ft.Video(
                        src=cartoonized_video_url,
                        width=500,
                        height=320,
                        autoplay=False,
                        controls=True
                    ),
                    border_radius=10
                ),
                ft.FilledButton(
                    text="Download",
                    icon=ft.icons.DOWNLOAD,
                    on_click=lambda _: page.launch_url(cartoonized_video_url)
                )
            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
            cartoonized_result.visible = True
            
            # Hide loading and update
            loading.visible = False
            status_message.value = "Your cartoonized video is ready!"
            page.update()
            
        except Exception as e:
            # Handle errors
            loading.visible = False
            status_message.value = f"Error: {str(e)}"
            status_message.color = ft.colors.RED_ACCENT
            status_message.visible = True
            print(traceback.format_exc())
            page.update()
    
    # Set up file pickers
    image_picker = ft.FilePicker(on_result=process_image)
    video_picker = ft.FilePicker(on_result=process_video)
    page.overlay.extend([image_picker, video_picker])
    
    # Create upload buttons
    upload_container = ft.Column([
        ft.ElevatedButton(
            content=ft.Row([
                ft.Icon(ft.icons.IMAGE),
                ft.Text("Get Cartoon Magic!", size=18)
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
            style=ft.ButtonStyle(
                color=ft.colors.WHITE,
                bgcolor=ft.colors.BLUE,
                shape=ft.RoundedRectangleBorder(radius=12),
                padding=20
            ),
            on_click=lambda _: image_picker.pick_files(
                allow_multiple=False,
                file_type=ft.FilePickerFileType.IMAGE
            )
        )
    ], alignment=ft.MainAxisAlignment.CENTER)
    
    # Create sample images section
    sample_section = ft.Column([
        ft.Divider(height=2, color=ft.colors.WHITE24),
        ft.Text("Sample Images", size=24, color=ft.colors.WHITE, text_align=ft.TextAlign.CENTER),
        ft.ResponsiveRow([
            ft.Column([
                ft.Image(
                    src="/static/sample_images/emma2.jpg",
                    width=300,
                    fit=ft.ImageFit.CONTAIN,
                    border_radius=10
                )
            ], col={"sm": 12, "md": 6}, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Column([
                ft.Image(
                    src="/static/sample_images/emma2_cartoonized.jpg",
                    width=300,
                    fit=ft.ImageFit.CONTAIN,
                    border_radius=10
                )
            ], col={"sm": 12, "md": 6}, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        ]),
        ft.ResponsiveRow([
            ft.Column([
                ft.Image(
                    src="/static/sample_images/spice.jpeg",
                    width=300,
                    fit=ft.ImageFit.CONTAIN,
                    border_radius=10
                )
            ], col={"sm": 12, "md": 6}, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Column([
                ft.Image(
                    src="/static/sample_images/spice_cartoonized.jpeg",
                    width=300,
                    fit=ft.ImageFit.CONTAIN,
                    border_radius=10
                )
            ], col={"sm": 12, "md": 6}, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        ]),
        ft.ResponsiveRow([
            ft.Column([
                ft.Image(
                    src="/static/sample_images/cake.jpeg",
                    width=300,
                    fit=ft.ImageFit.CONTAIN,
                    border_radius=10
                )
            ], col={"sm": 12, "md": 6}, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Column([
                ft.Image(
                    src="/static/sample_images/cake_cartoonized.jpeg",
                    width=300,
                    fit=ft.ImageFit.CONTAIN,
                    border_radius=10
                )
            ], col={"sm": 12, "md": 6}, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        ])
    ], scroll=ft.ScrollMode.AUTO)
    
    # Footer
    footer = ft.Container(
        content=ft.Row([
            ft.Icon(ft.icons.COPYRIGHT_OUTLINED, color=ft.colors.AMBER),
            ft.Text(" 2025 - Made by Bhuvan & Ranjan", color=ft.colors.WHITE, size=18)
        ], alignment=ft.MainAxisAlignment.CENTER),
        padding=ft.padding.only(top=20, bottom=20)
    )
    
    # Add all components to the page
    page.add(
        header,
        ft.Container(height=20),
        status_message,
        ft.Container(
            content=loading,
            alignment=ft.alignment.center
        ),
        ft.Container(height=20),
        upload_container,
        ft.Container(height=20),
        cartoonized_result,
        ft.Container(height=20),
        visualization_buttons,
        ft.Container(height=10),
        histogram_container,
        pie_chart_container,
        bar_graph_container,
        ft.Container(height=40),
        sample_section,
        footer
    )

# Run the app
if __name__ == "__main__":
    ft.app(target=main)