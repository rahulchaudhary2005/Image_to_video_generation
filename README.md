📸 AI Image-to-Video Generator (Wan 2.2 I2V)

Transform any image into a cinematic AI video using diffusion models and motion interpolation.

This project uses the Wan 2.2 Image-to-Video (14B) model with FP8 quantization and AoT compilation to generate smooth animated videos from a single image and text prompt.

🚀 Demo

Example inputs used in the project:

cinematic action scene

stylish selfie cat

astronauts exploring a moon base

These images are animated into short videos using a text prompt.

✨ Features

🖼️ Image → Video generation

🎬 Text-guided animation

⚡ FP8 quantized diffusion model

🎥 Frame interpolation using RIFE

🎛️ Advanced inference controls

🌐 Interactive Gradio Web UI

📦 Optimized for HuggingFace Spaces / ZeroGPU

🧠 Model

This project uses the Wan 2.2 Image-to-Video (I2V) diffusion model.

Main pipeline:

Input Image
     │
     ▼
Prompt Conditioning
     │
     ▼
Wan 2.2 Diffusion Transformer
     │
     ▼
Frame Generation
     │
     ▼
RIFE Frame Interpolation
     │
     ▼
Final Video Rendering

The model generates video frames and then improves motion smoothness using interpolation.

🏗️ Architecture
User Input
   │
   ▼
Gradio Interface
   │
   ▼
Image Processing
   │
   ▼
Wan I2V Diffusion Pipeline
   │
   ▼
Frame Generation
   │
   ▼
RIFE Frame Interpolation
   │
   ▼
Video Encoding (FFmpeg)
   │
   ▼
Output Video
📂 Project Structure
AI-Image-To-Video/
│
├── app.py                # Main Gradio application
├── aoti.py               # AoT compiled model loading
├── requirements.txt      # Python dependencies
├── packages.txt          # System dependencies
├── README.md
│
├── assets/
│   ├── input1.jpg
│   ├── input2.jpg
│   └── input3.jpg
│
└── train_log/
    └── RIFE_HDv3.py
⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/rahulchaudhary2005/ai-image-to-video.git

cd ai-image-to-video
2️⃣ Install dependencies
pip install -r requirements.txt

Dependencies include libraries like Diffusers, Transformers, TorchAO and OpenCV.

3️⃣ Install system packages
ffmpeg

This project uses FFmpeg for video rendering.

4️⃣ Run the application
python app.py

The Gradio interface will launch in your browser.

🎮 Usage

1️⃣ Upload an image
2️⃣ Enter a motion prompt

Example:

make this image come alive, cinematic motion, smooth animation

3️⃣ Adjust settings:

inference steps

duration

guidance scale

video FPS

4️⃣ Click Generate Video

The system will generate an animated video.

🎛️ Advanced Settings

The interface allows advanced control over generation:

Parameter	Description
Steps	Diffusion inference steps
Guidance Scale	Prompt strength
Duration	Video length
FPS Multiplier	Smooth motion
Scheduler	Diffusion sampling method
Seed	Reproducibility
📊 Core Technologies

This project combines multiple AI and multimedia tools:

Technology	Purpose
PyTorch	Deep learning framework
Diffusers	Diffusion model pipeline
TorchAO	Model quantization
Gradio	Web UI
OpenCV	Frame processing
RIFE	Frame interpolation
FFmpeg	Video rendering
⚡ Optimization Techniques

To improve performance the project uses:

FP8 quantization

AoT compiled blocks

scheduler optimization

frame interpolation instead of generating extra frames

These reduce GPU requirements significantly.

🖼 Example Use Cases

AI animation

content creation

generative filmmaking

social media clips

game concept visualization

🔮 Future Improvements

Planned upgrades:

video-to-video generation

longer video generation

motion control

camera movement prompts

3D scene generation

🤝 Contributing

Contributions are welcome!

fork → branch → commit → pull request
📜 License

This project follows the license of the underlying models and libraries.

⭐ Support

If you find this project useful:

⭐ Star the repository
🍴 Fork it
📢 Share it

👨‍💻 Author

Rahul chaudhary AIML Engineer

Built with ❤️ using diffusion models and generative AI.
