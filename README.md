# 📸 AI Image-to-Video Generator (Wan 2.2 I2V)

Transform any image into a **cinematic AI video** using diffusion models and motion interpolation.

This project leverages the **Wan 2.2 Image-to-Video (14B) diffusion model** with **FP8 quantization** and **AoT compiled transformer blocks** to generate smooth animated videos from a single image and a text prompt.

The system combines **diffusion-based motion synthesis, frame interpolation, and GPU optimization techniques** to produce high-quality animated clips efficiently.

---

# 🚀 Demo

Example input images used in this project:

* Cinematic action scene
* Stylish selfie cat
* Astronauts exploring a moon base

These images are animated into **short cinematic videos using text prompts**.

---

# ✨ Features

* 🖼️ **Image → Video Generation**
* 🎬 **Text-Guided Animation**
* ⚡ **FP8 Quantized Diffusion Model**
* 🎥 **Frame Interpolation using RIFE**
* 🎛️ **Advanced Inference Controls**
* 🌐 **Interactive Gradio Web Interface**
* 📦 **Optimized for HuggingFace Spaces / ZeroGPU**

---

# 🧠 Model

This project uses the **Wan 2.2 Image-to-Video (I2V) Diffusion Model**.

The model generates intermediate frames from a static image and then enhances temporal smoothness using motion interpolation.

### Main Pipeline

```
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
```

The diffusion model generates video frames and **RIFE improves motion smoothness by estimating motion between frames**.

---

# 🏗️ System Architecture

```
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
Final Output Video
```

---

# 📂 Project Structure

```
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
    └── RIFE_HDv3.py      # Frame interpolation model
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```
git clone https://github.com/rahulchaudhary2005/Image_to_video_generation.git

cd ai-image-to-video
```

---

## 2️⃣ Install Python Dependencies

```
pip install -r requirements.txt
```

This installs required libraries such as:

* Diffusers
* Transformers
* TorchAO
* OpenCV
* Gradio

---

## 3️⃣ Install System Dependency

```
ffmpeg
```

FFmpeg is used for **video rendering and encoding**.

---

## 4️⃣ Run the Application

```
python app.py
```

The **Gradio web interface** will launch automatically in your browser.

---

# 🎮 Usage

### Step 1

Upload an **input image**.

### Step 2

Enter a **motion prompt**.

Example:

```
make this image come alive, cinematic motion, smooth animation
```

### Step 3

Adjust generation settings:

* Inference steps
* Video duration
* Guidance scale
* FPS multiplier

### Step 4

Click **Generate Video**.

The system will generate an animated video clip.

---

# 🎛️ Advanced Settings

| Parameter      | Description               |
| -------------- | ------------------------- |
| Steps          | Diffusion inference steps |
| Guidance Scale | Prompt strength           |
| Duration       | Length of generated video |
| FPS Multiplier | Motion smoothness         |
| Scheduler      | Diffusion sampling method |
| Seed           | Reproducibility           |

---

# 📊 Core Technologies

| Technology | Purpose                  |
| ---------- | ------------------------ |
| PyTorch    | Deep learning framework  |
| Diffusers  | Diffusion model pipeline |
| TorchAO    | Model quantization       |
| Gradio     | Interactive UI           |
| OpenCV     | Frame processing         |
| RIFE       | Frame interpolation      |
| FFmpeg     | Video rendering          |

---

# ⚡ Optimization Techniques

This project implements several performance optimizations:

* **FP8 Quantization**
* **AoT Compiled Transformer Blocks**
* **Scheduler Optimization**
* **Frame Interpolation instead of full frame generation**

These optimizations significantly **reduce GPU memory usage and inference time**.

---

# 🖼 Example Use Cases

* AI animation generation
* Content creation
* Generative filmmaking
* Social media video clips
* Game concept visualization

---

# 🔮 Future Improvements

Planned upgrades include:

* Video-to-Video generation
* Longer video duration
* Motion control prompts
* Camera movement control
* 3D scene generation

---

# 🤝 Contributing

Contributions are welcome!

```
Fork → Branch → Commit → Pull Request
```

---

# 📜 License

This project follows the license of the underlying models and libraries.

---

# ⭐ Support

If you find this project useful:

⭐ Star the repository
🍴 Fork the project
📢 Share it with the community

---

# 👨‍💻 Author

**Rahul Chaudhary**
AI / ML Engineer

Built with ❤️ using **diffusion models and generative AI**.
