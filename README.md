# SECURECAM — AI-Powered Security Camera

An AI-powered security monitoring web application built using **Python, FastAPI, and YOLO object detection**. SECURECAM provides a web interface for running object detection using a custom-trained model.

---

## 📸 Screenshots

<!-- Replace the image paths below with your uploaded screenshot paths. -->

### Home Page

![SECURECAM Home Page](screenshots/homepage.png)

### Object Detection Interface

![Object Detection Interface](screenshots/detection.png)

### Detection Results

![Detection Results](screenshots/results.png)

---

## 🚀 Features

- **YOLO-based Detection:** Uses a custom-trained YOLO model for object detection.
- **FastAPI Backend:** Provides a lightweight Python API.
- **Web Interface:** Simple browser-based interface built with HTML, CSS, and JavaScript.
- **Custom Model Support:** Uses trained model weights stored in `best.pt`.
- **Real-Time Processing Support:** Designed for camera-based detection, depending on the configured input source.
- **REST API:** FastAPI endpoints with interactive API documentation.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| FastAPI | API and web server |
| Ultralytics YOLO | Object detection |
| OpenCV | Image and video processing |
| HTML, CSS, JavaScript | Frontend |
| Uvicorn | ASGI server |

---

## 📂 Project Structure

```text
SECURECAM/
│
├── static/
│   └── index.html
│
├── runs/
│   └── detect/
│       └── sixray_model/
│           └── weights/
│               └── best.pt
│
├── screenshots/
│   ├── homepage.png
│   ├── detection.png
│   └── results.png
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/HarshuPandhare/SECURECAM.git
cd SECURECAM
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### 5. Open the Website

Visit:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🧠 Model Information

SECURECAM uses a custom-trained YOLO model.

- **Model file:** `best.pt`
- **Framework:** Ultralytics YOLO
- **Purpose:** Object detection using the trained model.

Place the trained weights at the path expected by `app.py`. Update the model path in the application if your directory structure differs.

---

## 🌐 Deployment

The project was prepared for deployment using **Microsoft Azure App Service** with GitHub integration.

Deployment requires a compatible Python runtime, the project's dependencies, and sufficient memory and storage for the model and inference workload.

---

## 🔮 Future Improvements

- Live camera stream integration
- Detection history and event logging
- Security alerts and notifications
- Dashboard for monitoring detection activity
- User authentication and access control
- Cloud-based monitoring and storage

---

## 👨‍💻 Author

**Harshvardhan Pandhare**

GitHub: [@HarshuPandhare](https://github.com/HarshuPandhare)

---

## 📄 License

This project is intended for educational and development purposes. Add a license file if you want to specify the terms under which others may use or modify this project.
