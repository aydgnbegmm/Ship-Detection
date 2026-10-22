# 🚢 Deep Learning-Based Maritime Ship Detection & Tracking (SAYZEK-ATP)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-YOLOv8%20%7C%20PyTorch-orange.svg)](https://ultralytics.com/)
[![Program](https://img.shields.io/badge/Program-SAYZEK--ATP%20(Savunma%20Sanayii)-purple.svg)](https://sayzek.gov.tr/)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-green.svg)](https://drive.google.com/drive/folders/1--IUndq-0Vd9dJYt-wrboS0fQmBK-ZYX?usp=drive_link)
[![License](https://img.shields.io/badge/License-MIT-gray.svg)](LICENSE)

An end-to-end computer vision and deep learning framework for **robust maritime vessel detection and deck localization** under harsh marine environmental conditions (fog, rain, sea glare, high sea state turbulence, and varying scale/distance).

Developed within the scope of the **Savunma Sanayii Yapay Zeka Yetenek Kümelenmesi - Akademik Tez Programı (SAYZEK-ATP 2024-2025)** under the academic thesis titled:  
> *"Otonom Uçak İniş Sistemlerinde Derin Öğrenme Tabanlı Görüntü İşleme Teknikleri"*  
> (*Deep Learning-Based Computer Vision Techniques for Autonomous UAV Landing on Maritime Vessels*)

---

## 📌 Problem & Motivation

Autonomous landing of Unmanned Aerial Vehicles (UAVs) onto dynamic naval vessels represents one of the most demanding challenges in aerospace and defense:
1. **Marine Environment Degradation:** Wave dynamics, sea spray, fog, rain, and solar reflection (sunglint) severely degrade standard RGB camera sensors.
2. **GNSS Vulnerability:** Satellite navigation signals suffer from multipath degradation and electromagnetic interference in open sea operations.
3. **Dynamic Reference Frames:** Carrier vessels continuously pitch, roll, and heave in 6 degrees of freedom.

This repository provides an aggregated, annotated multi-weather maritime dataset paired with high-performance YOLOv8 detection and tracking pipelines tailored for edge compute avionics.

---

## 🏗️ System Pipeline

```
  ┌─────────────────────────┐
  │ Onboard Maritime Camera │
  │ (RGB / Optical Sensor)  │
  └────────────┬────────────┘
               │ Raw Frames
               ▼
  ┌─────────────────────────┐
  │   Pre-Processing &      │
  │   OSD/Glare Filtering   │
  └────────────┬────────────┘
               │ Filtered Stream
               ▼
  ┌─────────────────────────┐
  │    YOLOv8 Maritime      │
  │    Detection Engine     │ ◀─── Trained on Multi-Weather Dataset
  └────────────┬────────────┘
               │ Bounding Boxes & Classes [Ship / Deck / Vessel]
               ▼
  ┌─────────────────────────┐
  │   Visual Tracking &     │
  │   Deck Pose Estimation  │ ───▶ Navigation & Guidance Controller
  └─────────────────────────┘      (Autonomous Carrier Approach)
```

---

## 📊 Dataset Information

A consolidated, multi-condition dataset annotated in **Ultralytics YOLO format**, curated specifically for robust maritime target classification across adverse weather scenarios:

*   **Download Dataset:** [Google Drive Maritime Dataset Link](https://drive.google.com/drive/folders/1--IUndq-0Vd9dJYt-wrboS0fQmBK-ZYX?usp=drive_link)
*   **Annotated Classes:**
    *   `0: ship` (General commercial & civilian vessels)
    *   `1: naval_vessel` (Frigates, corvettes, military vessels)
    *   `2: cargo_ship` (Container and bulk carriers)
    *   `3: landing_deck` (Helideck / UAV recovery platform)
*   **Weather Conditions Covered:** Sunny/Clear, Dense Fog, Rainy, Sunglint/Direct Glare, Twilight/Night conditions, and Choppy High-Seas.

---

## 📂 Repository Structure

```
├── data.yaml              # Dataset path and class definitions
├── inference.py           # Real-time inference script with FPS and telemetry overlay
├── requirements.txt       # Environment dependencies
├── .gitignore
└── README.md              # Research & technical documentation
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/aydgnbegmm/Ship-Detection.git
cd Ship-Detection
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1--IUndq-0Vd9dJYt-wrboS0fQmBK-ZYX?usp=drive_link) and extract it into a `./dataset` folder matching the structure defined in `data.yaml`:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 3. Model Training
To train YOLOv8 on the maritime dataset:
```bash
yolo task=detect mode=train model=yolov8m.pt data=data.yaml epochs=100 imgsz=640 batch=16
```

### 4. Running Inference
Run the inference engine on test images, recorded video streams, or a live RTSP camera:
```bash
python inference.py --weights weights/best.pt --source data/test_video.mp4 --show --conf 0.40
```

---

## 👥 Academic & Research Affiliation

*   **Researcher:** **Begüm Aydoğan Çiçin** (*Mechatronics Engineering, Necmettin Erbakan University*)
*   **Academic Advisor:** **Dr. Öğr. Üyesi Alperen Eroğlu** (*Computer Engineering, Necmettin Erbakan University*)
*   **Industry Mentor:** **Alperen Taştekin** (*AI Software Engineer, Turkish Aerospace Industries / TUSAŞ*)
*   **Program:** Savunma Sanayii Yapay Zeka Yetenek Kümelenmesi (**SAYZEK**) - Akademik Tez Programı (**ATP**)

---

## 📬 Contact & Profiles

*   **Author:** Begüm Aydoğan Çiçin
*   **LinkedIn:** [linkedin.com/in/begüm-aydoğan-çiçin-a94098201](https://www.linkedin.com/in/begüm-aydoğan-çiçin-a94098201)
*   **Email:** `aydgnbegm@gmail.com`
