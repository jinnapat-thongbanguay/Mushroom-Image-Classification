# Mushroom Image Classification with RAG & CLIP

โปรเจกต์นี้เป็นระบบ จำแนกชนิดเห็ดจากภาพ (Mushroom Image Classification) พัฒนาโดยใช้แนวคิด Hybrid Computer Vision + Retrieval-Augmented Generation (RAG)
ผสานการทำงานของ CLIP, FAISS, และ PyTorch พร้อมแอปพลิเคชันสำหรับใช้งานจริงผ่าน Streamlit

## Features
- จำแนกชนิดเห็ดจากภาพถ่าย
- ใช้ CLIP (Contrastive Language–Image Pretraining) สำหรับดึง feature จากภาพ
- ใช้ FAISS เป็น Vector Database สำหรับค้นหาข้อมูลตัวอย่างที่ใกล้เคียง (RAG)
- ใช้ PyTorch สำหรับจัดการโมเดลและ inference
- แสดงผลผ่าน Web Application ด้วย Streamlit
- รองรับการอัปโหลดภาพจากผู้ใช้เพื่อทำการจำแนกแบบ Real-time

## System Overview
ระบบประกอบด้วย 3 ส่วนหลัก:
1. **Visual Encoder (CLIP)**  
   แปลงภาพเห็ดเป็น embedding vector
2. **Retrieval Module (FAISS-based RAG)**  
   ค้นหาภาพตัวอย่างที่ใกล้เคียงจาก Knowledge Base
3. **Classification Pipeline**  
   ใช้ข้อมูลที่ retrieved มาช่วยเพิ่มความแม่นยำในการจำแนก

## Tech Stack
- **Language:** Python  
- **Deep Learning Framework:** PyTorch  
- **Computer Vision Model:** CLIP  
- **Vector Database:** FAISS  
- **Web Framework:** Streamlit  
- **Concept:** Retrieval-Augmented Generation (RAG)

## Inference Result Visualization
<img width="2880" height="6011" alt="screencapture-mushroom-app-oy86puszyy-streamlit-app-2025-12-22-19_04_52" src="https://github.com/user-attachments/assets/88225f4e-53ff-4900-9bff-893d3b928a50" />

## Live Demo
The web application is deployed using **Streamlit** and can be accessed at:  
🔗 https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/](https://mushroom-app-oy86puszyy.streamlit.app/

