AI Dating Helper
================

这是一个基于 FastAPI + 前端（HTML/JS）的 AI 小助手，用于提供约会建议、聊天话术和形象照片处理。
核心功能：
1. 聊天辅助 —— 调用大语言模型（Groq API）生成自然流畅的聊天建议。
2. 图片编辑 —— 使用 Replicate 模型 (timbrooks/instruct-pix2pix) 实现风格化或修图。
3. 图片放大 —— 使用 Real-ESRGAN 模型进行清晰化处理。
4. 抠图功能 —— 使用 Rembg 模型进行背景移除。

使用方法
--------
1. 启动后端：
   ```bash
   uvicorn main:app --reload
    默认运行在 http://127.0.0.1:8000
2. 启动前端：
    python -m http.server 5500
    然后在浏览器访问 http://127.0.0.1:5500
