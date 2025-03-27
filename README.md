# 基于 PaddleOCR 的文件处理项目

## 〇、项目介绍

本项目希望借助 PaddleOCR 的 OCR 模型实现小票信息的处理。然后可能会进一步处理成文档。

## 一、项目环境

Python 版本：3.11

依赖的三方库：

- paddlepaddle==3.0.0rc1（目前官方文档快速入门中所使用的版本）
- paddleocr
- opencv-python（对图片进行预处理）
- protobuf==5.29.3 (5.29.3 是目前最高的支持 windows 的版本)
- setuptools（没有会导致 paddlepaddle 或者 paddleocr 报错）
- ipykernel（如果需要运行 notebooks 的话才需要，否则可以不装）

## 二、项目功能

- 基于 OpenCV 的图片预处理
- OCR 实现文字识别
- 文字内容格式整理和封装
- （预期）文档生成

## 三、未来计划

- [ ] 首先实现现有决定的项目功能
- [ ] 根据需求进行其他可能的功能的设计

## N、致谢

感谢 [p](httocr//github.com/PaddlePaddle/PaddleOCR) 项目的开源！
