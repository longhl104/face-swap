# AI-Powered Face Swap System

Develop an **AI-powered face swap model** capable of replacing a face in an input image with another face, while maintaining:
- **Natural facial expressions**
- **Accurate skin tone**
- **High-quality seamless blending**

The system must support **both images and videos**.

---

## **Requirements**

- **Dataset:**  
  [Labeled Faces in the Wild (LFW) on Kaggle](https://www.kaggle.com/datasets/atulanandjha/lfwpeople?resource=download)
- **Training Script:**  
  Implement a Python script for model training
- **Production-Ready Interface:**  
  Develop a robust script or service for real-time inference and usage
- **Accuracy Graphs:**  
  Generate and provide visual accuracy/loss metrics from training
- **Dataset Update Method:**  
  Provide a method for updating or augmenting the dataset

---

## **How to Download the Dataset**

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("atulanandjha/lfwpeople")

print("Path to dataset files:", path)
```

---

## **Bonus: Video Support**

- **Video Face Swapping:**  
  Extend the solution to handle face swaps on **video frames**, ensuring:
  - **Smooth pose and expression transitions** across frames
  - **Optimized for real-time processing**

---
