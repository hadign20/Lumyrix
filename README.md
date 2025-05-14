# 🔬 Lumyrix: A Modular Deep Learning Playground

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange)

**Lumyrix** is a scalable, modular deep learning framework designed to help you practice and master real-world ML workflows. Whether you're training CNNs on images, extracting radiomics features from medical data, or experimenting with classical ML models, Lumyrix gives you a clean, reproducible structure powered by PyTorch, GitHub, and AWS.

---

## 🚀 Key Features

- ✅ PyTorch-based deep learning modules (starting with image classification)
- ✅ Scikit-learn integration for SVM, RF, LR, NB, etc.
- ✅ Clean separation between data, models, features, and visualization
- ✅ YAML-based config system for all modules (easy to extend)
- ✅ Ready for AWS (S3, EC2, SageMaker, etc.)
- ✅ CI-ready with build status & testing badges
- ✅ Ideal for portfolio + job interview prep

---

## 🗂️ Project Structure
Lumyrix/
│
├── aws/ # AWS training & deployment scripts
├── config/ # All YAML configs (model, radiomics, viz, etc.)
├── data/ # Local data files (gitignored)
├── notebooks/ # Jupyter experiments
├── saved_models/ # Trained model checkpoints
├── src/ # All source code
│ ├── classical_ml/
│ ├── deep_learning/
│ ├── feature_analysis/
│ ├── feature_extraction/
│ ├── feature_selection/
│ ├── visualization/
│ └── main.py # Main training script
├── tests/ # (Coming soon) unit/integration tests
├── requirements.txt
├── LICENSE
└── README.md



---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/lumyrix.git
cd lumyrix

# Install dependencies
pip install -r requirements.txt

# Run the main training script
python src/main.py --config config/deep_learning/cifar10_baseline.yaml
