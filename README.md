# Tugas-Machine-Learning-1

## Table of Contents

- [General Information](#general-information)
- [Getting Started](#getting-started)
- [Project Status](#project-status)
- [Team Members](#team-members)
- [Program Structure](#program-structure)

## General Information

Tugas Besar 1 dalam mata kuliah IF3270 Pembelajaran Mesin bertujuan untuk melakukan segmentasi pelanggan berdasarkan karakteristik mereka menggunakan teknik klasifikasi. Dataset yang diberikan berisi informasi demografis dan perilaku pelanggan dengan berbagai fitur, seperti usia, pekerjaan, pengalaman kerja, tingkat pengeluaran, dan ukuran keluarga. Tugas ini mencakup eksplorasi data awal (EDA) untuk memahami pola yang terdapat dalam dataset serta pemodelan klasifikasi untuk mengelompokkan pelanggan ke dalam segmen tertentu. Model pembelajaran mesin yang digunakan diharapkan dapat memberikan wawasan yang bermanfaat bagi bisnis dalam menyesuaikan strategi pemasaran dan layanan terhadap kelompok pelanggan yang berbeda.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/FedrianzD/Tugas-Machine-Learning-1.git
cd Tugas-Machine-Learning-1
```

### 2. Install All Dependencies

```bash
pip install numpy matplotlib scikit-learn Digraph seaborn
```

### 3. Run the Program

```bash
CTRL + Enter on tes.ipynb
```

## Project Status

Project is complete

## Team Members

| **NIM**  |           **Nama**            | **Kontribusi**                                             |
| :------: | :---------------------------: | --------------------------------------------------------------------------------------------------------- |
| 13522020 | Aurelius Justin Philo Fanjaya | FFNN class, forward & back propagation, training, testing, autodiff                                       |
| 13522042 |         Amalia Putri          | Activation Function, Loss Function, melengkapi dokumen 50%                                                |
| 13522090 |        Fedrianz Dharma        | Visualisasi Weight Distribution, Visualisasi Weight Gradient Distribution, Laporan, Debug dan Fixing      |

## Program Structure

```
.
└── TUGAS-MACHINE-LEARNING-1
   ├── src
   │   ├── AutoDiff
   │   │   ├── AutoDiff.py
   │   │   ├── AutoDiffTest.py
   │   │   ├── FFNNAutoDiff.py
   │   │   └── Utils.py
   │   ├── models
   │   │   ├── linear.pkl
   │   │   ├── model.pkl
   │   │   ├── model2.pkl
   │   │   ├── model3.pkl
   │   │   ├── model4.pkl
   │   │   ├── model5.pkl
   │   │   ├── model6.pkl
   │   │   ├── model7.pkl
   │   │   ├── model8.pkl
   │   │   ├── model8batch.pkl
   │   │   ├── model_L1.pkl
   │   │   ├── model_L2.pkl
   │   │   ├── model_sklearn.pkl
   │   │   ├── model_sklearn_compare.pkl
   │   │   ├── model_uniform.pkl
   │   │   ├── model_zero.pkl
   │   │   ├── sigmoid.pkl
   │   │   └── tanh.pkl
   │   ├── FFNN.py
   │   ├── mnist_784_test.ipynb
   │   ├── simple_test.ipynb
   │   ├── training.ipynb
   │   └── visualizeTraining.ipynb
   └── README.md
```
