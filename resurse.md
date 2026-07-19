# Resurse externe
 
## Biblioteci
 
- TensorFlow / Keras (https://www.tensorflow.org/) — antrenare și rulare a modelelor de clasificare
- NumPy (https://numpy.org/) — procesare numerică a imaginilor
- Pillow (https://python-pillow.org/) — redimensionare imagini, generare preview, randare interfață
- ReportLab (https://www.reportlab.com/) — generare raport PDF
- Tkinter (https://docs.python.org/3/library/tkinter.html) — interfața grafică (biblioteca standard Python)
Versiunile exacte sunt în `requirements.txt`.
 
## Model pre-antrenat
 
- EfficientNetV2B0 — backbone-ul modelului principal de clasificare a bolilor, ales ca un bun compromis între acuratețe și viteză de antrenare pe Kaggle. Greutăți pre-antrenate pe ImageNet, folosite prin transfer learning. (Tan & Le, *EfficientNetV2: Smaller Models and Faster Training*, 2021 — https://arxiv.org/abs/2104.00298)
- MobileNetV2 — backbone-ul filtrului separat „este / nu este frunză". Greutăți pre-antrenate pe ImageNet, folosite prin transfer learning. (Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, 2018 — https://arxiv.org/abs/1801.04381)
## Seturi de date
 
- Intel Image Classification — https://www.kaggle.com/datasets/puneet6060/intel-image-classification
- New Plant Diseases Dataset — https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- PlantDoc Classification Dataset — https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset
- Sunflower Fruits and Leaves Dataset — https://www.kaggle.com/datasets/noamaanabdulazeem/sunflower-fruits-and-leaves-dataset
- Wheat Disease Dataset — https://www.kaggle.com/datasets/yasserhessein/wheat-disease-dataset-small
Detaliile de preprocesare și antrenare sunt în `leaf-lens-final-final.ipynb`.
