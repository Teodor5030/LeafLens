# 🌿 LeafLens
**Detectare boli la plante prin inteligență artificială**
## Descriere
LeafLens este o aplicație desktop (Python + TensorFlow) care identifică specia de plantă și starea ei de sănătate pornind de la o fotografie a unei frunze. Aplicația oferă un nivel de încredere pentru fiecare predicție, un sfat agricol relevant pentru boala detectată și poate genera un raport PDF cu rezultatul complet al analizei. Interfața este disponibilă atât în română, cât și în engleză.
Proiect realizat de **Marina Teodor** și **Ignat Eduard**, elevi ai Colegiului Național „Nicolae Titulescu", pentru concursul **Infoeducația, faza națională**.
## Public țintă
Se adresează în primul rând fermierilor mici, grădinarilor amatori și elevilor/studenților de la profil agricol — cuiva care are o poză cu o frunză și vrea un răspuns rapid, fără cont, fără abonament și fără conexiune la internet.
## Funcționalități
- Clasificare a bolilor la mai multe specii de plante (măr, roșie, cartof, struguri, grâu, floarea-soarelui etc.)
- Filtru de frunză: verifică dacă imaginea încărcată chiar conține o frunză, înainte de a rula clasificarea principală
- Top 3 predicții, fiecare cu procentul de încredere aferent
- Sfaturi agricole personalizate pentru fiecare boală detectată
- Generare raport PDF cu imaginea analizată, rezultatul și sfatul agricol
- Interfață bilingvă (română / engleză), comutabilă din aplicație
- Mod ecran complet (F11)
## Comparație cu alte aplicații
Există deja aplicații cunoscute pe zona asta — Plantix, PlantNet, Plant.id și altele asemănătoare. Toate rulează în cloud (trimit poza pe un server), iar PlantNet e axat pe identificarea speciei, nu pe boli. Niciuna nu oferă un raport exportabil cu rezultatul analizei.
LeafLens rulează complet local, o dată instalat: nicio poză nu iese de pe calculator. E gratuit, generează un raport PDF cu toate cele trei variante posibile și sfaturile aferente, și codul e public pe GitHub. În schimb, acoperă mai puține specii (în jur de 15) față de aplicațiile comerciale mari 
## Cerințe
- Python 3.13.3
- Librăriile din `requirements.txt`
> Notă: `tkinter` face parte din biblioteca standard Python. Pe Windows vine inclus implicit; pe unele distribuții Linux poate fi nevoie de instalarea separată a pachetului de sistem (ex: `sudo apt install python3-tk`).
## Instalare
1. Clonează sau descarcă acest repository.
2. Instalează dependențele:
   ```bash
   pip install -r requirements.txt
   ```
3. Descarcă modelul principal de clasificare (fișier prea mare pentru a fi inclus direct în repository):
   👉 [best_model.keras — Google Drive](https://drive.google.com/file/d/1Jkqvv-DVrSzrYL4nrGWDQKO-B44apfmh/view?usp=drive_link)
   Pune fișierul descărcat, cu numele **`best_model.keras`**, direct în folderul principal al aplicației, alături de `leaflens.py`. Dacă browserul salvează fișierul cu alt nume, redenumește-l manual în `best_model.keras`.
## Structura fișierelor
```
LeafLens/
├── leaflens.py
├── best_model.keras         (descărcat separat — vezi „Instalare")
├── leaf_detector.keras
├── label_map.json
├── requirements.txt
└── data/
    ├── boli.json
    ├── plante.json
    ├── sfaturi_agricole.json
    └── sfaturi_generale.json
```
## Utilizare
Din folderul principal, rulează:
```bash
python leaflens.py
```
Apoi, în aplicație:
1. Apasă **Alege poza** și selectează o fotografie clară, de aproape, cu o frunză.
2. Vezi rezultatul (planta detectată, boala/starea de sănătate și sfatul agricol asociat).
3. Opțional, apasă **Salvează PDF** pentru a exporta un raport al analizei.
## Date de antrenare
Modelul principal de clasificare a bolilor folosește EfficientNetV2B0 ca backbone, prin transfer learning cu greutăți pre-antrenate pe ImageNet — un bun compromis între acuratețe și viteză de antrenare pe Kaggle. Filtrul separat „este / nu este frunză" folosește MobileNetV2, tot prin transfer learning.
Modelul principal a fost antrenat folosind o combinație a următoarelor seturi de date publice:
- Intel Image Classification
- New Plant Diseases Dataset
- PlantDoc Classification Dataset
- Sunflower Fruits and Leaves
- Wheat Disease Dataset
*(Notebook-ul folosit pentru antrenarea modelului este adăugat în repository.)*
## Plan de dezvoltare
- Extinderea numărului de specii și boli acoperite
- Variantă mobilă (Android), pentru fotografiere directă în câmp
- Istoric al analizelor anterioare, ca să se poată urmări evoluția unei plante în timp
- Feedback de la utilizatori reali pentru rafinarea sfaturilor agricole
## Autori
- **Marina Teodor**
- **Ignat Eduard**
Colegiul Național „Nicolae Titulescu" — proiect realizat pentru **Infoeducația, faza națională**.
