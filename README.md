# 🌿 LeafLens

**Detectare boli la plante prin inteligență artificială**

## Descriere

LeafLens este o aplicație desktop (Python + TensorFlow) care identifică specia de plantă și starea ei de sănătate pornind de la o fotografie a unei frunze. Aplicația oferă un nivel de încredere pentru fiecare predicție, un sfat agricol relevant pentru boala detectată și poate genera un raport PDF cu rezultatul complet al analizei. Interfața este disponibilă atât în română, cât și în engleză.

Proiect realizat de **Marina Teodor** și **Ignat Eduard**, elevi ai Colegiului Național „Nicolae Titulescu", pentru concursul **Infoeducația, faza națională**.

## Funcționalități

- Clasificare a bolilor la mai multe specii de plante — modelul acoperă 17 specii și 47 de clase (plantă + stare); vezi tabelul din secțiunea [Plante și boli suportate](#plante-și-boli-suportate)
- Filtru de frunză: verifică dacă imaginea încărcată chiar conține o frunză, înainte de a rula clasificarea principală
- Top 3 predicții, fiecare cu procentul de încredere aferent
- Sfaturi agricole personalizate pentru fiecare boală detectată
- Generare raport PDF cu imaginea analizată, rezultatul și sfatul agricol
- Interfață bilingvă (română / engleză), comutabilă din aplicație
- **Procesare mai multor poze deodată** — poți încărca o serie de fotografii; aplicația le pune într-o coadă, iar butoanele „◀ Inapoi" / „Urmator ▶" (sau săgețile stânga/dreapta de la tastatură) navighează între ele. Rezultatele fiecărei poze rămân în memorie (cache), deci nu se recalculează dacă te întorci la o poză deja analizată.
- **Alegere manuală a variantei pentru sfatul agricol** — dacă prima predicție din Top 3 nu se potrivește (ex. planta e corectă dar boala nu, sau invers), poți da click pe oricare dintre celelalte două carduri, iar sfatul agricol afișat se schimbă în funcție de varianta aleasă. Alegerea manuală este menționată și în raportul PDF generat.
- **Dialog „Informații" (🌐 din bara de sus)** — afișează, în interfață, lista completă a plantelor și bolilor suportate de model, utilă ca referință rapidă fără a mai căuta în cod.
- **Validare fișiere la încărcare** — dacă printre fișierele alese se strecoară unul care nu e o imagine, acesta este ignorat automat, cu un mesaj de avertizare, în loc să blocheze restul procesării.
- **Comenzi rapide de la tastatură**:
  - `F11` — comută modul ecran complet
  - `Esc` — iese din modul ecran complet
  - `←` / `→` — navigare înainte/înapoi prin coada de poze

## Plante și boli suportate

| Plantă | Boli detectate |
|---|---|
| Măr | Rapăn (scab), Putregai negru, Rugină de ienupăr, Sănătos |
| Afin | Sănătos |
| Cireș | Mană făinoasă, Sănătos |
| Porumb | Pată cercospora/pată cenușie, Rugină comună, Arsura nordică a frunzelor, Sănătos |
| Strugure | Putregai negru, Esca, Mana frunzelor, Sănătos |
| Portocal | Citrus greening (Huanglongbing) |
| Piersic | Pată bacteriană, Sănătos |
| Ardei gras | Pată bacteriană, Sănătos |
| Cartof | Mană timpurie, Mană târzie, Sănătos |
| Zmeur | Sănătos |
| Soia | Sănătos |
| Dovleac | Mană făinoasă |
| Căpșun | Arsura frunzelor, Sănătos |
| Roșie | Pată bacteriană, Mană timpurie, Mană târzie, Mucegaiul frunzelor, Pată Septoria, Păianjen roșu, Pată țintă, Virusul frunzelor gălbui răsucite, Virusul mozaicului, Sănătos |
| Grâu | Rugină brună, Mană, Septoria, Rugină galbenă, Sănătos |
| Floarea-soarelui | Mană, Mucegai cenușiu, Cicatrici pe frunze, Frunză proaspătă/sănătoasă |

*(Lista completă e vizibilă și în aplicație, în dialogul „Informații".)*

## Cerințe

- Python 3.13.3
- Librăriile din `requirements.txt`
- Formate de imagine acceptate: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.gif`, `.tif`, `.tiff`

> Notă: `tkinter` face parte din biblioteca standard Python. Pe Windows vine inclus implicit; pe unele distribuții Linux poate fi nevoie de instalarea separată a pachetului de sistem (ex: `sudo apt install python3-tk`).

> Compatibilitate testată: aplicația a fost testată complet pe Windows și parțial pe Linux; comportamentul pe macOS nu a fost verificat.

## Instalare

1. Clonează sau descarcă acest repository.

2. Instalează dependențele:

```
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

```
python leaflens.py
```

Apoi, în aplicație:

1. Apasă **Alege poza** și selectează una sau mai multe fotografii clare, de aproape, cu o frunză.
2. Vezi rezultatul (planta detectată, boala/starea de sănătate și sfatul agricol asociat) și navighează între poze cu „◀ Inapoi" / „Urmator ▶" dacă ai încărcat mai multe.
3. Opțional, apasă **Salvează PDF** pentru a exporta un raport al analizei. Raportul conține:
   - imaginea analizată
   - planta detectată și condiția (boală / sănătos)
   - procentul de încredere al modelului, plus un nivel calitativ (Foarte ridicat / Ridicat / Mediu / Scăzut)
   - sfatul agricol corespunzător variantei alese (implicit prima din Top 3, sau varianta selectată manual — menționat explicit dacă a fost o alegere manuală)
   - recomandări generale
   - celelalte 2 variante posibile din Top 3, pentru context
   - un disclaimer: raportul are caracter orientativ și nu înlocuiește diagnosticul unui specialist agronomic sau fitopatolog
   - paginare automată, cu antet/subsol și număr de pagină, dacă informația nu încape pe o singură pagină

## Date de antrenare

Modelul de clasificare a fost antrenat folosind o combinație a următoarelor seturi de date publice:

- Intel Image Classification
- New Plant Diseases Dataset
- PlantDoc Classification Dataset
- Sunflower Fruits and Leaves
- Wheat Disease Dataset

*(Notebook-ul folosit pentru antrenarea modelului este adăugat în repository.)*

Detaliu tehnic: modelul principal de clasificare pornește de la arhitectura **EfficientNetV2B0** (imagini de intrare 224×224). Filtrul „e frunză / nu e frunză" folosește un model separat, mai mic (MobileNetV2), cu imagini de intrare 160×160 (`leaf_detector.keras`).

## Autori

- **Marina Teodor**
- **Ignat Eduard**

Colegiul Național „Nicolae Titulescu" — proiect realizat pentru **Infoeducația, faza națională**.
