# LeafLens — Documentația proiectului
### Secțiunea IV.2 — Infoeducația, faza națională

**Autori:** Marina Teodor, Ignat Eduard
**Instituție:** Colegiul Național „Nicolae Titulescu"
**Repository:** https://github.com/Teodor5030/LeafLens

---

## 1. Descrierea problemei

Grădinarii amatori, fermierii mici și studenții la agricultură nu au la dispoziție o modalitate rapidă, gratuită și ușor de folosit pentru a identifica bolile unei plante direct la fața locului. Soluțiile existente au, în practică, mai multe limitări:

- **Dependența de internet** — majoritatea aplicațiilor funcționează doar online, iar în grădină, solar sau câmp semnalul lipsește adesea sau este instabil.
- **Limitări impuse de furnizorii de AI** — multe aplicații sunt, de fapt, o interfață pusă peste modele generale (ChatGPT, Claude etc.), care au limite de utilizare și costuri asociate.
- **Abonamente** — accesul complet este de multe ori condiționat de plata unei taxe recurente.
- **Conturi și date personale** — utilizarea necesită crearea unui cont, cu riscurile aferente legate de confidențialitatea și securitatea datelor.
- **Complexitate** — interfețele sunt greoaie pentru un utilizator care vrea doar un răspuns rapid la o singură întrebare: „ce are planta asta?"

Rezultatul este că problema rămâne, în mare parte, nerezolvată pentru utilizatorul obișnuit: grupurile de Facebook dedicate fermierilor amatori sunt în continuare pline de întrebări fără un răspuns clar, iar discuțiile de acolo se termină de multe ori în contradicții între oameni, fără o concluzie utilă.

## 2. Descrierea soluției propuse

LeafLens este o aplicație desktop care rulează **integral local**, fără conexiune la internet necesară după instalare, fără cont, fără abonament și fără transmiterea vreunei date către un server extern. Utilizatorul încarcă o fotografie a unei frunze, iar aplicația returnează în câteva secunde specia plantei, starea ei de sănătate și un sfat agricol asociat.

Soluția se bazează pe doi algoritmi de învățare automată, care lucrează în secvență:

1. Un model care verifică dacă imaginea încărcată conține efectiv o frunză, înainte de a trece mai departe.
2. Un model principal, care clasifică boala plantei și oferă un nivel de încredere pentru fiecare predicție posibilă.

Pe lângă diagnostic, aplicația poate genera automat un raport PDF cu fotografia analizată, rezultatul obținut și sfatul agricol corespunzător, util pentru păstrarea unui istoric sau pentru a fi arătat mai departe (de exemplu unui specialist).

## 3. Publicul țintă

- Grădinari amatori care cultivă plante în curte, solar sau grădină de legume.
- Fermieri mici și mijlocii, fără acces la servicii agricole specializate.
- Studenți și elevi la profil agricol, ca instrument de verificare rapidă în timpul studiului sau al practicii.
- Profesori care predau noțiuni de fitopatologie și pot folosi aplicația ca material demonstrativ.

Element comun al acestui public: are nevoie de un răspuns rapid, ieftin și fără bariere tehnice (cont, plată, semnal internet constant).

## 4. Funcționalitățile aplicației

- Clasificarea bolilor pentru mai multe specii de plante (măr, roșie, cartof, struguri, grâu, floarea-soarelui etc.).
- Filtru de frunză: verifică dacă fotografia încărcată conține efectiv o frunză, înainte de a rula clasificarea principală, reducând astfel rezultatele eronate.
- Afișarea a **top 3 predicții**, fiecare însoțită de procentul de încredere aferent, nu doar un singur răspuns „forțat".
- Sfaturi agricole personalizate, adaptate bolii detectate.
- Generarea unui raport PDF cu imaginea analizată, rezultatul complet și sfatul agricol.
- Interfață bilingvă (română / engleză), comutabilă direct din aplicație.
- Mod ecran complet (F11), pentru o utilizare confortabilă în timpul prezentărilor sau al lucrului pe teren.

## 5. Arhitectura aplicației

## 5.1. Modelele folosite
 
LeafLens combină două modele separate:
 
| Model | Rol | Arhitectură | Fișier |
|---|---|---|---|
| Clasificator principal | Identifică planta și boala/starea de sănătate | EfficientNetV2B0 (47 clase) | `best_model.keras` |
| Detector de frunză | Verifică dacă imaginea încărcată conține chiar o frunză, înainte de a rula clasificatorul principal | MobileNetV2 (clasificare binară) | `leaf_detector.keras` |
 
Ambele modele au fost antrenate în notebook-ul `leaf-lens-final-final.ipynb`, inclus în repository.
 
## 5.2. Performanța clasificatorului principal pe setul de validare
 
Raportul de clasificare (precizie / recall / f1-score) obținut pe setul de validare, pentru toate cele 47 de clase:
 
| Clasă | Precizie | Recall | F1-score | Nr. imagini |
|---|---|---|---|---|
| Apple___Apple_scab | 1.00 | 0.97 | 0.98 | 505 |
| Apple___Black_rot | 0.99 | 1.00 | 1.00 | 497 |
| Apple___Cedar_apple_rust | 1.00 | 1.00 | 1.00 | 441 |
| Apple___healthy | 0.97 | 1.00 | 0.98 | 502 |
| Blueberry___healthy | 1.00 | 1.00 | 1.00 | 454 |
| Cherry_(including_sour)___Powdery_mildew | 1.00 | 1.00 | 1.00 | 421 |
| Cherry_(including_sour)___healthy | 0.99 | 1.00 | 1.00 | 456 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.98 | 0.90 | 0.94 | 410 |
| Corn_(maize)___Common_rust_ | 1.00 | 0.99 | 0.99 | 478 |
| Corn_(maize)___Northern_Leaf_Blight | 0.92 | 0.98 | 0.95 | 478 |
| Corn_(maize)___healthy | 1.00 | 1.00 | 1.00 | 465 |
| Grape___Black_rot | 0.98 | 1.00 | 0.99 | 473 |
| Grape___Esca_(Black_Measles) | 1.00 | 0.98 | 0.99 | 480 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1.00 | 1.00 | 1.00 | 430 |
| Grape___healthy | 1.00 | 1.00 | 1.00 | 423 |
| Orange___Haunglongbing_(Citrus_greening) | 1.00 | 1.00 | 1.00 | 503 |
| Peach___Bacterial_spot | 0.98 | 1.00 | 0.99 | 459 |
| Peach___healthy | 1.00 | 0.99 | 0.99 | 432 |
| Pepper,_bell___Bacterial_spot | 1.00 | 1.00 | 1.00 | 478 |
| Pepper,_bell___healthy | 1.00 | 0.99 | 0.99 | 497 |
| Potato___Early_blight | 0.99 | 0.99 | 0.99 | 487 |
| Potato___Late_blight | 0.98 | 0.99 | 0.98 | 486 |
| Potato___healthy | 1.00 | 0.98 | 0.99 | 456 |
| Raspberry___healthy | 1.00 | 1.00 | 1.00 | 445 |
| Soybean___healthy | 0.98 | 1.00 | 0.99 | 505 |
| Squash___Powdery_mildew | 1.00 | 1.00 | 1.00 | 434 |
| Strawberry___Leaf_scorch | 1.00 | 1.00 | 1.00 | 444 |
| Strawberry___healthy | 1.00 | 1.00 | 1.00 | 456 |
| Sunflower___Downy_mildew | 0.81 | 0.94 | 0.87 | 18 |
| Sunflower___Gray_mold | 0.91 | 1.00 | 0.95 | 10 |
| Sunflower___Leaf_scars | 0.91 | 0.95 | 0.93 | 21 |
| Sunflower___healthy | 0.95 | 0.95 | 0.95 | 20 |
| Tomato___Bacterial_spot | 0.96 | 0.99 | 0.98 | 426 |
| Tomato___Early_blight | 0.96 | 0.91 | 0.93 | 481 |
| Tomato___Late_blight | 0.97 | 0.96 | 0.96 | 464 |
| Tomato___Leaf_Mold | 1.00 | 0.97 | 0.98 | 471 |
| Tomato___Septoria_leaf_spot | 0.96 | 0.94 | 0.95 | 437 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0.91 | 0.99 | 0.94 | 435 |
| Tomato___Target_Spot | 0.91 | 0.93 | 0.92 | 457 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 1.00 | 0.96 | 0.98 | 492 |
| Tomato___Tomato_mosaic_virus | 1.00 | 1.00 | 1.00 | 449 |
| Tomato___healthy | 0.98 | 0.98 | 0.98 | 481 |
| Wheat___BrownRust | 0.76 | 0.84 | 0.80 | 19 |
| Wheat___Healthy | 0.71 | 0.83 | 0.77 | 18 |
| Wheat___Mildew | 0.90 | 0.75 | 0.82 | 24 |
| Wheat___Septoria | 0.88 | 0.71 | 0.79 | 52 |
| Wheat___YellowRust | 0.66 | 0.83 | 0.73 | 35 |
 
**Per total (17.805 imagini de validare):**
 
| Metrică | Precizie | Recall | F1-score |
|---|---|---|---|
| Acuratețe generală | | | **0.98** |
| Medie macro (fiecare clasă contează la fel) | 0.95 | 0.96 | 0.96 |
| Medie ponderată (clasele mari contează mai mult) | 0.98 | 0.98 | 0.98 |
 
**Observație importantă:** clasele `Sunflower` și `Wheat` au un număr de imagini de validare mult mai mic (10–52 poze) față de restul claselor (410–505 poze), și scoruri de precizie/recall vizibil mai slabe (66%–95%, față de aproape 90–100% la toate celelalte clase). Cele două specii au fost adăugate ulterior, din seturile de date separate **Sunflower Fruits and Leaves** și **Wheat Disease Dataset**, cu mult mai puține poze disponibile decât restul claselor din New Plant Diseases Dataset. Acest dezechilibru de date explică parțial bias-ul observat mai jos, unde poze de roșii au fost clasificate greșit ca Sunflower.
 
## 5.3. Detectorul binar „frunză / nu frunză"
 
**Scop:** filtrează pozele care nu conțin o frunză (de exemplu poze cu peisaje, obiecte, animale), înainte ca aplicația să ruleze clasificatorul principal de boli. Fără acest filtru, modelul principal ar încerca să încadreze orice poză într-una din cele 47 de clase de boli, chiar dacă poza nu are legătură cu o frunză.
 
**Date de antrenare:**
- Clasa `frunza`: 1880 de poze, adunate din primele 40 de poze din fiecare clasă a setului de antrenare al clasificatorului principal.
- Clasa `nu_frunza`: 1800 de poze din setul public **Intel Image Classification** (categorii generale: clădiri, munte, mare, stradă etc.).
- Împărțire train/valid: 3128 poze antrenare / 552 poze validare (~85%/15%).
**Arhitectură:**
- Bază: `MobileNetV2` preantrenat pe ImageNet, greutăți înghețate (`trainable = False`), input 160×160.
- Strat `Dense(64, activation="relu")`
- `Dropout(0.3)`
- Strat de ieșire: `Dense(1, activation="sigmoid")`
- Optimizator: Adam, learning rate 1e-3, loss `binary_crossentropy`.
- Early stopping: `patience=3`, monitorizat pe `val_accuracy`, cu restaurarea celor mai bune greutăți.
**Rezultat antrenare (8 epoci):**
 
| Epocă | Acuratețe antrenare | Acuratețe validare |
|---|---|---|
| 1 | 98.3% | 100% |
| 2 | 99.9% | 100% |
| 3 | 100% | 100% |
| 4 | 100% | 100% |
 
Modelul a convers foarte rapid la ~100% acuratețe de validare, ceea ce e de așteptat: distincția „frunză vs. peisaj/clădire/obiect" este mult mai simplă decât clasificarea bolilor.
 
**Convenție de interpretare a scorului:** valoare sub 0.5 → poza este o frunză; valoare peste 0.5 → poza nu este o frunză.
 
## 5.4. Evaluarea clasificatorului principal pe poze reale
 
Pentru a testa performanța reală a aplicației (nu doar acuratețea de validare din timpul antrenării), modelul a fost testat pe **143 de poze reale, niciodată văzute de model**, acoperind 14 clase verificate.
 
| Metrică | Valoare |
|---|---|
| Acuratețe top-1 (răspunsul corect e primul afișat) | **68.5%** |
| Acuratețe top-3 (răspunsul corect e printre primele 3) | **90.9%** |
 
Diferența mare dintre top-1 și top-3 arată că, în majoritatea cazurilor în care modelul greșește prima predicție, răspunsul corect este totuși "aproape" — motiv pentru care aplicația afișează întotdeauna top 3 predicții, nu doar prima.
 
### Acuratețe pe fiecare clasă (top-1)
 
| Clasă | Rezultat |
|---|---|
| Apple___Apple_scab | 9/10 (90%) |
| Apple___Cedar_apple_rust | 9/10 (90%) |
| Corn_(maize)___Common_rust_ | 7/10 (70%) |
| Corn_(maize)___Northern_Leaf_Blight | 10/12 (83%) |
| Grape___Black_rot | 5/8 (62%) |
| Potato___Early_blight | 6/14 (43%) |
| Potato___Late_blight | 8/8 (100%) |
| Tomato___Bacterial_spot | 2/9 (22%) |
| Tomato___Early_blight | 6/9 (67%) |
| Tomato___Late_blight | 9/10 (90%) |
| Tomato___Leaf_Mold | 4/6 (67%) |
| Tomato___Septoria_leaf_spot | 9/12 (75%) |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 11/15 (73%) |
| Tomato___Tomato_mosaic_virus | 3/10 (30%) |
 
### Matricea de confuzie – observații
 
Cele mai frecvente confuzii (cel puțin 2 cazuri), obținute din matricea de confuzie generată pe pozele de test:
 
| Nr. cazuri | Planta reală | Model a prezis |
|---|---|---|
| 5× | Potato___Early_blight | Potato___Late_blight |
| 3× | Tomato___Tomato_mosaic_virus | Tomato___Late_blight |
| 3× | Tomato___Tomato_mosaic_virus | Sunflower___Downy_mildew |
| 3× | Tomato___Bacterial_spot | Pepper,_bell___Bacterial_spot |
| 3× | Corn_(maize)___Common_rust_ | Corn_(maize)___Northern_Leaf_Blight |
| 2× | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato___Late_blight |
| 2× | Tomato___Early_blight | Sunflower___Leaf_scars |
| 2× | Tomato___Bacterial_spot | Tomato___Septoria_leaf_spot |
 
**Ce arată aceste confuzii:**
 
- **Potato Early blight ↔ Late blight** – cea mai frecventă confuzie; cele două boli au simptome vizuale asemănătoare pe frunza de cartof (pete brune, necroză).
- **Tomato mosaic virus** este cea mai slabă clasă (30%): se confundă atât cu Late blight, cât și cu clasele Sunflower – posibil semn că setul de antrenare pentru mozaicul de tomate nu acoperă suficient variația reală a bolii, sau că modelul are un ușor bias spre clasele Sunflower/Wheat, adăugate ulterior în setul de date.
- **Tomato Bacterial spot** (22%, cea mai slabă clasă) se confundă cu Pepper bell Bacterial spot (aceeași boală, plantă diferită) și cu Septoria leaf spot – ambele apar ca pete mici, întunecate pe frunză.
- **Corn Common rust ↔ Northern Leaf Blight** – confuzie așteptată, sunt cele mai frecvent confundate boli de porumb în literatura de specialitate.
## 5.5. Concluzii / limitări cunoscute
 
- Pe setul de validare, modelul are 98% acuratețe generală — foarte ridicată — dar acest lucru nu se traduce direct în performanța pe poze reale (68.5% top-1), semn tipic de supra-adaptare la caracteristicile seturilor de date publice folosite la antrenare.
- Modelul funcționează foarte bine, atât în validare cât și pe poze reale, pe clase cu simptome vizuale distincte (Potato Late blight 100%, Apple scab/Cedar apple rust 90%).
- Cele mai slabe clase pe poze reale sunt cele cu simptome asemănătoare altor boli sau altor specii (Tomato Bacterial spot, Tomato mosaic virus, Potato Early blight) — pentru acestea, afișarea top-3 predicții din aplicație compensează parțial acuratețea mai mică de top-1.
- Confuziile cu clasele Sunflower/Wheat rămân un punct de urmărit față de versiunile anterioare ale modelului; ele au și cel mai mic număr de poze de antrenare/validare dintre toate clasele, ceea ce poate explica atât scorurile mai slabe pe aceste clase, cât și tendința modelului de a le folosi ca răspuns greșit pentru alte boli.
Ambele modele au fost antrenate în notebook-uri Kaggle, care oferă acces gratuit la GPU (T4x2), eliminând nevoia unui hardware dedicat pentru antrenare.

Rezultatele modelului principal (etichetele claselor) sunt mapate printr-un fișier `label_map.json`, iar informațiile despre boli, plante și sfaturile agricole (în ambele limbi) sunt stocate în fișiere JSON separate, în folderul `data/` (`boli.json`, `plante.json`, `sfaturi_agricole.json`, `sfaturi_generale.json`).

**Fluxul aplicației:**
utilizatorul alege o fotografie → modelul de detectare a frunzei validează imaginea → dacă este validă, modelul principal clasifică boala → aplicația afișează top 3 predicții cu procentele de încredere și sfatul agricol asociat → opțional, utilizatorul exportă rezultatul ca PDF.

Seturile de date publice folosite pentru antrenare:
- New Plant Diseases Dataset
- PlantDoc Classification Dataset
- Sunflower Fruits and Leaves
- Wheat Disease Dataset
- Intel Image Classification

## 6. Elemente distinctive / puncte forte în comparație cu competiția

- **Funcționează complet offline** — spre deosebire de majoritatea aplicațiilor din categorie, care depind de o conexiune la internet și nu funcționează în grădină sau câmp acolo unde semnalul este slab sau absent.
- **Fără limitele impuse de un AI extern** — aplicația nu depinde de un API precum ChatGPT sau Claude, deci nu este afectată de limite de utilizare, costuri per cerere sau întreruperi ale serviciului.
- **Complet gratuită, fără abonament** — nu există niciun cost recurent pentru utilizator.
- **Fără cont și fără colectare de date** — nimic din ce fotografiază utilizatorul nu părăsește dispozitivul, ceea ce elimină riscul de scurgere a datelor.
- **Validare în doi pași** — filtrul de frunză reduce riscul unor rezultate absurde (de exemplu, clasificarea unei fotografii care nu conține o frunză), spre deosebire de aplicațiile care trimit orice imagine direct către un singur model general.
- **Interfață simplă, bilingvă**, gândită pentru a ajunge rapid la un rezultat, fără pași inutili.

## 7. Ghid de instalare și configurare a aplicației

**Cerințe:**
- Python 3.13.3
- Bibliotecile listate în `requirements.txt`
- `tkinter` (parte din biblioteca standard Python; pe Windows este inclus implicit, pe unele distribuții Linux poate fi necesară instalarea separată: `sudo apt install python3-tk`)

**Pași de instalare:**

1. Se clonează sau se descarcă repository-ul proiectului.
2. Se instalează dependențele:
   ```
   pip install -r requirements.txt
   ```
3. Se descarcă modelul principal de clasificare, `best_model.keras` (fișier prea mare pentru a fi inclus direct în repository), de la linkul indicat în README, și se plasează în folderul principal al aplicației, alături de `leaflens.py`. Dacă browserul salvează fișierul sub alt nume, acesta trebuie redenumit manual în `best_model.keras`.

**Structura fișierelor:**
```
LeafLens/
├── leaflens.py
├── best_model.keras         (descărcat separat)
├── leaf_detector.keras
├── label_map.json
├── requirements.txt
└── data/
    ├── boli.json
    ├── plante.json
    ├── sfaturi_agricole.json
    └── sfaturi_generale.json
```

**Utilizare:**

Din folderul principal, se rulează:
```
python leaflens.py
```

În aplicație:
1. Se apasă **Alege poza** și se selectează o fotografie clară, de aproape, cu o frunză.
2. Se vizualizează rezultatul (planta detectată, boala/starea de sănătate și sfatul agricol asociat).
3. Opțional, se apasă **Salvează PDF** pentru a exporta un raport al analizei.

## 8. Justificarea folosirii tehnologiilor alese

Python a fost ales pentru că este, la momentul actual, cel mai potrivit mediu pentru dezvoltarea de aplicații bazate pe inteligență artificială, având cel mai bogat ecosistem de biblioteci (TensorFlow, Keras) și cea mai vastă documentație disponibilă. Pentru antrenarea modelelor am folosit notebook-uri Kaggle, care oferă acces gratuit la GPU (T4x2) — o resursă esențială pentru un proiect de elevi, fără buget pentru infrastructură de calcul. Arhitectura EfficientNetV2B0 a fost aleasă pentru modelul principal deoarece oferă un echilibru bun între acuratețe și dimensiune, potrivit pentru transfer learning pe seturi de date de mărime limitată. Pentru modelul de detectare a frunzei am ales MobileNetV2, o arhitectură mult mai mică și mai rapidă, potrivită pentru o sarcină simplă (binară) care trebuie să ruleze de fiecare dată, înaintea clasificării principale, fără să introducă o întârziere vizibilă. În fine, interfața a fost realizată în Tkinter, biblioteca grafică inclusă implicit în Python, tocmai pentru a păstra instalarea aplicației cât mai simplă, fără dependențe suplimentare și fără diferențe de comportament între sisteme de operare. Împreună, aceste alegeri permit ca întregul flux — de la încărcarea imaginii până la diagnostic — să ruleze local, pe un calculator obișnuit, fără costuri și fără conexiune la internet.

## 9. Opinia autorilor despre idee și utilitatea aplicației

Ideea proiectului a pornit de la o discuție cu un coleg care se pregătea pentru admiterea la medicină, despre identificarea bolilor la oameni cu ajutorul inteligenței artificiale. Analizând subiectul, am realizat că diagnosticarea persoanelor este un domeniu mult mai greu de abordat: corpul uman este mult mai complex de „citit" vizual, există deja specialiști foarte bine pregătiți (medicii) și proiecte de cercetare de milioane de euro dezvoltate exact în acest domeniu — un proiect de liceu nu ar fi adus, realist, valoare într-un spațiu deja atât de dezvoltat. Frunzele plantelor, în schimb, sunt mult mai ușor de analizat vizual, iar problema identificării bolilor la plante, la nivelul utilizatorului obișnuit, este încă departe de a fi rezolvată: grupurile de Facebook dedicate fermierilor amatori sunt și acum pline de întrebări fără un răspuns clar, iar comentariile se transformă frecvent în contradicții între oameni, fără să ajungă la o concluzie utilă. În același timp, soluțiile care există sunt greu de folosit sau sunt, în esență, o interfață pusă peste ChatGPT, Claude sau alte modele generale, cu scopul principal de a genera venit prin abonamente, nu de a rezolva cu adevărat accesul rapid și gratuit la un diagnostic. Considerăm că LeafLens este util tocmai pentru că elimină aceste bariere. De exemplu, un grădinar amator care observă pete ciudate pe frunzele roșiilor din solar, fără semnal la internet, poate fotografia frunza și primi pe loc un diagnostic cu recomandare de tratament, în loc să posteze poza într-un grup de Facebook și să aștepte răspunsuri incerte și contradictorii de la alți amatori, sau să plătească un abonament pentru o aplicație care oricum nu ar funcționa fără conexiune.
