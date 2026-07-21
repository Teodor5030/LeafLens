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

Aplicația este scrisă în **Python**, cu interfața grafică realizată în **Tkinter**, biblioteca standard inclusă în Python — aceasta a fost o alegere deliberată pentru a păstra instalarea simplă, fără dependențe grafice suplimentare.

Procesarea imaginii se face în doi pași, folosind două modele separate, antrenate cu **TensorFlow/Keras**:

1. **Modelul de detectare a frunzei** (`leaf_detector.keras`) — construit pe arhitectura **MobileNetV2**, un model mic și rapid, folosit ca filtru: verifică binar dacă imaginea conține o frunză, înainte ca aceasta să ajungă la modelul principal.
2. **Modelul principal de clasificare** (`best_model.keras`) — construit pe arhitectura **EfficientNetV2B0**, antrenat prin transfer learning pe o combinație de seturi de date publice, pentru a recunoaște boala și specia plantei.

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
