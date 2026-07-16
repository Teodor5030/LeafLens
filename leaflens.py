# LeafLens - aplicatia mea de detectie boli la plante din poze
# scuze in avans daca gasiti bucati de cod care ar putea fi scrise mai elegant,
# Notes:
# - modelele .keras (best_model, leaf_detector) trebuie sa fie langa exe/script, altfel pica la incarcare
# - am testat doar pe Windows si un pic pe Linux, pe Mac nu stiu ce face

import os

# liniile astea 2 TREBUIE sa fie inainte de "import tensorflow", altfel nu au niciun efect
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import tempfile
import json
import sys
import logging
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageDraw, ImageTk
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)  # scapa de spam-ul de INFO/WARNING pe care tf il arunca in consola

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas

TITLU_FEREASTRA = "LeafLens"
EXT_IMAGINI = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
DIM_INTRARE = 224
DIM_INTRARE_FRUNZA = 160
NUMAR_PREDICTII = 3

# paleta de culori pt toata interfata (dark green theme, imi place mai mult decat alb standard)
# le-am ales cu ajutorul lui ChatGPT (foarte bun pe tematice)
culori = {
    "fundal": "#0F1F15",
    "suprafata": "#162210",
    "card": "#1C2E1A",
    "chenar": "#2A4A2A",
    "verde": "#4ADE80",
    "verde2": "#22C55E",
    "btn_verde": "#16A34A",
    "btn_hover": "#15803D",
    "btn_gri": "#1F2937",
    "text": "#F0FDF4",
    "text_slab": "#6B7280",
    "text_dim": "#9CA3AF",
    "ok_fond": "#14532D",
    "ok_text": "#86EFAC",
    "rau_fond": "#7F1D1D",
    "rau_text": "#FCA5A5",
    "incarcare_f": "#78350F",
    "incarcare_t": "#FCD34D",
    "nefrunza_fond": "#78350F",
    "nefrunza_text": "#FCD34D",
    "card_evident": "#243D22",
}

# index model si eticheta clasa (format "Planta___Boala")
# !ATENTIE!: ordinea/indexii astia trebuie sa corespunda EXACT cu ordinea claselor din antrenare!
boli = {
    0: "Apple___Apple_scab", 1: "Apple___Black_rot", 2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy", 4: "Blueberry___healthy",
    5: "Cherry_(including_sour)___Powdery_mildew", 6: "Cherry_(including_sour)___healthy",
    7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 8: "Corn_(maize)___Common_rust_",
    9: "Corn_(maize)___Northern_Leaf_Blight", 10: "Corn_(maize)___healthy",
    11: "Grape___Black_rot", 12: "Grape___Esca_(Black_Measles)",
    13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 14: "Grape___healthy",
    15: "Orange___Haunglongbing_(Citrus_greening)", 16: "Peach___Bacterial_spot",
    17: "Peach___healthy", 18: "Pepper,_bell___Bacterial_spot", 19: "Pepper,_bell___healthy",
    20: "Potato___Early_blight", 21: "Potato___Late_blight", 22: "Potato___healthy",
    23: "Raspberry___healthy", 24: "Soybean___healthy", 25: "Squash___Powdery_mildew",
    26: "Strawberry___Leaf_scorch", 27: "Strawberry___healthy", 28: "Tomato___Bacterial_spot",
    29: "Tomato___Early_blight", 30: "Tomato___Late_blight", 31: "Tomato___Leaf_Mold",
    32: "Tomato___Septoria_leaf_spot", 33: "Tomato___Spider_mites Two-spotted_spider_mite",
    34: "Tomato___Target_Spot", 35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    36: "Tomato___Tomato_mosaic_virus", 37: "Tomato___healthy",
    38: "Wheat___BrownRust", 39: "Wheat___Healthy", 40: "Wheat___Mildew",
    41: "Wheat___Septoria", 42: "Wheat___YellowRust",
    43: "Sunflower___Downy_mildew", 44: "Sunflower___Fresh_Leaf",
    45: "Sunflower___Gray_mold", 46: "Sunflower___Leaf_scars",
}

# datele (nume plante/boli, sfaturi) le tin in fisiere json separate in folderul data/
# ideea a fost sa putem edita textele fara sa umblu prin cod de fiecare data
NUME_PLANTE = {}
NUME_BOLI = {}
sfaturi = {}
SFATURI_GENERALE = {}


def incarca_date_json():
    """Incarca datele din fisierele JSON externe."""
    global NUME_PLANTE, NUME_BOLI, sfaturi, SFATURI_GENERALE

    # Determina folderul aplicatiei
    if getattr(sys, "frozen", False):
        folder = Path(sys.executable).parent
    else:
        folder = Path(__file__).parent

    folder_data = folder / "data"

    # Incarca plante
    fisier_plante = folder_data / "plante.json"
    if fisier_plante.exists():
        with open(fisier_plante, 'r', encoding='utf-8') as f:
            NUME_PLANTE = json.load(f)
    else:
        print(f"Warning: {fisier_plante} not found, using empty plant names")

    # Incarca boli
    fisier_boli = folder_data / "boli.json"
    if fisier_boli.exists():
        with open(fisier_boli, 'r', encoding='utf-8') as f:
            NUME_BOLI = json.load(f)
    else:
        print(f"Warning: {fisier_boli} not found, using empty disease names")

    # Incarca sfaturi agricole
    fisier_sfaturi = folder_data / "sfaturi_agricole.json"
    if fisier_sfaturi.exists():
        with open(fisier_sfaturi, 'r', encoding='utf-8') as f:
            sfaturi = json.load(f)
    else:
        print(f"Warning: {fisier_sfaturi} not found, using empty agricultural tips")

    # Incarca sfaturi generale
    fisier_sfaturi_generale = folder_data / "sfaturi_generale.json"
    if fisier_sfaturi_generale.exists():
        with open(fisier_sfaturi_generale, 'r', encoding='utf-8') as f:
            SFATURI_GENERALE = json.load(f)
    else:
        print(f"Warning: {fisier_sfaturi_generale} not found, using empty general tips")


# Incarca datele la pornirea aplicatiei
incarca_date_json()

# toate textele din interfata, ro + en, ca sa nu am stringuri hardcodate peste tot
# (aproape peste tot, mai exista cate un emoji sau text scris direct undeva, nu m-am tinut 100% de regula asta)
TEXTE = {
    "ro": {
        "subtitlu": "Detectare boli la plante prin inteligenta artificiala",
        "fullscreen": "Ecran complet (F11)",
        "buton_limba": "EN",
        "incarcare_modele": "Se incarca modelele, asteptati...",
        "model_negasit": "Nu gasesc modelul: {cale}",
        "gata": "Gata! Alege poze sau trage-le aici.",
        "sectiune_imagine": "IMAGINE ANALIZATA",
        "hint_poza": "Apasa 'Alege poze' pentru a incarca imagini",
        "btn_alege": "Alege poze",
        "btn_analizeaza": "Analizeaza",
        "btn_anterior": "◀  Inapoi",
        "btn_urmator": "Urmator  ▶",
        "coada_pozitie": "Poza {curenta} / {total}",
        "coada_goala": "Coada goala",
        "poze_in_coada": "{n} poze in coada",
        "btn_pdf": "Salveaza PDF",
        "btn_sterge": "Sterge tot",
        "dialog_titlu_poze": "Alege fotografii cu frunze",
        "asteapta_scanare": "ASTEAPTA SCANARE",
        "sectiune_rezultate": "TOP 3 REZULTATE",
        "card_loc_1": "#1",
        "card_loc_2": "#2",
        "card_loc_3": "#3",
        "camp_planta": "Planta",
        "camp_conditie": "Conditie",
        "camp_incredere": "Incredere",
        "sfat_titlu": "SFAT AGRICOL",
        "sfat_pentru": "Pentru varianta {loc}: {planta} - {boala}",
        "apasa_pt_alt_sfat": "Apasa pe o alta varianta de mai sus pentru a vedea tratamentul corespunzator, daca prima nu se potriveste.",
        "sfat_necunoscut": "Consultati un specialist agronomic pentru mai multe detalii.",
        "tip_implicit": "Incarca o imagine si apasa Analizeaza.",
        "planta_sanatoasa": "SANATOS",
        "boala_detectata": "BOALA",
        "sanatos": "Sanatos",
        "nu_e_frunza_tag": "NU E FRUNZA",
        "nu_e_frunza_titlu": "Poza nu contine o frunza",
        "nu_e_frunza_tip": "Incarca o poza clara, de aproape, cu o frunza de planta, ca sa poata fi analizata.",
        "dialog_titlu_pdf": "Unde salvam raportul?",
        "fisier_invalid": "Fisier ignorat (nu e imagine): {nume}",
        "atentie": "Atentie",
        "fara_poza": "Nu ai ales nicio poza!",
        "salvat_titlu": "Salvat!",
        "salvat_mesaj": "Raportul a fost salvat la:\n{cale}",
        "pdf_titlu": "LeafLens",
        "pdf_subtitlu": "Raport de analiza planta",
        "pdf_sectiune_imagine": "IMAGINE ANALIZATA:",
        "pdf_detalii_titlu": "Detalii analiza",
        "pdf_planta": "Planta detectata:",
        "pdf_conditie": "Conditie:",
        "pdf_incredere": "Incredere model:",
        "pdf_nivel_incredere": "Nivel de incredere:",
        "pdf_nivel_foarte_ridicata": "Foarte ridicat",
        "pdf_nivel_ridicata": "Ridicat",
        "pdf_nivel_medie": "Mediu",
        "pdf_nivel_scazuta": "Scazut",
        "pdf_varianta_selectata": "Rezultat ales manual din Top 3: varianta {loc}",
        "pdf_sfat": "SFAT AGRICOL",
        "pdf_recomandari_generale": "Recomandari generale",
        "pdf_alte_titlu": "Alte posibilitati din Top 3",
        "pdf_pagina": "Pagina",
        "pdf_disclaimer": "Acest raport este generat automat de o aplicatie de inteligenta artificiala si are caracter orientativ. Nu inlocuieste diagnosticul unui specialist agronomic sau fitopatolog.",
        "btn_informatii": "Informatii",
        "dialog_titlu_informatii": "Plante si boli suportate",
        "sectiune_plante": "PLANTE SUPORTATE",
        "sectiune_boli": "BOLI SUPORTATE",
    },
    "en": {
        "subtitlu": "AI-powered plant disease detection",
        "fullscreen": "Fullscreen (F11)",
        "buton_limba": "RO",
        "incarcare_modele": "Loading models, please wait...",
        "model_negasit": "Model not found: {cale}",
        "gata": "Ready! Choose photos or drag them here.",
        "sectiune_imagine": "ANALYZED IMAGE",
        "hint_poza": "Press 'Choose Photos' to load images",
        "btn_alege": "Choose Photos",
        "btn_analizeaza": "Analyze",
        "btn_anterior": "◀  Prev",
        "btn_urmator": "Next  ▶",
        "coada_pozitie": "Photo {curenta} / {total}",
        "coada_goala": "Queue empty",
        "poze_in_coada": "{n} photos in queue",
        "btn_pdf": "Save PDF",
        "btn_sterge": "Clear all",
        "dialog_titlu_poze": "Choose leaf photos",
        "asteapta_scanare": "AWAITING SCAN",
        "sectiune_rezultate": "TOP 3 RESULTS",
        "card_loc_1": "#1",
        "card_loc_2": "#2",
        "card_loc_3": "#3",
        "camp_planta": "Plant",
        "camp_conditie": "Condition",
        "camp_incredere": "Confidence",
        "sfat_titlu": "FARMING TIP",
        "sfat_pentru": "For result {loc}: {planta} - {boala}",
        "apasa_pt_alt_sfat": "Tap another result above to see its matching treatment, in case the first one doesn't fit.",
        "sfat_necunoscut": "Consult an agricultural specialist for more details.",
        "tip_implicit": "Load un image and press Analyze.",
        "planta_sanatoasa": "HEALTHY",
        "boala_detectata": "DISEASE",
        "sanatos": "Healthy",
        "nu_e_frunza_tag": "NOT A LEAF",
        "nu_e_frunza_titlu": "Photo doesn't contain a leaf",
        "nu_e_frunza_tip": "Load a clear, close-up photo of a plant leaf so it can be analyzed.",
        "dialog_titlu_pdf": "Where should the report be saved?",
        "fisier_invalid": "Skipped (not an image): {nume}",
        "atentie": "Warning",
        "fara_poza": "No photo selected!",
        "salvat_titlu": "Saved!",
        "salvat_mesaj": "The report was saved to:\n{cale}",
        "pdf_titlu": "LeafLens",
        "pdf_subtitlu": "Plant Analysis Report",
        "pdf_sectiune_imagine": "ANALYZED IMAGE:",
        "pdf_detalii_titlu": "Analysis details",
        "pdf_planta": "Detected plant:",
        "pdf_conditie": "Condition:",
        "pdf_incredere": "Model confidence:",
        "pdf_nivel_incredere": "Confidence level:",
        "pdf_nivel_foarte_ridicata": "Very high",
        "pdf_nivel_ridicata": "High",
        "pdf_nivel_medie": "Medium",
        "pdf_nivel_scazuta": "Low",
        "pdf_varianta_selectata": "Manually selected result from Top 3: option {loc}",
        "pdf_sfat": "FARMING TIP",
        "pdf_recomandari_generale": "General recommendations",
        "pdf_alte_titlu": "Other possibilities from Top 3",
        "pdf_pagina": "Page",
        "pdf_disclaimer": "This report is automatically generated by an AI application and is for guidance only. It does not replace the diagnosis of an agricultural specialist or plant pathologist.",
        "btn_informatii": "Info",
        "dialog_titlu_informatii": "Supported Plants and Diseases",
        "sectiune_plante": "SUPPORTED PLANTS",
        "sectiune_boli": "SUPPORTED DISEASES",
    },
}


# ---- helper functions / functii ajutatoare, folosite peste tot in restul fisierului ----

def este_sanatos(eticheta_bruta):
    e = eticheta_bruta.lower()
    return "healthy" in e or "fresh_leaf" in e or "fresh leaf" in e


def desparte_eticheta(eticheta_bruta):
    parti = eticheta_bruta.split("___")
    planta_raw = parti[0].strip()
    boala_raw = parti[1].strip() if len(parti) > 1 else ""
    return planta_raw, boala_raw


def traduce_planta(planta_raw, limba):
    if planta_raw in NUME_PLANTE:
        return NUME_PLANTE[planta_raw][limba]
    return planta_raw.replace("_", " ").replace(",", "")


def traduce_boala(boala_raw, limba):
    if boala_raw in NUME_BOLI:
        return NUME_BOLI[boala_raw][limba]
    return boala_raw.replace("_", " ")


def gaseste_sfat(boala_raw, limba):
    if este_sanatos(boala_raw):
        return sfaturi["healthy"][limba]
    if boala_raw in sfaturi:
        return sfaturi[boala_raw][limba]
    return TEXTE[limba]["sfat_necunoscut"]


def nume_scurt_predictie(planta_raw, boala_raw, e_sanatos, limba):
    t = TEXTE[limba]
    planta_disp = traduce_planta(planta_raw, limba)
    boala_disp = t["sanatos"] if e_sanatos else traduce_boala(boala_raw, limba)
    return f"{planta_disp} - {boala_disp}"


def cauta_fisier(nume):
    # cand e compilat cu pyinstaller sys.frozen e True si sys.executable e path-ul catre exe
    # altfel (rulat normal cu python) folosim __file__ ca de obicei
    if getattr(sys, "frozen", False):
        folder = Path(sys.executable).parent
    else:
        folder = Path(__file__).parent
    cale = folder / nume
    return str(cale) if cale.exists() else nume


def este_fisier_imagine(cale):
    return Path(cale).suffix.lower() in EXT_IMAGINI


def decodeaza_cale_drop(raw):
    if isinstance(raw, bytes):
        for enc in ("utf-8", "mbcs", "cp1252"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _hex_la_rgb(hex_str):
    """Converteste o culoare hex ('#RRGGBB') intr-un tuplu (r,g,b) 0-1,
    asa cum il asteapta reportlab. Asa refolosim paleta de culori a
    aplicatiei si in PDF, nu doar in interfata."""
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # reportlab vrea valori intre 0 si 1, nu 0-255


def _imparte_text_simplu(canvas_obj, text, font, marime, latime_max):
    """Imparte un text in randuri care se incadreaza in latime_max,
    masurand latimea reala a textului cu fontul curent (mai precis
    decat o estimare dupa numarul de caractere)."""
    cuvinte = text.split()
    randuri = []
    linie = ""
    for cuvant in cuvinte:
        test = f"{linie} {cuvant}".strip()
        if canvas_obj.stringWidth(test, font, marime) <= latime_max:
            linie = test
        else:
            if linie:
                randuri.append(linie)
            linie = cuvant
    if linie:
        randuri.append(linie)
    return randuri


class ConstructorRaportPDF:
    """Construieste raportul PDF pe (eventual) mai multe pagini: antet si
    subsol repetate pe fiecare pagina, cu paginare automata cand
    continutul nu mai incape pe pagina curenta."""
    #tot ce se scrie
    # in pdf trece pe aici (titluri, paragrafe, bullets), asa ca macar e reutilizabil

    MARGINE_X = 30
    MARGINE_JOS = 55

    def __init__(self, canvas_obj, dimensiune_pagina, texte, limba):
        self.c = canvas_obj
        self.W, self.H = dimensiune_pagina
        self.t = texte
        self.limba = limba
        self.pagina = 1
        self.y = self.H
        self.latime_utila = self.W - 2 * self.MARGINE_X
        self._deseneaza_antet(prima=True)

    def _deseneaza_antet(self, prima):
        c = self.c
        inaltime_antet = 70 if prima else 46
        c.setFillColorRGB(*_hex_la_rgb(culori["btn_verde"]))
        c.rect(0, self.H - inaltime_antet, self.W, inaltime_antet, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        if prima:
            c.setFont("Helvetica-Bold", 24)
            c.drawString(self.MARGINE_X, self.H - 42, self.t["pdf_titlu"])
            c.setFont("Helvetica", 10)
            c.drawString(self.MARGINE_X, self.H - 58, self.t["pdf_subtitlu"])
            c.setFont("Helvetica", 8)
            c.drawRightString(self.W - self.MARGINE_X, self.H - 42,
                              datetime.now().strftime("%d.%m.%Y  %H:%M"))
        else:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(self.MARGINE_X, self.H - 29, f"{self.t['pdf_titlu']} - {self.t['pdf_subtitlu']}")
            c.setFont("Helvetica", 8)
            c.drawRightString(self.W - self.MARGINE_X, self.H - 29,
                              datetime.now().strftime("%d.%m.%Y  %H:%M"))
        self.y = self.H - inaltime_antet - 22

    def _deseneaza_subsol(self):
        c = self.c
        c.setStrokeColorRGB(*_hex_la_rgb(culori["chenar"]))
        c.setLineWidth(0.6)
        c.line(self.MARGINE_X, 36, self.W - self.MARGINE_X, 36)
        c.setFillColorRGB(0.55, 0.55, 0.55)
        c.setFont("Helvetica", 6.5)
        randuri = _imparte_text_simplu(c, self.t["pdf_disclaimer"], "Helvetica", 6.5, self.latime_utila)
        y = 27
        for rand in randuri[:2]:
            c.drawString(self.MARGINE_X, y, rand)
            y -= 8
        c.setFont("Helvetica", 8)
        c.drawRightString(self.W - self.MARGINE_X, 27, f"{self.t['pdf_pagina']} {self.pagina}")

    def pagina_noua(self):
        self._deseneaza_subsol()
        self.c.showPage()
        self.pagina += 1
        self._deseneaza_antet(prima=False)

    def verifica_spatiu(self, inaltime_necesara):
        if self.y - inaltime_necesara < self.MARGINE_JOS:
            self.pagina_noua()

    def spatiu(self, px):
        self.y -= px

    def titlu_sectiune(self, text):
        self.verifica_spatiu(28)
        c = self.c
        c.setFillColorRGB(*_hex_la_rgb(culori["btn_hover"]))
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(self.MARGINE_X, self.y, text.upper())
        self.y -= 6
        c.setStrokeColorRGB(*_hex_la_rgb(culori["chenar"]))
        c.setLineWidth(1)
        c.line(self.MARGINE_X, self.y, self.W - self.MARGINE_X, self.y)
        self.y -= 14

    def paragraf(self, text, font="Helvetica", marime=9.5, culoare=(0.2, 0.2, 0.2),
                 interlinie=13, indent=0):
        c = self.c
        latime = self.latime_utila - indent
        for rand in _imparte_text_simplu(c, text, font, marime, latime):
            self.verifica_spatiu(interlinie)
            c.setFillColorRGB(*culoare)
            c.setFont(font, marime)
            c.drawString(self.MARGINE_X + indent, self.y, rand)
            self.y -= interlinie

    def bullet(self, text, culoare_punct=None, **kw):
        self.verifica_spatiu(14)
        c = self.c
        c.setFillColorRGB(*_hex_la_rgb(culoare_punct or culori["btn_verde"]))
        c.circle(self.MARGINE_X + 3, self.y + 3, 2, fill=1, stroke=0)
        self.paragraf(text, indent=13, **kw)

    def finalizeaza(self):
        self._deseneaza_subsol()


class ButonRotund(tk.Canvas):
    """Buton cu colturi rotunjite, randate cu supersampling in PIL ca sa nu
    mai apara colturi negre/patrate (Canvas.create_arc nu antialiaseaza)."""
    # gasit trick-ul asta cautand pe net dupa "tkinter rounded button no antialiasing"
    # basically desenez mai mare apoi micsorez imaginea, asa iese neted

    SUPRASCALARE = 4  # la 2x tot se mai vedea un pic de aliasing, la 4x arata bine si nu-i prea lent

    def __init__(self, parinte, text, la_click=None, culoare="#16A34A",
                 culoare_hover="#15803D", culoare_text="#FFFFFF",
                 latime=160, inaltime=42, raza=10, font=None,
                 bg_fundal=None, **kw):
        fundal_real = bg_fundal or culori["fundal"]
        super().__init__(parinte, width=latime, height=inaltime,
                         bg=fundal_real, highlightthickness=0, **kw)
        self.la_click = la_click
        self.culoare = culoare
        self.culoare_hover = culoare_hover
        self.culoare_text = culoare_text
        self.raza = raza
        self.text = text
        self.font = font or ("Helvetica", 11, "bold")
        self.latime = latime
        self.inaltime = inaltime
        self.bg_fundal = fundal_real
        self.activ = True
        self._imagine_ref = None
        self.deseneaza(culoare)
        self.bind("<Enter>", lambda e: self.mouse_intrat())
        self.bind("<Leave>", lambda e: self.mouse_iesit())
        self.bind("<Button-1>", lambda e: self.apasat())

    def _randeaza_fundal_rotunjit(self, culoare_fill):
        s = self.SUPRASCALARE
        w, h, r = self.latime * s, self.inaltime * s, self.raza * s
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        desen = ImageDraw.Draw(img)
        desen.rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=culoare_fill)
        img = img.resize((self.latime, self.inaltime), Image.LANCZOS)
        fundal = Image.new("RGB", (self.latime, self.inaltime), self.bg_fundal)
        fundal.paste(img, (0, 0), img)
        return ImageTk.PhotoImage(fundal)

    def deseneaza(self, culoare_fill):
        self.delete("all")
        self._imagine_ref = self._randeaza_fundal_rotunjit(culoare_fill)
        self.create_image(0, 0, image=self._imagine_ref, anchor="nw")
        c_txt = self.culoare_text if self.activ else "#4B5563"
        self.create_text(self.latime // 2, self.inaltime // 2,
                         text=self.text, fill=c_txt, font=self.font)

    def mouse_intrat(self):
        if self.activ:
            self.deseneaza(self.culoare_hover)
            self.config(cursor="hand2")

    def mouse_iesit(self):
        if self.activ:
            self.deseneaza(self.culoare)
            self.config(cursor="")

    def apasat(self):
        if self.activ and self.la_click:
            self.la_click()

    def seteaza_activ(self, da_sau_nu):
        self.activ = da_sau_nu
        self.deseneaza(self.culoare if da_sau_nu else culori["btn_gri"])

    def seteaza_text(self, text_nou):
        self.text = text_nou
        self.deseneaza(self.culoare if self.activ else culori["btn_gri"])


class CardPredictie(tk.Frame):
    """Casuta compacta pentru o predictie - afisata in rand orizontal.
    Poate fi apasata pentru a alege pentru care predictie se afiseaza
    sfatul agricol (util cand prima predictie nu se potriveste)."""
    # ideea de "click pe card ca sa schimbi sfatul afisat" a venit dupa ce am testat
    # aplicatia cu un prieten si modelul zicea des planta buna dar boala gresita (sau invers)

    WRAP = 145

    def __init__(self, parinte, index, cheie_loc, la_click=None, **kw):
        super().__init__(
            parinte,
            bg=culori["card"],
            highlightbackground=culori["chenar"],
            highlightthickness=1,
            padx=8,
            pady=8,
            cursor="hand2",
            **kw,
        )
        self.index = index
        self.la_click = la_click

        self.lbl_loc = tk.Label(
            self, text=cheie_loc, font=("Helvetica", 8, "bold"),
            bg=culori["card"], fg=culori["text_slab"],
        )
        self.lbl_loc.pack(anchor="w")

        self.lbl_stare = tk.Label(
            self, text="—", font=("Helvetica", 7, "bold"),
            bg=culori["card"], fg=culori["text_slab"], padx=6, pady=1,
        )
        self.lbl_stare.pack(anchor="w", pady=(4, 2))

        self.lbl_icon = tk.Label(
            self, text="🌿", font=("Helvetica", 20),
            bg=culori["card"],
        )
        self.lbl_icon.pack(pady=(0, 2))

        self.lbl_planta_titlu = tk.Label(
            self, text="", font=("Helvetica", 7),
            bg=culori["card"], fg=culori["text_slab"],
        )
        self.lbl_planta = tk.Label(
            self, text="—", font=("Helvetica", 9, "bold"),
            bg=culori["card"], fg=culori["text"],
            wraplength=self.WRAP, justify="center",
        )
        self.lbl_planta_titlu.pack()
        self.lbl_planta.pack(pady=(0, 4))

        self.lbl_boala_titlu = tk.Label(
            self, text="", font=("Helvetica", 7),
            bg=culori["card"], fg=culori["text_slab"],
        )
        self.lbl_boala = tk.Label(
            self, text="—", font=("Helvetica", 9),
            bg=culori["card"], fg=culori["text_dim"],
            wraplength=self.WRAP, justify="center",
        )
        self.lbl_boala_titlu.pack()
        self.lbl_boala.pack(pady=(0, 6))

        self.lbl_procent = tk.Label(
            self, text="—", font=("Helvetica", 11, "bold"),
            bg=culori["card"], fg=culori["verde"],
        )
        self.lbl_procent.pack()

        self.bara = ttk.Progressbar(
            self, length=140, mode="determinate",
            style="Incredere.Horizontal.TProgressbar",
        )
        self.bara.pack(fill="x", pady=(4, 0))

        self._leaga_click_recursiv(self)

    def _leaga_click_recursiv(self, widget):
        # leg click-ul pe TOATE widget-urile copil, nu doar pe frame-ul principal,
        # altfel daca dai click exact pe un label din interior nu se intampla nimic
        widget.bind("<Button-1>", self._pe_click)
        for copil in widget.winfo_children():
            self._leaga_click_recursiv(copil)

    def _pe_click(self, event):
        if self.la_click:
            self.la_click(self.index)

    def seteaza_etichete_campuri(self, camp_planta, camp_conditie):
        self.lbl_planta_titlu.config(text=camp_planta)
        self.lbl_boala_titlu.config(text=camp_conditie)

    def seteaza_loc(self, text_loc):
        self.lbl_loc.config(text=text_loc)

    def afiseaza_gol(self, text_asteapta="—"):
        self.config(bg=culori["card"], highlightbackground=culori["chenar"])
        for w in (self.lbl_loc, self.lbl_stare, self.lbl_icon,
                  self.lbl_planta_titlu, self.lbl_planta,
                  self.lbl_boala_titlu, self.lbl_boala,
                  self.lbl_procent):
            w.config(bg=culori["card"])
        self.lbl_stare.config(text=text_asteapta, bg=culori["card"], fg=culori["text_slab"])
        self.lbl_icon.config(text="🌿")
        self.lbl_planta.config(text="—", fg=culori["text"])
        self.lbl_boala.config(text="—", fg=culori["text_dim"])
        self.lbl_procent.config(text="—", fg=culori["text_slab"])
        self.bara["value"] = 0

    def afiseaza_predictie(self, planta_raw, boala_raw, scor, e_sanatos, limba, t, evideniat=False):
        bg = culori["card_evident"] if evideniat else culori["card"]
        chenar = culori["verde"] if evideniat else culori["chenar"]
        self.config(bg=bg, highlightbackground=chenar)

        planta_disp = traduce_planta(planta_raw, limba)
        procent = round(scor * 100, 1)

        if e_sanatos:
            boala_disp = t["sanatos"]
            fond_st, text_st = culori["ok_fond"], culori["ok_text"]
            icon = "✅"
        else:
            boala_disp = traduce_boala(boala_raw, limba)
            fond_st, text_st = culori["rau_fond"], culori["rau_text"]
            icon = "⚠️"

        for w in (self.lbl_loc, self.lbl_icon, self.lbl_planta_titlu,
                  self.lbl_planta, self.lbl_boala_titlu, self.lbl_boala,
                  self.lbl_procent):
            w.config(bg=bg)

        self.lbl_stare.config(
            text=t["planta_sanatoasa"] if e_sanatos else t["boala_detectata"],
            bg=fond_st, fg=text_st,
        )
        self.lbl_icon.config(text=icon)
        self.lbl_planta.config(text=planta_disp, fg=culori["text"])
        self.lbl_boala.config(text=boala_disp, fg=culori["text_dim"] if e_sanatos else culori["rau_text"])
        self.lbl_procent.config(text=f"{procent}%", fg=culori["verde"])
        self.bara["value"] = procent


class AplicatieLeafLens:
    """Clasa principala, tine toata starea aplicatiei"""

    def __init__(self, fereastra):
        self.fereastra = fereastra
        self.fereastra.title(TITLU_FEREASTRA)
        self.fereastra.configure(bg=culori["fundal"])
        self.fereastra.geometry("760x880")
        self.fereastra.resizable(True, True)
        self.fereastra.minsize(720, 780)

        self.limba = "ro"

        self.model_incarcat = None
        self.model_frunza = None
        self.poza_aleasa = None
        self.referinta_preview = None
        self.e_fullscreen = False

        self.coada_poze = []
        self.index_curent = 0
        self.cache_rezultate = {}
        self.analiza_in_curs = False
        self.index_sfat_selectat = 0

        self.fisier_model = cauta_fisier("best_model.keras")
        self.fisier_etichete = cauta_fisier("label_map.json")
        self.fisier_leaf_detector = cauta_fisier("leaf_detector.keras")

        self.stare_status_cheie = "incarcare_modele"
        self.stare_status_params = {}
        self.stare_status_tip = "incarcare"
        self.stare_afisare = "asteapta"

        self.rezultat_predictii_raw = None
        self.debug = False  # nu fac nimic cu asta inca, l-am pus cand debugam si nu l-am mai scos

        self.fereastra.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.fereastra.bind("<Escape>", lambda e: self.iesi_fullscreen())
        self.fereastra.bind("<Left>", lambda e: self.navigare_anterior())
        self.fereastra.bind("<Right>", lambda e: self.navigare_urmator())

        self.configureaza_progressbar()
        self.construieste_interfata()
        self.porneste_incarcare_model()

    def texte(self):
        return TEXTE[self.limba]

    def comuta_limba(self):
        self.limba = "en" if self.limba == "ro" else "ro"
        self.aplica_limba()

    def toggle_fullscreen(self):
        self.e_fullscreen = not self.e_fullscreen
        self.fereastra.attributes("-fullscreen", self.e_fullscreen)

    def iesi_fullscreen(self):
        self.e_fullscreen = False
        self.fereastra.attributes("-fullscreen", False)

    def configureaza_progressbar(self):
        stil = ttk.Style()
        stil.theme_use("clam")
        stil.configure("Verde.Horizontal.TProgressbar",
                       troughcolor=culori["suprafata"], background=culori["verde"],
                       lightcolor=culori["verde"], darkcolor=culori["verde2"],
                       bordercolor=culori["suprafata"], thickness=5)
        stil.configure("Incredere.Horizontal.TProgressbar",
                       troughcolor=culori["suprafata"], background=culori["verde"],
                       lightcolor=culori["verde"], darkcolor=culori["verde2"],
                       bordercolor=culori["suprafata"], thickness=5)

    def construieste_interfata(self):
        t = self.texte()

        zona_titlu = tk.Frame(self.fereastra, bg=culori["fundal"], pady=12)
        zona_titlu.pack(fill="x", padx=24)

        row = tk.Frame(zona_titlu, bg=culori["fundal"])
        row.pack(anchor="w")
        tk.Label(row, text="🌿", font=("Helvetica", 24),
                 bg=culori["fundal"], fg=culori["verde"]).pack(side="left", padx=(0, 8))

        texte_titlu = tk.Frame(row, bg=culori["fundal"])
        texte_titlu.pack(side="left")
        tk.Label(texte_titlu, text="LeafLens", font=("Helvetica", 20, "bold"),
                 bg=culori["fundal"], fg=culori["text"]).pack(anchor="w")
        self.subtitlu_label = tk.Label(texte_titlu, text=t["subtitlu"],
                                       font=("Helvetica", 9), bg=culori["fundal"], fg=culori["text_slab"])
        self.subtitlu_label.pack(anchor="w")

        self.btn_fullscreen_label = tk.Label(zona_titlu, text=f"⛶  {t['fullscreen']}",
                                             font=("Helvetica", 9), bg=culori["fundal"],
                                             fg=culori["text_slab"], cursor="hand2")
        self.btn_fullscreen_label.pack(side="right", anchor="n")
        self.btn_fullscreen_label.bind("<Button-1>", lambda e: self.toggle_fullscreen())

        self.btn_limba_label = tk.Label(zona_titlu, text=f"🌐  {t['buton_limba']}",
                                        font=("Helvetica", 9, "bold"), bg=culori["fundal"],
                                        fg=culori["verde"], cursor="hand2")
        self.btn_limba_label.pack(side="right", anchor="n", padx=(0, 16))
        self.btn_limba_label.bind("<Button-1>", lambda e: self.comuta_limba())

        self.btn_informatii_label = tk.Label(zona_titlu, text=t["btn_informatii"],
                                             font=("Helvetica", 9, "bold"), bg=culori["fundal"],
                                             fg=culori["verde"], cursor="hand2")
        self.btn_informatii_label.pack(side="right", anchor="n", padx=(0, 16))
        self.btn_informatii_label.bind("<Button-1>", lambda e: self.afiseaza_dialog_informatii())

        tk.Frame(self.fereastra, bg=culori["chenar"], height=1).pack(fill="x", padx=24)

        self.zona_status = tk.Frame(self.fereastra, bg=culori["incarcare_f"], padx=14, pady=6)
        self.zona_status.pack(fill="x", padx=24, pady=(8, 0))
        self.punct_status = tk.Label(self.zona_status, text="o", font=("Helvetica", 9),
                                     bg=culori["incarcare_f"], fg=culori["incarcare_t"])
        self.punct_status.pack(side="left", padx=(0, 6))
        self.text_status = tk.Label(self.zona_status, text=t["incarcare_modele"],
                                    font=("Helvetica", 10), bg=culori["incarcare_f"],
                                    fg=culori["incarcare_t"], anchor="w")
        self.text_status.pack(side="left")

        zona_scroll = tk.Frame(self.fereastra, bg=culori["fundal"])
        zona_scroll.pack(fill="both", expand=True)

        self.canvas_principal = tk.Canvas(zona_scroll, bg=culori["fundal"], highlightthickness=0)
        scrollbar_principal = ttk.Scrollbar(
            zona_scroll, orient="vertical", command=self.canvas_principal.yview)
        self.canvas_principal.configure(yscrollcommand=scrollbar_principal.set)
        self.canvas_principal.pack(side="left", fill="both", expand=True)
        scrollbar_principal.pack(side="right", fill="y")

        corp_wrapper = tk.Frame(self.canvas_principal, bg=culori["fundal"])
        self._corp_window_id = self.canvas_principal.create_window(
            (0, 0), window=corp_wrapper, anchor="nw")

        def _actualizeaza_zona_scroll(event=None):
            self.canvas_principal.configure(scrollregion=self.canvas_principal.bbox("all"))

        def _potriveste_latimea(event):
            self.canvas_principal.itemconfig(self._corp_window_id, width=event.width)

        corp_wrapper.bind("<Configure>", _actualizeaza_zona_scroll)
        self.canvas_principal.bind("<Configure>", _potriveste_latimea)

        def _pe_rotita_mouse(event):
            delta = event.delta
            if delta:
                self.canvas_principal.yview_scroll(int(-1 * (delta / 120)), "units")
            elif getattr(event, "num", None) == 4:
                self.canvas_principal.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                self.canvas_principal.yview_scroll(1, "units")

        self.canvas_principal.bind("<Enter>", lambda e: (
            self.canvas_principal.bind_all("<MouseWheel>", _pe_rotita_mouse),
            self.canvas_principal.bind_all("<Button-4>", _pe_rotita_mouse),
            self.canvas_principal.bind_all("<Button-5>", _pe_rotita_mouse),
        ))
        self.canvas_principal.bind("<Leave>", lambda e: (
            self.canvas_principal.unbind_all("<MouseWheel>"),
            self.canvas_principal.unbind_all("<Button-4>"),
            self.canvas_principal.unbind_all("<Button-5>"),
        ))

        corp = tk.Frame(corp_wrapper, bg=culori["fundal"])
        corp.pack(fill="both", expand=True, padx=24, pady=8)

        card_poza = tk.Frame(corp, bg=culori["card"],
                             highlightbackground=culori["chenar"], highlightthickness=1)
        card_poza.pack(fill="x", pady=(0, 6))
        self.card_poza_titlu = tk.Label(card_poza, text=t["sectiune_imagine"], font=("Helvetica", 8, "bold"),
                                        bg=culori["card"], fg=culori["text_slab"], padx=14, pady=5)
        self.card_poza_titlu.pack(anchor="w")
        tk.Frame(card_poza, bg=culori["chenar"], height=1).pack(fill="x")

        self.zona_preview = tk.Canvas(card_poza, width=652, height=180,
                                      bg=culori["suprafata"], highlightthickness=0)
        self.zona_preview.pack(fill="x")
        hint_text = t["hint_poza"]
        self.text_hint = self.zona_preview.create_text(
            326, 90, text=hint_text,
            font=("Helvetica", 11), fill=culori["text_slab"], width=480, justify="center")
        rand_coada = tk.Frame(card_poza, bg=culori["card"], padx=14, pady=6)
        rand_coada.pack(fill="x")

        self.btn_anterior = ButonRotund(
            rand_coada, text=t["btn_anterior"], la_click=self.navigare_anterior,
            culoare=culori["btn_gri"], culoare_hover="#374151",
            latime=108, inaltime=32, raza=8, font=("Helvetica", 9, "bold"),
            bg_fundal=culori["card"],
        )
        self.btn_anterior.pack(side="left")
        self.btn_anterior.seteaza_activ(False)

        self.lbl_coada = tk.Label(
            rand_coada, text=t["coada_goala"], font=("Helvetica", 10, "bold"),
            bg=culori["card"], fg=culori["text"],
        )
        self.lbl_coada.pack(side="left", expand=True)

        self.btn_urmator = ButonRotund(
            rand_coada, text=t["btn_urmator"], la_click=self.navigare_urmator,
            culoare=culori["btn_gri"], culoare_hover="#374151",
            latime=108, inaltime=32, raza=8, font=("Helvetica", 9, "bold"),
            bg_fundal=culori["card"],
        )
        self.btn_urmator.pack(side="right")
        self.btn_urmator.seteaza_activ(False)

        rand_butoane = tk.Frame(corp, bg=culori["fundal"])
        rand_butoane.pack(fill="x", pady=(0, 6))

        self.btn_alege = ButonRotund(rand_butoane, text=t["btn_alege"],
                                     la_click=self.alege_poze,
                                     culoare=culori["btn_verde"],
                                     culoare_hover=culori["btn_hover"],
                                     latime=165, inaltime=38)
        self.btn_alege.pack(side="left", padx=(0, 6))

        self.btn_pdf = ButonRotund(rand_butoane, text=t["btn_pdf"],
                                   la_click=self.salveaza_pdf,
                                   culoare=culori["btn_gri"],
                                   culoare_hover="#374151",
                                   latime=145, inaltime=38)
        self.btn_pdf.pack(side="left", padx=(0, 6))
        self.btn_pdf.seteaza_activ(False)

        self.btn_sterge = ButonRotund(rand_butoane, text=t["btn_sterge"],
                                      la_click=self.reseteaza_tot,
                                      culoare=culori["btn_gri"],
                                      culoare_hover="#374151",
                                      latime=95, inaltime=38)
        self.btn_sterge.pack(side="right")

        self.val_progress = tk.DoubleVar()
        self.bara_progress = ttk.Progressbar(corp, variable=self.val_progress,
                                             mode="indeterminate", length=652, style="Verde.Horizontal.TProgressbar")
        self.bara_progress.pack(fill="x", pady=(0, 6))

        #REZULTATE PRIMELE 3 Afisare
        card_rezultat = tk.Frame(corp, bg=culori["card"],
                                 highlightbackground=culori["chenar"], highlightthickness=1)
        card_rezultat.pack(fill="x", pady=(0, 6))

        self.card_rezultat_titlu = tk.Label(
            card_rezultat, text=t["sectiune_rezultate"],
            font=("Helvetica", 8, "bold"),
            bg=culori["card"], fg=culori["text_slab"], padx=14, pady=5,
        )
        self.card_rezultat_titlu.pack(anchor="w")
        tk.Frame(card_rezultat, bg=culori["chenar"], height=1).pack(fill="x")

        detalii = tk.Frame(card_rezultat, bg=culori["card"], padx=10, pady=10)
        detalii.pack(fill="x")

        self.zona_cards = tk.Frame(detalii, bg=culori["card"])
        self.zona_cards.pack(fill="x")
        for col in range(NUMAR_PREDICTII):
            self.zona_cards.columnconfigure(col, weight=1, uniform="pred")

        chei_loc = ["card_loc_1", "card_loc_2", "card_loc_3"]
        self.cards_predictii = []
        for i in range(NUMAR_PREDICTII):
            card = CardPredictie(self.zona_cards, i, t[chei_loc[i]],
                                 la_click=self.selecteaza_sfat)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 3, 0))
            card.seteaza_etichete_campuri(t["camp_planta"], t["camp_conditie"])
            card.afiseaza_gol(t["asteapta_scanare"])
            self.cards_predictii.append(card)

        self.lbl_apasa_alt_sfat = tk.Label(
            detalii, text=t["apasa_pt_alt_sfat"], font=("Helvetica", 8, "italic"),
            bg=culori["card"], fg=culori["text_slab"], wraplength=680, justify="left",
        )

        # Banner unificat pentru "nu e frunza" (inlocuieste cele 3 carduri,
        # nu mai apare tripleta acelasi mesaj)
        self.zona_nu_frunza = tk.Frame(
            detalii, bg=culori["card"],
            highlightbackground=culori["nefrunza_fond"], highlightthickness=1,
            padx=16, pady=18,
        )
        self.lbl_nu_frunza_icon = tk.Label(
            self.zona_nu_frunza, text="❓", font=("Helvetica", 34),
            bg=culori["card"],
        )
        self.lbl_nu_frunza_icon.pack()
        self.lbl_nu_frunza_tag = tk.Label(
            self.zona_nu_frunza, text=t["nu_e_frunza_tag"],
            font=("Helvetica", 8, "bold"), bg=culori["nefrunza_fond"],
            fg=culori["nefrunza_text"], padx=8, pady=2,
        )
        self.lbl_nu_frunza_tag.pack(pady=(6, 6))
        self.lbl_nu_frunza_titlu = tk.Label(
            self.zona_nu_frunza, text=t["nu_e_frunza_titlu"],
            font=("Helvetica", 12, "bold"), bg=culori["card"], fg=culori["text"],
        )
        self.lbl_nu_frunza_titlu.pack()

        self.cutie_sfat = tk.Frame(detalii, bg=culori["suprafata"], padx=12, pady=6)
        self.cutie_sfat.pack(fill="x", pady=(8, 0))
        self.sfat_titlu_label = tk.Label(
            self.cutie_sfat, text=t["sfat_titlu"], font=("Helvetica", 8, "bold"),
            bg=culori["suprafata"], fg=culori["verde"],
        )
        self.sfat_titlu_label.pack(anchor="w")
        self.lbl_sfat_pentru = tk.Label(
            self.cutie_sfat, text="", font=("Helvetica", 8),
            bg=culori["suprafata"], fg=culori["text_slab"],
        )
        self.lbl_sfat_pentru.pack(anchor="w", pady=(2, 0))
        self.text_sfat = tk.Label(
            self.cutie_sfat, text=t["tip_implicit"],
            font=("Helvetica", 10), bg=culori["suprafata"], fg=culori["text_dim"],
            wraplength=680, justify="left",
        )
        self.text_sfat.pack(anchor="w", pady=(3, 0))

    def aplica_limba(self):
        t = self.texte()

        self.subtitlu_label.config(text=t["subtitlu"])
        self.btn_fullscreen_label.config(text=f"⛶  {t['fullscreen']}")
        self.btn_limba_label.config(text=f"🌐  {t['buton_limba']}")
        self.btn_informatii_label.config(text=t["btn_informatii"])
        self.card_poza_titlu.config(text=t["sectiune_imagine"])

        self.card_rezultat_titlu.config(text=t["sectiune_rezultate"])
        self.btn_alege.seteaza_text(t["btn_alege"])
        self.btn_anterior.seteaza_text(t["btn_anterior"])
        self.btn_urmator.seteaza_text(t["btn_urmator"])
        self.btn_pdf.seteaza_text(t["btn_pdf"])
        self.btn_sterge.seteaza_text(t["btn_sterge"])
        self.sfat_titlu_label.config(text=t["sfat_titlu"])
        self.lbl_apasa_alt_sfat.config(text=t["apasa_pt_alt_sfat"])

        chei_loc = ["card_loc_1", "card_loc_2", "card_loc_3"]
        for i, card in enumerate(self.cards_predictii):
            card.seteaza_loc(t[chei_loc[i]])
            card.seteaza_etichete_campuri(t["camp_planta"], t["camp_conditie"])

        self.lbl_nu_frunza_tag.config(text=t["nu_e_frunza_tag"])
        self.lbl_nu_frunza_titlu.config(text=t["nu_e_frunza_titlu"])

        if not self.coada_poze:
            hint_text = t["hint_poza"]
            self.zona_preview.itemconfig(self.text_hint, text=hint_text)

        self.actualizeaza_navigare()

        mesaj_status = t[self.stare_status_cheie].format(**self.stare_status_params)
        self._seteaza_status_text(mesaj_status, self.stare_status_tip)

        if self.stare_afisare == "asteapta":
            self._afiseaza_asteapta_scanare()
        elif self.stare_afisare == "nu_frunza":
            self._randeaza_nu_e_frunza()
        elif self.stare_afisare == "rezultat":
            self._randeaza_rezultat()

    def actualizeaza_status(self, cheie, tip="info", **params):
        self.stare_status_cheie = cheie
        self.stare_status_params = params
        self.stare_status_tip = tip
        mesaj = self.texte()[cheie].format(**params)
        self._seteaza_status_text(mesaj, tip)

    def _seteaza_status_text(self, mesaj, tip):
        if tip == "incarcare":
            bg, fg = culori["incarcare_f"], culori["incarcare_t"]
        elif tip == "ok":
            bg, fg = culori["ok_fond"], culori["ok_text"]
        elif tip == "eroare":
            bg, fg = culori["rau_fond"], culori["rau_text"]
        else:
            bg, fg = culori["suprafata"], culori["text_dim"]

        def aplica():
            self.zona_status.config(bg=bg)
            self.punct_status.config(bg=bg, fg=fg)
            self.text_status.config(text=mesaj, bg=bg, fg=fg)

        self.fereastra.after(0, aplica)

    def porneste_incarcare_model(self):
        # incarc modelele pe un thread separat ca sa nu inghete fereastra Tkinter cat timp
        # se incarca (modelele mari pot dura cateva secunde bune)
        def worker():
            self.actualizeaza_status("incarcare_modele", "incarcare")
            if not os.path.exists(self.fisier_model):
                self.actualizeaza_status("model_negasit", "eroare", cale=self.fisier_model)
                return

            self.model_incarcat = tf.keras.models.load_model(self.fisier_model, compile=False)

            if os.path.exists(self.fisier_leaf_detector):
                self.model_frunza = tf.keras.models.load_model(
                    self.fisier_leaf_detector,
                    compile=False,
                    safe_mode=False,
                    custom_objects={"preprocess_input": preprocess_input},
                )
            else:
                # daca nu gasim modelul de detectie frunza, mergem mai departe fara el
                # (nu e critic, doar nu mai filtram pozele care nu contin o frunza)
                self.model_frunza = None
                print(f"Atentie: nu gasesc {self.fisier_leaf_detector} - filtrul de frunza e dezactivat")

            if os.path.exists(self.fisier_etichete):
                with open(self.fisier_etichete) as f:
                    date = json.load(f)
                    for k, v in date.items():
                        boli[int(k)] = v

            self.actualizeaza_status("gata", "ok")

        threading.Thread(target=worker, daemon=True).start()

    def adauga_in_coada(self, cai, auto_analizeaza=True):
        t = self.texte()
        noi = []
        for cale in cai:
            cale = os.path.normpath(cale.strip().strip('"'))
            if not os.path.isfile(cale):
                continue
            if not este_fisier_imagine(cale):
                print(t["fisier_invalid"].format(nume=Path(cale).name))
                continue
            if cale not in self.coada_poze:
                noi.append(cale)

        if not noi:
            return

        index_prima_poza_noua = len(self.coada_poze)
        self.coada_poze.extend(noi)

        # Sari automat pe prima poza nou adaugata, indiferent daca
        # mai erau deja poze in coada inainte. (Quality of life)
        self.index_curent = index_prima_poza_noua
        self.afiseaza_poza_curenta()
        self.actualizeaza_navigare()

        if auto_analizeaza and self.model_incarcat:
            self.ruleaza_analiza()

    def poza_curenta(self):
        if not self.coada_poze:
            return None
        self.index_curent = max(0, min(self.index_curent, len(self.coada_poze) - 1))
        return self.coada_poze[self.index_curent]

    def actualizeaza_navigare(self):
        t = self.texte()
        total = len(self.coada_poze)
        if total == 0:
            self.lbl_coada.config(text=t["coada_goala"], fg=culori["text_slab"])
            self.btn_anterior.seteaza_activ(False)
            self.btn_urmator.seteaza_activ(False)
            return

        self.lbl_coada.config(
            text=t["coada_pozitie"].format(curenta=self.index_curent + 1, total=total),
            fg=culori["verde"],
        )
        poate_naviga = not self.analiza_in_curs
        self.btn_anterior.seteaza_activ(poate_naviga and self.index_curent > 0)
        self.btn_urmator.seteaza_activ(poate_naviga and self.index_curent < total - 1)

        if poate_naviga and self.index_curent > 0:
            self.btn_anterior.culoare = culori["btn_verde"]
            self.btn_anterior.deseneaza(culori["btn_verde"])
        if poate_naviga and self.index_curent < total - 1:
            self.btn_urmator.culoare = culori["btn_verde"]
            self.btn_urmator.deseneaza(culori["btn_verde"])

    def navigare_anterior(self):
        if self.analiza_in_curs or self.index_curent <= 0:
            return
        self.index_curent -= 1
        self.afiseaza_poza_curenta()
        self.actualizeaza_navigare()

    def navigare_urmator(self):
        if self.analiza_in_curs or self.index_curent >= len(self.coada_poze) - 1:
            return
        self.index_curent += 1
        self.afiseaza_poza_curenta()
        self.actualizeaza_navigare()
        if self.poza_curenta() not in self.cache_rezultate and self.model_incarcat:
            self.ruleaza_analiza()

    def afiseaza_poza_curenta(self):
        cale = self.poza_curenta()
        self.poza_aleasa = cale
        self.index_sfat_selectat = 0

        if not cale:
            self.referinta_preview = None
            self.zona_preview.delete("all")
            t = self.texte()
            hint_text = t["hint_poza"]
            self.text_hint = self.zona_preview.create_text(
                326, 90, text=hint_text,
                font=("Helvetica", 11), fill=culori["text_slab"], width=480, justify="center")
            self.stare_afisare = "asteapta"
            self.rezultat_predictii_raw = None
            self._afiseaza_asteapta_scanare()
            self.btn_pdf.seteaza_activ(False)
            self.btn_pdf.culoare = culori["btn_gri"]
            self.btn_pdf.deseneaza(culori["btn_gri"])
            return

        self.afiseaza_preview(cale)
        self.incarca_rezultat_din_cache(cale)

    def incarca_rezultat_din_cache(self, cale):
        if cale not in self.cache_rezultate:
            self.stare_afisare = "asteapta"
            self.rezultat_predictii_raw = None
            self._afiseaza_asteapta_scanare()
            self.btn_pdf.seteaza_activ(False)
            self.btn_pdf.culoare = culori["btn_gri"]
            self.btn_pdf.deseneaza(culori["btn_gri"])
            return

        entry = self.cache_rezultate[cale]
        self.stare_afisare = entry["stare_afisare"]
        self.rezultat_predictii_raw = entry.get("predictii")

        if self.stare_afisare == "rezultat":
            self._randeaza_rezultat()
            self.btn_pdf.seteaza_activ(True)
            self.btn_pdf.culoare = "#1D4ED8"
            self.btn_pdf.deseneaza("#1D4ED8")
        elif self.stare_afisare == "nu_frunza":
            self._randeaza_nu_e_frunza()
            self.btn_pdf.seteaza_activ(False)
            self.btn_pdf.culoare = culori["btn_gri"]
            self.btn_pdf.deseneaza(culori["btn_gri"])

    def salveaza_in_cache(self, cale, stare_afisare, predictii=None):
        self.cache_rezultate[cale] = {
            "stare_afisare": stare_afisare,
            "predictii": predictii,
        }

    def alege_poze(self):
        t = self.texte()
        cai = filedialog.askopenfilenames(
            title=t["dialog_titlu_poze"],
            filetypes=[("Imagini", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Toate", "*.*")])
        if not cai:
            return
        self.adauga_in_coada(list(cai))

    def afiseaza_preview(self, cale):
        img = Image.open(cale)
        img.thumbnail((700, 180), Image.LANCZOS)
        self.referinta_preview = ImageTk.PhotoImage(img)
        self.zona_preview.delete("all")
        self.zona_preview.create_image(326, 90, image=self.referinta_preview, anchor="center")

    def ruleaza_analiza(self):
        cale = self.poza_curenta()
        if not cale or not self.model_incarcat or self.analiza_in_curs:
            return

        if cale in self.cache_rezultate:
            self.incarca_rezultat_din_cache(cale)
            return

        self.analiza_in_curs = True
        self.btn_alege.seteaza_activ(False)
        self.btn_pdf.seteaza_activ(False)
        self.btn_anterior.seteaza_activ(False)
        self.btn_urmator.seteaza_activ(False)
        self.bara_progress.start(8)

        def worker():
            rezultat = self._analizeaza_imagine(cale)
            self.fereastra.after(0, lambda: self._aplica_rezultat_analiza(cale, rezultat))
            self.fereastra.after(0, self.termina_analiza)

        threading.Thread(target=worker, daemon=True).start()

    def _analizeaza_imagine(self, cale):
        # rulat pe thread separat (vezi ruleaza_analiza), deci aici NU atingem widget-uri Tkinter direct
        img_original = Image.open(cale).convert("RGB")
        dim_originala = img_original.size  # nu o folosesc inca, ma gandeam s-o punem si-n raportul pdf candva

        e_frunza = True
        if self.model_frunza is not None:
            # 160x160 e dimensiunea pe care a fost antrenat leaf_detector-ul (mai mic = predictie mai rapida)
            img_binar = img_original.resize((DIM_INTRARE_FRUNZA, DIM_INTRARE_FRUNZA), Image.LANCZOS)
            tensor_binar = np.expand_dims(np.array(img_binar).astype(np.float32), axis=0)
            scor_binar = float(self.model_frunza.predict(tensor_binar, verbose=0)[0][0])
            # print("debug scor frunza:", scor_binar)  # las asta comentat, il dezcomentez cand testez threshold-ul
            e_frunza = scor_binar < 0.5

        if not e_frunza:
            return {"stare_afisare": "nu_frunza", "predictii": None}

        img = img_original.resize((DIM_INTRARE, DIM_INTRARE), Image.LANCZOS)
        tensor = np.expand_dims(np.array(img).astype(np.float32), axis=0)
        predictie = self.model_incarcat.predict(tensor, verbose=0)[0]

        indici_top = np.argsort(predictie)[::-1][:NUMAR_PREDICTII]
        predictii = []
        for idx in indici_top:
            eticheta = boli.get(int(idx), "Necunoscut")
            planta_raw, boala_raw = desparte_eticheta(eticheta)
            e_sanatos = este_sanatos(eticheta)
            scor = float(predictie[idx])
            predictii.append((planta_raw, boala_raw, scor, e_sanatos))

        return {"stare_afisare": "rezultat", "predictii": predictii}

    def _aplica_rezultat_analiza(self, cale, rezultat):
        self.salveaza_in_cache(cale, rezultat["stare_afisare"], rezultat["predictii"])
        if cale != self.poza_curenta():
            return

        if rezultat["stare_afisare"] == "nu_frunza":
            self.afiseaza_nu_e_frunza()
        else:
            self.afiseaza_rezultat(rezultat["predictii"])

    def afiseaza_rezultat(self, predictii):
        self.rezultat_predictii_raw = predictii
        self.stare_afisare = "rezultat"
        self.index_sfat_selectat = 0
        self._randeaza_rezultat()

        self.btn_pdf.seteaza_activ(True)
        self.btn_pdf.culoare = "#1D4ED8"
        self.btn_pdf.deseneaza("#1D4ED8")

    def selecteaza_sfat(self, index):
        """Apelata cand utilizatorul apasa pe una din cele 3 casute de
        predictie, ca sa vada tratamentul corespunzator acelei variante -
        util cand prima predictie nu se potriveste cu ce vede pe planta."""
        if not self.rezultat_predictii_raw or index >= len(self.rezultat_predictii_raw):
            return
        if self.stare_afisare != "rezultat":
            return
        self.index_sfat_selectat = index
        self._randeaza_rezultat()

    def _randeaza_rezultat(self):
        t = self.texte()
        self.zona_nu_frunza.pack_forget()
        self.zona_cards.pack(fill="x", before=self.cutie_sfat)
        for i, card in enumerate(self.cards_predictii):
            if i < len(self.rezultat_predictii_raw):
                p_raw, b_raw, scor, sanatos = self.rezultat_predictii_raw[i]
                card.afiseaza_predictie(p_raw, b_raw, scor, sanatos, self.limba, t,
                                        evideniat=(i == self.index_sfat_selectat))
            else:
                card.afiseaza_gol()

        self.lbl_apasa_alt_sfat.pack(fill="x", pady=(6, 0), before=self.cutie_sfat)

        indice = min(self.index_sfat_selectat, len(self.rezultat_predictii_raw) - 1)
        planta_raw, boala_raw, scor, e_sanatos = self.rezultat_predictii_raw[indice]
        planta_disp = traduce_planta(planta_raw, self.limba)
        boala_disp = t["sanatos"] if e_sanatos else traduce_boala(boala_raw, self.limba)

        chei_loc = ["card_loc_1", "card_loc_2", "card_loc_3"]
        self.lbl_sfat_pentru.config(
            text=t["sfat_pentru"].format(loc=t[chei_loc[indice]], planta=planta_disp, boala=boala_disp)
        )
        self.text_sfat.config(text=gaseste_sfat(boala_raw, self.limba))

    def afiseaza_nu_e_frunza(self):
        self.stare_afisare = "nu_frunza"
        self._randeaza_nu_e_frunza()

        self.btn_pdf.seteaza_activ(False)
        self.btn_pdf.culoare = culori["btn_gri"]
        self.btn_pdf.deseneaza(culori["btn_gri"])
        self.rezultat_predictii_raw = None

        cale = self.poza_curenta()
        if cale:
            self.salveaza_in_cache(cale, "nu_frunza", None)

    def _randeaza_nu_e_frunza(self):
        t = self.texte()
        self.zona_cards.pack_forget()
        self.lbl_apasa_alt_sfat.pack_forget()
        self.zona_nu_frunza.pack(fill="x", before=self.cutie_sfat)
        self.lbl_sfat_pentru.config(text="")
        self.text_sfat.config(text=t["nu_e_frunza_tip"])

    def _afiseaza_asteapta_scanare(self):
        t = self.texte()
        self.zona_nu_frunza.pack_forget()
        self.lbl_apasa_alt_sfat.pack_forget()
        self.zona_cards.pack(fill="x", before=self.cutie_sfat)
        for card in self.cards_predictii:
            card.afiseaza_gol(t["asteapta_scanare"])
        self.lbl_sfat_pentru.config(text="")
        self.text_sfat.config(text=t["tip_implicit"])

    def termina_analiza(self):
        self.analiza_in_curs = False
        self.bara_progress.stop()
        self.val_progress.set(0)
        self.btn_alege.seteaza_activ(True)
        self.actualizeaza_navigare()

    def salveaza_pdf(self):
        t = self.texte()
        if not self.poza_aleasa:
            messagebox.showwarning(t["atentie"], t["fara_poza"])
            return
        unde = filedialog.asksaveasfilename(
            title=t["dialog_titlu_pdf"],
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=f"leaflens_{datetime.now().strftime('%d%m%Y_%H%M')}.pdf")
        if not unde:
            return
        self.genereaza_pdf(unde)
        messagebox.showinfo(t["salvat_titlu"], t["salvat_mesaj"].format(cale=unde))

    def _nivel_incredere_pdf(self, procent, t):
        if procent >= 90:
            return t["pdf_nivel_foarte_ridicata"]
        if procent >= 70:
            return t["pdf_nivel_ridicata"]
        if procent >= 50:
            return t["pdf_nivel_medie"]
        return t["pdf_nivel_scazuta"]

    def genereaza_pdf(self, cale_output):
        # aici se construieste tot raportul pdf, sectiune cu sectiune (imagine, rezultat,
        # sfat, recomandari generale, celelalte variante din top 3). ConstructorRaportPDF
        # se ocupa de paginare automata cand nu mai incape continutul pe pagina curenta.
        t = self.texte()
        chei_loc = ["card_loc_1", "card_loc_2", "card_loc_3"]

        if not self.rezultat_predictii_raw:
            return

        # Varianta pe care utilizatorul a ales-o efectiv (poate sa nu fie #1
        # din Top 3, daca a apasat pe alta casuta ca sa vada alt tratament).
        indice_ales = min(self.index_sfat_selectat, len(self.rezultat_predictii_raw) - 1)
        planta_raw, boala_raw, scor, e_sanatos = self.rezultat_predictii_raw[indice_ales]
        planta_disp = traduce_planta(planta_raw, self.limba)
        boala_disp = t["sanatos"] if e_sanatos else traduce_boala(boala_raw, self.limba)
        sfat_disp = gaseste_sfat(boala_raw, self.limba)
        procent = round(scor * 100, 1)
        nivel_text = self._nivel_incredere_pdf(procent, t)

        c = pdf_canvas.Canvas(cale_output, pagesize=A4)
        W, H = A4
        raport = ConstructorRaportPDF(c, (W, H), t, self.limba)

        # IMAGINE ANALIZATA
        img_pil = Image.open(self.poza_aleasa).convert("RGB")
        img_pil.thumbnail((int(raport.latime_utila), 210), Image.LANCZOS)
        iw, ih = img_pil.size
        x_poza = (W - iw) / 2

        raport.verifica_spatiu(ih + 34)
        c.setFillColorRGB(*_hex_la_rgb(culori["text_slab"]))
        c.setFont("Helvetica-Bold", 9)
        c.drawString(raport.MARGINE_X, raport.y, t["pdf_sectiune_imagine"])
        raport.y -= 14

        c.setStrokeColorRGB(*_hex_la_rgb(culori["chenar"]))
        c.setLineWidth(1)
        c.rect(x_poza - 3, raport.y - ih - 3, iw + 6, ih + 6, fill=0)

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img_pil.save(tmp.name, "JPEG", quality=88)
        tmp.close()
        c.drawImage(tmp.name, x_poza, raport.y - ih, iw, ih)
        os.unlink(tmp.name)
        raport.y -= (ih + 22)

        #Banner rezultat principal
        raport.verifica_spatiu(60)
        if e_sanatos:
            c.setFillColorRGB(*_hex_la_rgb(culori["ok_fond"]))
        else:
            c.setFillColorRGB(*_hex_la_rgb(culori["rau_fond"]))
        c.roundRect(raport.MARGINE_X, raport.y - 52, raport.latime_utila, 52, 8, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 15)
        c.drawCentredString(W / 2, raport.y - 22,
                            t["planta_sanatoasa"] if e_sanatos else t["boala_detectata"])
        c.setFont("Helvetica", 11)
        c.drawCentredString(W / 2, raport.y - 40, f"{planta_disp}  -  {boala_disp}")
        raport.y -= 64

        if indice_ales != 0:
            raport.paragraf(
                t["pdf_varianta_selectata"].format(loc=t[chei_loc[indice_ales]]),
                font="Helvetica-Oblique", marime=8.5,
                culoare=_hex_la_rgb(culori["text_slab"]), interlinie=11,
            )
            raport.spatiu(6)

        #Detalii analiza
        raport.titlu_sectiune(t["pdf_detalii_titlu"])
        for eticheta, valoare in (
                (t["pdf_planta"], planta_disp),
                (t["pdf_conditie"], boala_disp),
                (t["pdf_incredere"], f"{procent}%"),
                (t["pdf_nivel_incredere"], nivel_text),
        ):
            raport.verifica_spatiu(16)
            c.setFont("Helvetica-Bold", 9.5)
            c.setFillColorRGB(*_hex_la_rgb(culori["text_slab"]))
            c.drawString(raport.MARGINE_X, raport.y, eticheta)
            c.setFont("Helvetica", 9.5)
            c.setFillColorRGB(0.15, 0.15, 0.15)
            c.drawString(raport.MARGINE_X + 150, raport.y, valoare)
            raport.y -= 16
        raport.spatiu(10)

        #Sfat agricol complet, pentru varianta aleasa (nu mai e trunchiat)
        raport.titlu_sectiune(t["pdf_sfat"])
        raport.paragraf(sfat_disp, marime=10, culoare=(0.15, 0.3, 0.18), interlinie=14)
        raport.spatiu(14)

        #Recomandari generale, utile indiferent de rezultat
        raport.titlu_sectiune(t["pdf_recomandari_generale"])
        for sfat_general in SFATURI_GENERALE[self.limba]:
            raport.bullet(sfat_general, marime=9, culoare=(0.3, 0.3, 0.3), interlinie=12)
            raport.spatiu(3)
        raport.spatiu(10)

        #Celelalte variante din Top 3, cu tratamentul propriu fiecareia
        alte = [(i, p) for i, p in enumerate(self.rezultat_predictii_raw) if i != indice_ales]
        if alte:
            raport.titlu_sectiune(t["pdf_alte_titlu"])
            for i, (p_raw, b_raw, sc, sanatos) in alte:
                nume = nume_scurt_predictie(p_raw, b_raw, sanatos, self.limba)
                raport.verifica_spatiu(16)
                c.setFont("Helvetica-Bold", 9.5)
                c.setFillColorRGB(*_hex_la_rgb(culori["btn_gri"]))
                c.drawString(raport.MARGINE_X, raport.y,
                             f"{t[chei_loc[i]]}  -  {nume}  ({round(sc * 100, 1)}%)")
                raport.y -= 14
                sfat_alt = gaseste_sfat(b_raw, self.limba)
                raport.paragraf(sfat_alt, marime=8.5, culoare=(0.42, 0.42, 0.42),
                                interlinie=11, indent=10)
                raport.spatiu(8)

        raport.finalizeaza()
        c.save()

    def afiseaza_dialog_informatii(self):
        """Afiseaza un dialog cu lista de plante si boli lor suportate."""
        t = self.texte()

        dialog = tk.Toplevel(self.fereastra)
        dialog.title(t["dialog_titlu_informatii"])
        dialog.geometry("700x600")
        dialog.configure(bg=culori["fundal"])
        dialog.resizable(True, True)

        # Centrare dialog
        dialog.transient(self.fereastra)
        dialog.grab_set()

        # Container cu scroll
        container = tk.Frame(dialog, bg=culori["fundal"])
        container.pack(fill="both", expand=True, padx=20, pady=20)

        canvas = tk.Canvas(container, bg=culori["fundal"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=culori["fundal"])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Titlu
        tk.Label(
            scrollable_frame,
            text=t["sectiune_plante"],
            font=("Helvetica", 14, "bold"),
            bg=culori["fundal"], fg=culori["verde"]
        ).pack(anchor="w", pady=(0, 15))

        # Grupeaza boiile dupa planta din dictionarul boli
        plante_cu_boli = {}
        for idx, eticheta in boli.items():
            planta_raw, boala_raw = desparte_eticheta(eticheta)
            if planta_raw not in plante_cu_boli:
                plante_cu_boli[planta_raw] = []
            plante_cu_boli[planta_raw].append(boala_raw)

        # Afiseaza toate plantele din NUME_PLANTE
        for planta_raw in sorted(NUME_PLANTE.keys()):
            nume_planta = NUME_PLANTE[planta_raw]
            nume_display = nume_planta["ro"] if self.limba == "ro" else nume_planta["en"]

            # Card pentru planta
            planta_card = tk.Frame(scrollable_frame, bg=culori["card"], padx=15, pady=12)
            planta_card.pack(fill="x", pady=(0, 12))

            # Nume planta
            tk.Label(
                planta_card,
                text=f"🌱 {nume_display}",
                font=("Helvetica", 11, "bold"),
                bg=culori["card"], fg=culori["verde"],
                anchor="w"
            ).pack(fill="x", pady=(0, 8))

            # Lista boli pentru aceasta planta
            if planta_raw in plante_cu_boli:
                boli_plante = sorted(plante_cu_boli[planta_raw])
                for boala_raw in boli_plante:
                    if boala_raw in NUME_BOLI:
                        nume_boala = NUME_BOLI[boala_raw]
                        nume_boala_display = nume_boala["ro"] if self.limba == "ro" else nume_boala["en"]

                        # Verifica daca e healthy
                        if este_sanatos(boala_raw):
                            text = f"  ✅ {nume_boala_display}"
                            fg_color = culori["ok_text"]
                        else:
                            text = f"  ⚠️  {nume_boala_display}"
                            fg_color = culori["rau_text"]

                        tk.Label(
                            planta_card,
                            text=text,
                            font=("Helvetica", 9),
                            bg=culori["card"], fg=fg_color,
                            anchor="w"
                        ).pack(fill="x", pady=2)
            else:
                # Planta nu are boli detectabile, doar healthy
                healthy_text = t["sanatos"] if self.limba == "ro" else "Healthy"
                tk.Label(
                    planta_card,
                    text=f"  ✅ {healthy_text}",
                    font=("Helvetica", 9),
                    bg=culori["card"], fg=culori["ok_text"],
                    anchor="w"
                ).pack(fill="x", pady=2)

        # Buton inchidere
        btn_inchide = ButonRotund(
            scrollable_frame,
            text="Close" if self.limba == "en" else "Inchide",
            la_click=lambda: dialog.destroy(),
            culoare=culori["btn_verde"],
            culoare_hover=culori["btn_hover"],
            latime=120, inaltime=35
        )
        btn_inchide.pack(pady=10)

    def reseteaza_tot(self):
        t = self.texte()
        self.coada_poze = []
        self.index_curent = 0
        self.cache_rezultate = {}
        self.analiza_in_curs = False
        self.poza_aleasa = None
        self.referinta_preview = None
        self.index_sfat_selectat = 0
        self.zona_preview.delete("all")
        hint_text = t["hint_poza"]
        self.text_hint = self.zona_preview.create_text(
            326, 90, text=hint_text,
            font=("Helvetica", 11), fill=culori["text_slab"], width=480, justify="center")
        self.btn_pdf.seteaza_activ(False)
        self.btn_pdf.culoare = culori["btn_gri"]
        self.btn_pdf.deseneaza(culori["btn_gri"])

        self.stare_afisare = "asteapta"
        self._afiseaza_asteapta_scanare()
        self.rezultat_predictii_raw = None
        self.actualizeaza_navigare()


# MAIN
if __name__ == "__main__":
    fereastra_principala = tk.Tk()
    AplicatieLeafLens(fereastra_principala)
    fereastra_principala.mainloop()
