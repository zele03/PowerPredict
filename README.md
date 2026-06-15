# PowerPredict

PowerPredict je studentski projekat za predikciju satne potrošnje električne
energije domaćinstva na osnovu
[Household Power Consumption dataset-a](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set).

Projekat prati kompletan tok rada nad vremenskom serijom: preprocessing sirovih
podataka, feature engineering, podelu na train/validation/test skupove, baseline
model, treniranje neuronskih modela GRU i LSTM, kao i treniranje klasičnih tree
regresionih modela - Random Forest Regressor i Gradient Boosting Regressor.

Glavni cilj projekta je da se uporedi više pristupa za predikciju potrošnje:

- jednostavan `lag_24h` baseline,
- sekvencijalni PyTorch modeli GRU i LSTM,
- tabularni scikit-learn modeli Random Forest i Gradient Boosting.

Svi modeli se evaluiraju istim metrikama: MAE, RMSE, MAPE, sMAPE i WAPE. Projekat
generiše CSV izveštaje, predikcije, sačuvane modele, feature importance tabele i
grafike za poređenje rezultata.

## Struktura Projekta

```text
PowerPredict/
|-- main.py
|-- pyproject.toml
|-- README.md
|-- data/
|   |-- graphs/
|   |-- logs/
|   |-- predictions/
|   |-- processed/
|   `-- raw/
|-- docs/
|   |-- documentation.md
|   `-- powerpredict_walkthrough.ipynb
|-- models/
|   |-- gradient_boosting/
|   |-- gru/
|   |-- lstm/
|   `-- random_forest/
`-- src/
    |-- step01_preprocessing.py
    |-- step02_features.py
    |-- step03_split.py
    |-- step04_baseline_model.py
    |-- step05_baseline_visualization.py
    |-- step06_gru_model.py
    |-- step07_gru_visualization.py
    |-- step08_lstm_model.py
    |-- step09_lstm_visualization.py
    |-- step10_tree_models.py
    `-- step11_tree_visualization.py
```

## Workflow

Projekat je podeljen u više jasnih koraka:

| Korak | Fajl | Opis |
| --- | --- | --- |
| 01 | `src/step01_preprocessing.py` | Čišćenje i priprema sirovog dataset-a |
| 02 | `src/step02_features.py` | Kreiranje vremenskih, lag i rolling feature-a |
| 03 | `src/step03_split.py` | Podela na train, validation i test skupove |
| 04 | `src/step04_baseline_model.py` | Baseline predikcija pomoću `lag_24h` |
| 05 | `src/step05_baseline_visualization.py` | Grafici za baseline model |
| 06 | `src/step06_gru_model.py` | Treniranje GRU modela |
| 07 | `src/step07_gru_visualization.py` | GRU grafici i baseline poređenje |
| 08 | `src/step08_lstm_model.py` | Treniranje LSTM modela |
| 09 | `src/step09_lstm_visualization.py` | LSTM grafici i baseline poređenje |
| 10 | `src/step10_tree_models.py` | Random Forest i Gradient Boosting trening |
| 11 | `src/step11_tree_visualization.py` | Tree model grafici i poređenje svih modela |

## Instalacija

Klonirati repozitorijum:

```bash
git clone https://github.com/zele03/PowerPredict.git
cd PowerPredict
```

Instalirati zavisnosti pomoću `uv`:

```bash
uv sync
```

Za GRU i LSTM pipeline potreban je i PyTorch. Ako nije instaliran u okruženju,
instalirati odgovarajuću verziju prema CPU/CUDA konfiguraciji računara.

## Pokretanje

Osnovni pipeline:

```bash
uv run main.py
```

Ova komanda pokreće:

- preprocessing,
- feature engineering,
- train/validation/test split,
- baseline model,
- baseline vizualizacije.

GRU, LSTM i tree modeli se pokreću odvojeno, jer treniranje može trajati duže i
generiše dodatne modele, predikcije, izveštaje i grafike.

## Glavne Komande

```bash
uv run main.py                    # osnovni baseline workflow
uv run main.py gru-pipeline       # GRU trening, izveštaji, predikcije i grafici
uv run main.py lstm-pipeline      # LSTM trening, izveštaji, predikcije i grafici
uv run main.py tree-pipeline      # RF/GB trening, izveštaji, predikcije i grafi
```

Preporučeni redosled pokretanja:

```bash
uv run main.py
uv run main.py gru-pipeline
uv run main.py lstm-pipeline
uv run main.py tree-pipeline
```

## Modeli

### Baseline

Baseline model koristi vrednost potrošnje iz istog sata prethodnog dana
(`lag_24h`). Ovaj model je jednostavna referentna tačka: ako napredniji modeli ne
nadmaše baseline, onda dodatna kompleksnost nije opravdana.

### GRU i LSTM

GRU i LSTM su rekurentni neuronski modeli implementirani u PyTorch-u. Oni koriste
sekvence prethodnih sati i na osnovu tog vremenskog konteksta predviđaju
potrošnju za naredni trenutak.

Ovi modeli koriste grid search nad parametrima kao što su:

- dužina sekvence,
- broj skrivenih jedinica,
- broj slojeva,
- dropout,
- learning rate,
- batch size.

Najbolji sačuvani model bira se po validation MAE, a test skup se koristi za
finalnu procenu.

### Random Forest i Gradient Boosting

Random Forest i Gradient Boosting su tree-based regresioni modeli implementirani
u scikit-learn-u.

Random Forest trenira više nezavisnih stabala odluke i konačnu predikciju dobija
kao prosek njihovih predikcija. U ovom projektu koristi bootstrap uzorkovanje
redova, dok pri splitovima koristi sve dostupne feature-e.

Gradient Boosting trenira stabla sekvencijalno. Prvo polazi od jednostavne
predikcije, a svako sledeće stablo uči preostalu grešku prethodnog modela. Nova
predikcija dobija se dodavanjem male korekcije kontrolisane parametrom
`learning_rate`.

Tree modeli ne dobijaju sekvencu kao GRU/LSTM. Umesto toga koriste jedan red
tabularnih feature-a, ali taj red već sadrži vremenske i istorijske informacije
kroz lag i rolling kolone.

## Feature Engineering

Nakon preprocessinga kreiraju se feature-i koji pomažu modelima da razumeju
vremenske obrasce potrošnje:

- vremenski feature-i, kao što su sat, dan u nedelji i mesec,
- lag feature-i, kao što su potrošnja pre 1h, 24h i 168h,
- rolling statistike nad prethodnim vrednostima,
- indikatori za vikend/praznik ako postoje u dataset-u.

## Rezultati i Izlazi

Pipeline generiše sledeće tipove fajlova:

```text
data/logs/          CSV izveštaji i metrike
data/predictions/   predikcije po modelima
data/graphs/        vizualizacije rezultata
models/             sačuvani modeli
```

Za tree modele se dodatno generišu `feature_importance.csv` fajlovi, koji pokazuju
koji feature-i su najviše doprineli odlukama modela.

Grafici za poređenje svih modela nalaze se u:

```text
data/graphs/all_models_compare_graphs/
```

## Git Napomene

Sirovi podaci, obrađeni dataset-i i veliki generisani modeli nisu
praćeni kroz Git. Random Forest modeli su posebno veliki zato što čuvaju veliki
broj stabala odluke, pa je folder `models/random_forest/` dodat u `.gitignore`.

Manji modeli, izveštaji i grafici ostavljeni su u repozitorijumu u koliko budu potrebni za detaljniju analizu.

## Dokumentacija

Detaljnija projektna dokumentacija nalazi se u:

```text
docs/documentation.md
```

Interaktivan pregled projekta nalazi se u:

```text
docs/powerpredict_walkthrough.ipynb
```
