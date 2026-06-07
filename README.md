# PowerPredict

PowerPredict je studentski projekat za predikciju satne potrošnje električne
energije domaćinstva na osnovu [Household Power Consumption dataset](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set) dataset-a. Projekat
prati kompletan tok rada: čišćenje sirovih podataka, kreiranje vremenskih i lag
feature-a, podelu na train/validation/test skupove, baseline model, treniranje GRU iLSTM rekurentnih neuronskih mreža, korišćenjem grid search mehanizma i poređenje dobijenih rezltata.

Cilj projekta je da se proveri koliko modeli koji koriste prethodne sate kao
sekvencu mogu da unaprede predikciju u odnosu na jednostavan `lag_24h` baseline.
Sistem evaluira modele nad validation i test skupovima pomoću metrika MAE, RMSE,
MAPE, sMAPE i WAPE, a zatim generiše tabele, predikcije i grafike za lakše
poređenje rezultata.

## Struktura Projekta

```text
main.py
src/
data/
docs/
models/
```

Glavni koraci workflow-a:

```text
src/step01_preprocessing.py
src/step02_features.py
src/step03_split.py
src/step04_baseline_model.py
src/step05_baseline_visualization.py
src/step06_gru_model.py
src/step07_gru_visualization.py
src/step08_lstm_model.py
src/step09_lstm_visualization.py
```

## Instalacija

Klonirati repozitorijum:

```bash
git clone https://github.com/zele03/PowerPredict.git
cd PowerPredict
```

Kreirati virtuelno okruženje i instalirati zavisnosti pomoću `uv`:

```bash
uv sync
```

## Osnovno Pokretanje

Pokrenuti standardni pipeline:

```bash
uv run main.py
```

Ova komanda pokreće preprocessing, feature engineering,
train/validation/test split, baseline evaluaciju i baseline vizualizacije.

GRU i LSTM modeli se ne treniraju u osnovnom pokretanju.

## GRU Model

GRU model se ne pokreće podrazumevano zato što zahteva PyTorch okruženje i može
da koristi CUDA/GPU podršku. Model koristi sekvence prethodnih sati i grid search
nad hiperparametrima kao što su dužina sekvence, broj skrivenih jedinica, broj
slojeva, dropout i learning rate.

Nakon što se završi osnovni pipeline, GRU se trenira eksplicitno:

```bash
uv run main.py gru-pipeline
```

Najbolji sačuvani GRU model bira se po validation MAE.

## LSTM Model

LSTM model je dodat kao drugi rekurentni pristup za poređenje sa baseline i GRU
modelom. Kao i GRU, koristi PyTorch, sekvence prethodnih sati i isti skup
evaluacionih metrika, ali LSTM arhitektura kroz memorijsku ćeliju može bolje da
zadrži duže zavisnosti u vremenskoj seriji.

Nakon osnovnog pipeline-a, LSTM se trenira eksplicitno:

```bash
uv run main.py lstm-pipeline
```

Najbolji sačuvani LSTM model bira se po validation MAE

## Detaljno uputstvo za GRU i LSTM setup i objašnjenje komandi nalazi se u:

```text
docs/gru_lstm_training_instructions.md
```

## Glavne Komande

```bash
uv run main.py                    # osnovni baseline workflow
uv run main.py gru-pipeline       # trenira GRU, poredi rezultate, crta grafike
uv run main.py lstm-pipeline      # trenira LSTM, poredi rezultate, crta grafike
```

## Dokumentacija

Detaljna projektna dokumentacija:

```text
docs/documentation.md
```

## Vodič kroz projekat

```text
docs/powerpredict_walkthrough.ipynb
```
