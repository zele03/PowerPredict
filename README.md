# PowerPredict

PowerPredict je studentski projekat za predikciju potrošnje električne energije
domaćinstva na osnovu Household Power Consumption dataset-a.

Sistem predviđa potrošnju energije po satima i evaluira modele nad validation i
test skupovima pomoću metrika MAE, RMSE, MAPE, sMAPE i WAPE.

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

GRU model se ne trenira u osnovnom pokretanju.

## GRU Model

GRU model se ne pokrece podrazumevano zato što zahteva PyTorch okruženje i može
da koristi CUDA/GPU podrsku.

Nakon što se završi osnovni pipeline, GRU se trenira eksplicitno:

```bash
uv run main.py gru-pipeline
```

Detaljno uputstvo za GRU setup i objasnjenje komandi nalazi se u:

```text
docs/gru_training_instructions.md
```

## Glavne Komande

```bash
uv run main.py                    # osnovni baseline workflow
uv run main.py gru-pipeline       # trenira GRU, poredi rezultate, crta grafike...
```

## Dokumentacija

Detaljna projektna dokumentacija:

```text
docs/documentation.md
```
