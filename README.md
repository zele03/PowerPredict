# PowerPredict

PowerPredict je studentski projekat za predikciju potrosnje elektricne energije
domacinstva na osnovu Household Power Consumption dataset-a.

Sistem predvidja potrosnju energije po satima i evaluira modele nad validation i
test skupovima pomocu metrika WAPE, MAE, RMSE, MAPE i sMAPE.

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

Kreirati virtuelno okruzenje i instalirati zavisnosti pomocu `uv`:

```bash
uv sync
```

## Osnovno Pokretanje

Pokrenuti standardni pipeline:

```bash
uv run main.py
```

Ova komanda pokrece preprocessing, feature engineering,
train/validation/test split, baseline evaluaciju i baseline vizualizacije.

GRU model se ne trenira u osnovnom pokretanju.

## GRU Model

GRU model se ne pokrece podrazumevano zato sto zahteva PyTorch okruzenje i moze
da koristi CUDA/GPU podrsku.

Nakon sto se zavrsi osnovni pipeline, GRU se trenira eksplicitno:

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
