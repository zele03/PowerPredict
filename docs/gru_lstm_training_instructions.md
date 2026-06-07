# Uputstvo Za Treniranje GRU i LSTM Modela

Ovaj dokument objašnjava kako se pokreću GRU i LSTM delovi projekta nakon
kloniranja ili pull-ovanja projekta.

Osnovni workflow ne trenira GRU i LSTM modele, zato što trening rekurentnih
neuronskih mreža zahteva PyTorch okruženje i može da traje znatno duže od
preprocessinga, baseline evaluacije i baseline vizualizacije. Oba pipeline-a
treniraju više modela sa različitim hiperparametrima jedan za drugim, odnosno
koriste grid search metod podešavanja hiperparametara.

## 1. Pokrenuti Osnovni Pipeline

Prvo pokrenuti:

```bash
uv run main.py
```

Ovo pokreće:

```text
step01 preprocessing
step02 feature engineering
step03 vremenski train/validation/test split
step04 baseline model
step05 baseline vizualizacije
```

Pre GRU ili LSTM treninga treba da postoje:

```text
data/processed/split/train.csv
data/processed/split/validation.csv
data/processed/split/test.csv
data/logs/baseline_report.csv
data/graphs/baseline_graphs/
```

## 2. Pripremiti PyTorch

GRU i LSTM trening zahtevaju PyTorch. GPU/CUDA podrška je preporučena, ali modeli
mogu da rade i na CPU-u.

Provera PyTorch instalacije:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Ako `torch.cuda.is_available()` ispiše `True`, modeli će koristiti GPU.

Ako PyTorch nije instaliran ili instalirana verzija ne odgovara CUDA setup-u na
toj mašini, instalirati PyTorch zvaničnom komandom za konkretno okruženje:

```text
https://pytorch.org/get-started/locally/
```

## 3. Pokrenuti GRU Pipeline

**Upozorenje:** `gru-pipeline` pokreće grid search i trenira veliki broj GRU
modela jedan za drugim. Komandu ne treba pokretati rutinski, jer izvršavanje može
trajati dugo, posebno na CPU-u ili bez CUDA/GPU podrške.

Kada se osnovni pipeline završi, pokrenuti:

```bash
uv run main.py gru-pipeline
```

Ova komanda trenira sve GRU konfiguracija iz HYPERPARAMETER_CONFIGS, poredi rezultate i pravi odgovarajuće log file-ove i grafike.

GRU modeli se treniraju jedan po jedan, redom iz liste
`HYPERPARAMETER_CONFIGS` u `src/step06_gru_model.py`.

GRU arhitektura koristi gated recurrent unit slojeve za učenje zavisnosti kroz
vremenske sekvence. U ovom projektu model dobija sekvence prethodnih sati i
predviđa potrošnju za naredni sat, uz poređenje različitih dužina sekvence,
veličina skrivenog stanja, broja slojeva, dropout vrednosti i learning rate-a.

## 4. Pokrenuti LSTM Pipeline

**Upozorenje:** `lstm-pipeline` takođe pokreće grid search i trenira veliki broj
LSTM modela jedan za drugim. Zbog toga može trajati dugo, posebno na CPU-u ili
bez CUDA/GPU podrške.

Kada se osnovni pipeline završi, pokrenuti:

```bash
uv run main.py lstm-pipeline
```

Ova komanda trenira sve GRU konfiguracija iz HYPERPARAMETER_CONFIGS, poredi rezultate i pravi odgovarajuće log file-ove i grafike.

LSTM modeli se treniraju jedan po jedan, redom iz liste
`HYPERPARAMETER_CONFIGS` u `src/step08_lstm_model.py`.

LSTM arhitektura koristi long short-term memory slojeve sa memorijskom ćelijom,
što joj omogućava da bolje zadrži duže zavisnosti u vremenskoj seriji. U ovom
projektu LSTM koristi isti pripremljeni skup feature-a i isti princip sekvenci
kao GRU, pa rezultati mogu direktno da se porede sa baseline i GRU modelima.

## 5. Kako Se Bira Najbolji GRU ili LSTM

Svaki GRU i LSTM model se trenira na train skupu.

Validation skup se koristi za:

```text
early stopping
izbor najboljih hiperparametara
```

Najbolji GRU i najbolji LSTM se biraju po:

```text
validation_MAE
```

Test skup se koristi za:

```text
finalnu evaluaciju
GRU/LSTM grafove
baseline vs najbolji GRU/LSTM compare report
baseline vs najbolji GRU/LSTM uporedne grafove
```

Ovo znači da se hiperparametri ne biraju po test skupu, već se test koristi kao
finalna provera modela.

## 6. Glavne Komande

```bash
uv run main.py               # osnovni baseline workflow
uv run main.py gru-pipeline  # trenira sve GRU modele, poredi ih i pravi GRU grafove
uv run main.py lstm-pipeline # trenira sve LSTM modele, poredi ih i pravi LSTM grafove
```
