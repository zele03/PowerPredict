# PowerPredict – Dokumentacija

**Autori:** Luka Zelembaba, Dušan Đurić  
**Datum:** jun 2026

## 1. Uvod
Projekat je urađen u okviru predmeta **Inteligentni sistemi** na Fakultetu Tehničkih Nauka u Novom Sadu.

Cilj projekta je predikcija vremenskih serija – potrošnja električne energije – pomoću neuronskih mreža.

Rad obuhvata pripremu i čišćenje podataka i njihovu transformaciju u oblik pogodan za analizu. Razvoj modela za predikciju, koji obuhvata izbor arhitekture neuronske mreže, trening nad pripremljenim podacima, evaluaciju performansi modela u odnosu na definisane metrike, poređenje rezultata iste arhitekture u drugačijim konfiguracijama.

## 2. Tehnički aspekti projekta

Projekat je implementiran u programskom jeziku **Python** i organizovan kao sekvencijalni pipeline kroz fajlove `src/step01_...` do `src/step07_...`. Osnovni workflow pokreće se iz `main.py` i obuhvata preprocessing, feature engineering, vremensku podelu podataka, baseline evaluaciju i generisanje baseline grafova. GRU deo se pokreće zasebno, zato što zahteva PyTorch okruženje i vremenski je zahtevniji.

Za upravljanje okruženjem i zavisnostima koristi se **uv**, dok su glavne biblioteke navedene u `pyproject.toml`. U osnovnom delu projekta koriste se:

- **pandas** za učitavanje, obradu, resamplovanje i transformaciju vremenskih serija,
- **numpy** za numeričke operacije i pripremu podataka za evaluaciju,
- **matplotlib** za generisanje grafičkih prikaza rezultata,
- **tqdm** za prikaz napretka tokom dužih izvršavanja,
- **argparse** za izbor workflow-a pri pokretanju aplikacije.

Za GRU model koristi se **PyTorch**, ali se on učitava samo pri pokretanju GRU pipeline-a i nije deo osnovnog pokretanja projekta. Ovakva organizacija omogućava da se preprocessing i baseline analiza izvrše bez potrebe za instaliranim PyTorch/CUDA okruženjem, dok se neuronski model trenira eksplicitno komandom opisanom u posebnom uputstvu.

Rezultati obrade, modeli, predikcije, log fajlovi i grafovi čuvaju se u folderima `data/graphs`, `data/logs`, `data/predictions` i `models/`, što omogućava proverljiv i ponovljiv tok rada kroz sve faze projekta.

## 3. Dataset
- **Izvor podataka:** Individual Household Electric Power Consumption dataset (UCI Machine Learning Repository) containing 2075259 measurements gathered in a house located in Sceaux (7km of Paris, France)  
  - Originalno objavljen na [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)  
  - Dostupan i na [Kaggle](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)  
- **Struktura:** originalni podaci obuhvataju minutna merenja potrošnje električne energije, ali i druge prateće vrednosti vezane za rad kućnog elektro‑sistema.
- Kratak opis bitnih karakteristika dataset-a dostupan na: `data/logs/dataset_report.csv`.
- **Preprocessing koraci:**
  - spajanje datuma i vremena u jednu `datetime` kolonu radi formiranja jedinstvene vremenske oznake za lakšu obradu vremenskih serija,
  - brisanje dana sa više od 60 nedostajućih ili praznih redova (bez obavljenih merenja),
    - kod dana gde se takvi slučajevi javljaju u manjem broju od 60, obrisani su samo ti redovi,
  - resamplovanje minutnih podataka u satne proseke, 
  - odabir relevantnih kolona (`Global_active_power`, `Global_reactive_power`, `Global_intensity`),
  - detalji o celom procesu preprocessinga dostupni su u log fajlu: `data/logs/preprocessing_report.csv`.
- **Feature Engineering koraci:**
  - dodavanje target kolone: `Energy_kWh` (identična numerička vrednost kao `Global_active_power`, ali eksplicitno definisana kao ciljna promenljiva),
  - dodavanje vremenskih atributa:
    - `hour` – sat u danu,
    - `day_of_week` – dan u nedelji (0 = ponedeljak, 6 = nedelja),
    - `month` – mesec u godini,
    - `is_weekend` – indikator vikenda (subota/nedelja),
    - `is_night` – indikator noćnih sati (0–6h).
  - dodavanje lag atributa:
    - `lag_1h` – vrednost potrošnje iz prethodnog sata,
    - `lag_24h` – vrednost iz istog sata prethodnog dana,
    - `lag_168h` – vrednost iz istog sata prethodne nedelje.
  - dodavanje rolling statistika:
    - `rolling_mean_24h` – pokretni prosek potrošnje u poslednja 24 sata,
    - `rolling_std_7d` – standardna devijacija potrošnje u poslednjih 7 dana (168 sati).
  - dodavanje holiday atributa:
    - `is_holiday` – binarni indikator praznika, na osnovu francuskih državnih praznika

  - **Napomena:** Prilikom generisanja feature‑a uklonjeni su prvi redovi bez dovoljno istorije za lag/rolling statistike. Sve numeričke kolone zaokružene su na četiri decimale radi konzistentnosti.

- **Train, validation i test split koraci:**
    - podela podataka izvršena po vremenskoj osi (time‑based split), bez random shuffle‑a,
  - **train skup:** podaci od 2006‑12‑24 do 2009‑9‑14 (≈70% ukupnog seta),
  - **validation skup:** podaci od 2009‑09‑14 do 2010‑04‑18 (≈15% ukupnog seta),
  - **test skup:** podaci od 2010‑04‑18 do 2010‑11‑25 (≈15% ukupnog seta),
  - detalji dostupni u log fajlu: `data/logs/split_report.csv`

## 4. Baseline model

- **Opis:**
  - Korišćen je `lag_24h` – predikcija potrošnje na osnovu vrednosti iz istog sata prethodnog dana, najjednostavniji model.
  - Ovaj pristup služi kao referentna tačka za poređenje sa kompleksnijim modelima.

- **Izbor metrika:**
  - **Glavna metrika:**
    - **MAE (Mean Absolute Error):** prosečna greška izražena u kWh.

  - **Pomoćne metrike:**
    - **RMSE (Root Mean Squared Error):** kažnjava velike greške, izraženo u kWh.
    - **MAPE (Mean Absolute Percentage Error):** procentualna greška, ali nepouzdana zbog velikog broja malih vrednosti (oko 40% dataset‑a <0.5 kWh).
    - **sMAPE (Symmetric MAPE):** simetrična verzija MAPE, stabilnija kod malih vrednosti.
    - **WAPE (Weighted Absolute Percentage Error):** meri ukupnu grešku u odnosu na ukupnu potrošnju, izraženo u procentima
    - **Hourly MAE, RMSE, MAPE, sMAPE, WAPE:** računati radi detaljne analize performansi po satima.

- **Rezultati**

    | Skup       | MAE (kWh) | RMSE (kWh) | MAPE (%) | sMAPE (%) | WAPE (%) |
    |------------|-----------|------------|----------|-----------|----------|
    | Validation | 0.66      | 0.96       | 74.79    | 52.07     | 53.05    |
    | Test       | 0.51      | 0.76       | 66.80    | 49.29     | 53.47    |

  - **Grafički prikaz** rezultata dostupan u: `data\graphs\baseline_graphs`.

- **Analiza po satima (test skup)**

    | Kategorija       | Sati                 | MAE (kWh)       |
    |------------------|----------------------|-----------------|
    | Najgori sati     | 14h–18h, 23h         | 0.6             |
    | Najbolji sati    | 05h-10h              | 0.2-0.5         |

- **Output fajlovi:**
  - Baseline report sa summary i hourly metrikama čuva se u:

    ```text
    data/logs/baseline_report.csv
    ```

  - Grafički prikazi baseline rezultata čuvaju se u:

    ```text
    data/graphs/baseline_graphs/
    ```

  - Baseline grafovi obuhvataju hourly MAE, hourly WAPE, scatter odnos stvarnih i predviđenih vrednosti i distribuciju apsolutne greške na test skupu.

- **Zaključci:**
  - Baseline `lag_24h` pokazuje slabosti u popodnevnim satima (14h–18h), posebno oko 15h–17h, gde se potrošnja značajno razlikuje od prethodnog dana.
  - Jutarnji sati (7h–10h) su stabilniji i daju bolje rezultate.
  - Loš rezultat u 23h ukazuje na varijabilnost prelaza iz večernje aktivnosti u noć, što `lag_24h` ne uspeva da uhvati.

## 5. GRU model

- **Opis:**
  - Za predikciju potrošnje električne energije korišćen je GRU (Gated Recurrent Unit), rekurentna neuronska mreža pogodna za vremenske serije.
  - Za razliku od baseline modela, koji koristi samo vrednost `lag_24h`, GRU dobija sekvencu prethodnih sati i iz nje uči vremenske obrasce potrošnje.
  - Model za jednu predikciju koristi prethodnih `sequence_length` sati.
  - Ulazni skup atributa obuhvata vremenske, lag, rolling i holiday feature-e. Iz ulaza se izbacuju `datetime` i `Global_active_power`, kako model ne bi koristio ciljnu vrednost za isti sat.

- **Priprema podataka za GRU:**
  - Train, validation i test skup se učitavaju iz `data/processed/split`.
  - Skaliranje se fituje samo nad train skupom, kako validation i test skup ne bi uticali na parametre skaliranja.
  - Za svaki target sat formira se sekvenca prethodnih `sequence_length` sati.
  - Sekvence se prihvataju samo ako su vremenski kontinualne, odnosno ako su svi sati u prozoru uzastopni.
  - Validation i test sekvence koriste kontekst iz prethodnog skupa, kako bi prvi redovi imali dostupnu istoriju bez narušavanja vremenskog redosleda.

- **Arhitektura modela:**
  - GRU sloj prima sekvencu feature-a oblika:

    ```text
    batch_size x sequence_length x broj_featurea
    ```

  - Izlaz poslednjeg vremenskog koraka prosleđuje se kroz linearni sloj koji daje jednu numeričku predikciju potrošnje `Energy_kWh`.
  - Dropout se primenjuje između GRU slojeva kada model ima više od jednog sloja.
  - Kao loss funkcija koristi se MSELoss, a optimizator je Adam.

- **Hiperparametri:**
  - GRU konfiguracije se generišu grid search pristupom iz prostora hiperparametara definisanog u `GRU_GRID_SEARCH_SPACE` u `src/step06_gru_model.py`.
  - Dobijene konfiguracije se čuvaju u `HYPERPARAMETER_CONFIGS` i treniraju se jedna po jedna.
  - Svaka konfiguracija ima jedinstven naziv, koji se koristi za čuvanje modela, predikcija, reportova i grafova.
  - Poređeni hiperparametri:
    - `hidden_size` – veličina skrivene memorije GRU sloja,
    - `num_layers` – broj naslaganih GRU slojeva,
    - `dropout` – regularizacija radi smanjenja overfitting-a,
    - `learning_rate` – brzina kojom optimizer menja težine,
    - `batch_size` – broj sekvenci koje model obrađuje u jednom trening koraku,
    - `sequence_length` – broj prethodnih sati koji se koriste za jednu predikciju.
  - Konfiguracije kod kojih je `num_layers = 1` i `dropout > 0` se ne treniraju, jer PyTorch dropout u GRU sloju ima efekat samo između više GRU slojeva.

- **Trening i izbor modela:**
  - Svaki GRU model trenira se samo na train skupu.
  - Validation skup se koristi za early stopping i izbor najbolje konfiguracije hiperparametara.
  - Najbolji GRU model bira se po `validation_MAE`.
  - Test skup se koristi za finalnu evaluaciju, grafički prikaz rezultata i poređenje najboljeg GRU modela sa baseline modelom.
  - Ovakva podela sprečava da se hiperparametri biraju direktno prema test skupu.

- **Rezultati GRU konfiguracija**

    | Model | sequence length | hidden | layers | dropout | learning rate | Validation MAE | Test MAE | Test RMSE | Test MAPE | Test sMAPE | Test WAPE (%) |
    |-------|-----------------|--------|--------|---------|---------------|----------------|----------|-----------|-----------|------------|---------------|
    | `gru_seq24_h64_l1_d0p0_lr0p0005_bs64` | 24 | 64 | 1 | 0.0 | 0.0005 | 0.3714 | 0.3291 | 0.4791 | 41.89 | 35.11 | 34.30 |

  - Najbolji model prema `validation_MAE` koristi sekvencu od 24 prethodna sata, `hidden_size = 64`, jedan GRU sloj i learning rate `0.0005`.

- **Poređenje sa baseline modelom:**
  - Baseline model na test skupu ima MAE oko 0.51 kWh, RMSE oko 0.76 kWh i WAPE oko 53.47%.
  - Najbolji GRU model na test skupu ima MAE 0.3291 kWh, RMSE 0.4791 kWh i WAPE 34.30%.
  - U odnosu na baseline, najbolji GRU smanjuje MAE za oko 35.76%, RMSE za oko 37.05% i WAPE za oko 35.84%.
  - Poređenje se čuva u `data/logs/compare_logs/baseline_gru_compare_report.csv`.
  - GRU bolje koristi širi vremenski kontekst i dodatne feature-e u odnosu na jednostavni `lag_24h` baseline.
  - Najveće koristi se očekuju u satima gde se potrošnja ne ponavlja dovoljno dobro po jednostavnom pravilu "isti sat prethodnog dana".

- **Output fajlovi:**
  - Trenirani PyTorch modeli čuvaju se u:

    ```text
    models/{model_name}.pt
    ```

  - Predikcije GRU modela čuvaju se u:

    ```text
    data/predictions/gru_predictions/
    ```

  - Reportovi, training history i summary fajlovi čuvaju se u:

    ```text
    data/logs/gru_logs/
    ```

  - Poređenje najboljeg GRU modela sa baseline modelom čuva se u:

    ```text
    data/logs/compare_logs/baseline_gru_compare_report.csv
    ```

  - Grafički prikazi dostupni su u:

    ```text
    data/graphs/gru_graphs/
    data/graphs/baseline_gru_compare_graphs/
    ```

  - Grafici promene greške kroz epohe za train i validation skup čuvaju se za svaku GRU konfiguraciju u:

    ```text
    data/graphs/gru_graphs/{model_name}_training_history.png
    ```

- **Zaključci:**
  - Sekvencijalni pristup je pogodan za ovaj problem, jer model koristi obrazac iz prethodnih sati umesto samo jedne prethodne vrednosti.
  - Najbolja konfiguracija za izbor hiperparametara određuje se prema validation MAE, dok se test metrike koriste za finalno izveštavanje i poređenje sa baseline modelom.
  - Grid search je omogućio sistematično poređenje više GRU konfiguracija, pri čemu je najbolji rezultat na validation skupu ostvarila konfiguracija `gru_seq24_h64_l1_d0p0_lr0p0005_bs64`.

