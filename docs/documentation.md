# PowerPredict – Dokumentacija

**Autori:** Luka Zelembaba, Dušan Đurić  
**Datum:** jun 2026

## 1. Uvod
Projekat je urađen u okviru predmeta **Inteligentni sistemi** na Fakultetu Tehničkih Nauka u Novom Sadu.

Cilj projekta je predikcija vremenskih serija – potrošnja električne energije – pomoću neuronskih mreža.

Rad obuhvata pripremu i čišćenje podataka i njihovu transformaciju u oblik pogodan za analizu. Razvoj modela za predikciju, koji obuhvata izbor arhitekture neuronske mreže, trening nad pripremljenim podacima, evaluaciju performansi modela u odnosu na definisane metrike, poređenje rezultata istih arhitektura u drugačijim konfiguracijama, poređenje rezultata različitih arhitektura.

## 2. Dataset
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
  - **test skup:** podaci od 2011‑01‑01 do 2011‑12‑31 (≈15% ukupnog seta),
  - detalji dostupni u log fajlu: `data/logs/split_report.csv`

## 3. Baseline model

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

- **Zaključci:**
  - Baseline `lag_24h` pokazuje slabosti u popodnevnim satima (14h–18h), posebno oko 15h–17h, gde se potrošnja značajno razlikuje od prethodnog dana.
  - Jutarnji sati (7h–10h) su stabilniji i daju bolje rezultate.
  - Loš rezultat u 23h ukazuje na varijabilnost prelaza iz večernje aktivnosti u noć, što `lag_24h` ne uspeva da uhvati.

## 4. GRU model

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
  - Trenirano je više GRU konfiguracija, definisanih u `HYPERPARAMETER_CONFIGS` u `src/step06_gru_model.py`.
  - Svaka konfiguracija ima jedinstven naziv, koji se koristi za čuvanje modela, predikcija, reportova i grafova.
  - Poređeni hiperparametri:
    - `hidden_size` – veličina skrivene memorije GRU sloja,
    - `num_layers` – broj naslaganih GRU slojeva,
    - `dropout` – regularizacija radi smanjenja overfitting-a,
    - `learning_rate` – brzina kojom optimizer menja težine,
    - `batch_size` – broj sekvenci koje model obrađuje u jednom trening koraku,
    - `sequence_length` – broj prethodnih sati koji se koriste za jednu predikciju.

- **Trening i izbor modela:**
  - Svaki GRU model trenira se samo na train skupu.
  - Validation skup se koristi za early stopping i izbor najbolje konfiguracije hiperparametara.
  - Najbolji GRU model bira se po `validation_MAE`.
  - Test skup se koristi za finalnu evaluaciju, grafički prikaz rezultata i poređenje najboljeg GRU modela sa baseline modelom.
  - Ovakva podela sprečava da se hiperparametri biraju direktno prema test skupu.

- **Rezultati GRU konfiguracija**

    | Model | hidden | layers | dropout | learning rate | Validation MAE | Test MAE | Test RMSE | Test MAPE | Test sMAPE | Test WAPE (%) |
    |-------|--------|--------|---------|---------------|----------------|----------|-----------|-----------|------------|---------------|
    | `gru_seq24_h32_l1_d0_lr0.001_bs64` | 32 | 1 | 0.0 | 0.001 | 0.3745 | 0.3372 | 0.4810 | 45.78 | 36.40 | 35.15 |
    | `gru_seq24_h64_l2_d0.2_lr0.001_bs64` | 64 | 2 | 0.2 | 0.001 | 0.3775 | 0.3411 | 0.4856 | 46.39 | 36.69 | 35.56 |
    | `gru_seq24_h128_l2_d0.3_lr0.0005_bs64` | 128 | 2 | 0.3 | 0.0005 | 0.3785 | 0.3354 | 0.4819 | 43.66 | 36.18 | 34.96 |

  - Kolona `Validation MAE` koristi se za izbor najbolje konfiguracije hiperparametara.
  - Test metrike prikazuju finalne performanse modela nakon izbora konfiguracije.
  - Na prikazanim rezultatima najmanji `Validation MAE` ima konfiguracija sa `hidden_size = 32` i jednim GRU slojem, dok konfiguracija sa `hidden_size = 128` daje najbolji test MAE/WAPE među prikazanim modelima.

- **Poređenje sa baseline modelom:**
  - Baseline model na test skupu ima MAE oko 0.51 kWh, RMSE oko 0.76 kWh i WAPE oko 53.47%.
  - GRU modeli na test skupu imaju MAE oko 0.34 kWh i RMSE oko 0.48 kWh i  WAPE oko 35%.
  - GRU značajno smanjuje grešku u odnosu na `lag_24h`, što pokazuje da model uspešnije koristi širi vremenski kontekst i dodatne feature-e.
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

- **Zaključci:**
  - GRU model značajno nadmašuje baseline `lag_24h` model na test skupu.
  - Sekvencijalni pristup je pogodniji za ovaj problem, jer model koristi obrazac iz prethodna 24 sata umesto samo jedne prethodne vrednosti.
  - Najbolja konfiguracija za izbor hiperparametara određuje se prema validation MAE, dok se test metrike koriste za finalno izveštavanje i poređenje sa baseline modelom.
  - U okviru testiranih konfiguracija, jednostavnija GRU arhitektura (`hidden_size = 32`, jedan GRU sloj) ostvarila je najbolji `validation_MAE`, što ukazuje da **složeniji modeli nisu doneli bolju generalizaciju** za ovaj slučaj.

