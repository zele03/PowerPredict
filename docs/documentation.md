# PowerPredict – Dokumentacija

**Autori:** Luka Zelembaba, Dušan Đurić  
**Datum:** jun 2026

## 1. Uvod
Projekat je urađen u okviru predmeta **Inteligentni sistemi** na Fakultetu Tehničkih Nauka u Novom Sadu.  
Cilj projekta je predikcija vremenskih serija – potrošnja električne energije – pomoću neuronskih mreža.  
Rad obuhvata pripremu i čišćenje podataka, njihovu transformaciju u oblik pogodan za analizu, kao i kasniji razvoj modela za predikciju, koji obuhvata izbor odgovarajuće arhitekture neuronske mreže, 
trening nad pripremljenim podacima i evaluaciju performansi modela u odnosu na definisane metrike.


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
  - dodavanje target kolone: `Energy_kWh` (identična numericka vrednost kao `Global_active_power`, ali eksplicitno definisana kao ciljna promenljiva),
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
  - **Glavne metrike:**
    - **WAPE (Weighted Absolute Percentage Error):** glavna procentualna metrika, meri ukupnu grešku u odnosu na ukupnu potrošnju.
    - **MAE (Mean Absolute Error):** prosečna greška izražena u kWh.
    - **RMSE (Root Mean Squared Error):** kaznjava velike greške, izraženo u kWh.

  - **Pomoćne metrike:**
    - **MAPE (Mean Absolute Percentage Error):** procentualna greška, ali nepouzdana zbog velikog broja malih vrednosti (oko 40% dataset‑a <0.5 kWh).
    - **sMAPE (Symmetric MAPE):** simetrična verzija MAPE, stabilnija kod malih vrednosti.
    - **Hourly WAPE, MAE, RMSE, MAPE, sMAPE:** računati radi detaljne analize performansi po satima.


- **Rezulati**

    | Skup        | WAPE (%) | MAE (kWh) | RMSE (kWh) | MAPE (%) | sMAPE (%) |
    |-------------|----------|-----------|------------|----------|-----------|
    | Validation  | 53.05    | 0.66      | 0.96       | 74.79    | 52.07     |
    | Test        | 53.47    | 0.51      | 0.76       | 66.80    | 49.29     |

  - **Grafički prikaz** rezultata dostupan u: `data\graphs\baseline_graphs`.

- **Analiza po satima (test skup)**

    | Kategorija       | Sati                 | WAPE (%) approx |
    |------------------|----------------------|-----------------|
    | Najgori sati     | 14h–18h, 23h         | 59–66           |
    | Najbolji sati    | 05h-10h              | 42–48           |

- **Zaključci:**

  - Baseline `lag_24h` pokazuje slabosti u popodnevnim satima (14h–18h), posebno oko 15h–17h, gde se potrošnja značajno razlikuje od prethodnog dana.
  - Jutarnji sati (7h–10h) su stabilniji i daju bolje rezultate.
  - Loš rezultat u 23h ukazuje na varijabilnost prelaza iz večernje aktivnosti u noć, što `lag_24h` ne uspeva da uhvati.


