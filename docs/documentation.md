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
- **Preprocessing koraci:**
  - spajanje datuma i vremena u jednu `datetime` kolonu radi formiranja jedinstvene vremenske oznake za lakšu obradu vremenskih serija,
  - brisanje dana sa više od 60 nedostajućih ili praznih redova (bez obavljenih merenja),
    - kod dana gde se takvi slučajevi javljaju u manjem broju od 60, obrisani su samo ti redovi,
  - resamplovanje minutnih podataka u satne proseke, 
  - odabir relevantnih kolona (`Global_active_power`, `Global_reactive_power`, `Global_intensity`),
  - detalji o celom procesu preprocessinga dostupni su u log fajlu: `data/logs/full_report.csv`.
