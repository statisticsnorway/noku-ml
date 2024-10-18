# NNI Tester

English version:

- [English](NNI_README.md)

Dette programmet har som mål å evaluere ytelsen til NNI-modellen vi bruker i produksjonen av NØKU-statistikken. Dette gjøres ved hjelp av en bootstrap-metode som behandler flere års reg_type 01 foretaksskjema som populasjonen. Et tilfeldig utvalg av dette datasettet blir valgt tilfeldig over X antall iterasjoner for å fungere som et utvalg og 'giver' informasjon til de resterende radene i populasjonen. Resultatene evalueres ved å sammenligne predikert salg med faktisk rapportert salgsinntekt fra undersøkelsesdeltakerne.

#### Vennligst merk: 

Selv om koden er i stand til å analysere andre grupper og flere grupper samtidig, er alle testene som vises her kun for varehandel og fokuserer kun på imputering av salgsinntekter. Dette ble hovedsakelig gjort for enkelhetens skyld og for å demonstrere et konseptbevis.

Her er en gjennomgang av analysen:

## Histogramanalyse:

Ikke strengt nødvendig for NNI-testing, men gir viktig kontekst for skjevhet og fordeling av data på et '3-siffer' næringnivå. Dette har viktige implikasjoner for hvor mange foretak vi bør manuelt redigere/sjekke, men også for hvor mange skjema vi bør levere uansett.

![image](https://github.com/user-attachments/assets/ce08705b-96da-4cc1-b758-671e1fe71e8b)

## Bootstrap-testing

Bootstrap-testing ble brukt for å evaluere modellens ytelse, med mål om å få innsikt i hvordan den kan prestere år etter år, i stedet for bare en enkelt iterasjon.

![image](https://github.com/user-attachments/assets/839eaefa-2a58-4c61-831a-c1c9d0931c5c)

#### Resultater:

MAE i resultatene er faktisk ganske rimelig, selv om den i noen iterasjoner var betydelig høyere. En nærmere titt på residualfeilene vekker imidlertid noen bekymringer. Det ser ut til å være noen store uteliggere, og i tillegg ser det ut til å være noen skjevheter i residualene.

## Evaluering av en ny modell

En ny modell ble evaluert over to år (bare to år med publiserte data er tilgjengelige i skyplattformen for øyeblikket). Denne modellen endret ikke mye av det vi bruker i dag, ettersom hovedmålet med analysen var å vurdere muligheten for å bruke reg_type 2-selskaper slik at det kan bli mulig å ikke lenger levere skjema til reg_type 1-selskaper. Vi brukte imidlertid en knn = X-metode, i stedet for en enkelt nabo. De samme funksjonene ble brukt til å trene modellen, så det er potensiale for å legge til flere funksjoner i fremtiden.

#### Resultater:

![image](https://github.com/user-attachments/assets/63041104-08f2-40bd-b338-158235e958e1)

MAE var faktisk større i dette tilfellet, men residualplottet ser ut til å være betydelig forbedret.

En A/B-test ble også produsert (som ikke kan deles her på grunn av konfidensialitetsregler) som sammenlignet våre faktisk publiserte resultater med hva de ville vært ved bruk av den nye metoden. Den faktiske endringen var ubetydelig for de fleste næringer - men noen næringer hadde store forskjeller. Så noen ekstra kontroller må legges til.

## Områder for forbedring

- Legg til kontroller for uteliggere i resultatene.
- Bygg en mer robust modell for å erstatte NNI-modellen.

