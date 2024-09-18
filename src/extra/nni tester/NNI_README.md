# noku_ml

engelsk versjon:

- [English](./NNI_README_NO.md)

Hovedproblemet som løses, er unøyaktige/lavkvalitets svar levert av respondenter til en undersøkelse - ofte med svært materielle konsekvenser for det endelige resultatet. Dette tar normalt et team av statistikere et helt år å korrigere (noen ganger resulterer det i at man må kontakte respondentene på nytt) - Jeg skal løse denne oppgaven ved hjelp av maskinlæring og andre statistiske tiltak.

Resultater: En full produksjonskjøring (normalt fullført av et team på 5-7 personer over et helt år) fullført på 600,38 sekunder. Resultatene passerer flere logiske tester og sammenlignet med tidligere produksjoner viser de seg å være svært gunstige. R^2 når man sammenligner det dette programmet produserer mot det som faktisk ble publisert var omtrent 98% med en gjennomsnittlig absolutt feil på omtrent 5.000 NOK - noe som er lavt gitt egenskapene til våre data.

Føl deg fri til å klone repoet hvis du har passende tilgang. Jeg vil også demonstrere hva koden gjør her i denne ReadMe-filen:

## Visualiseringer:

Flere visualiseringer brukes til å analysere dataene på industrinivå. Plottene er interaktive, brukeren kan velge år, diagramtyper, fokusvariabler osv. Alle de vanlige interaktive verktøyene fra Plotly er også tilgjengelige. Noen visualiseringer er animerte, og hvis brukeren trykker på spill av, vil de se endringer over tid. Her er hvordan noen av resultatene ser ut (som naturligvis vil justere seg hvis noe annet ble valgt i rullegardinmenyene).

**Enkle plot:** 
