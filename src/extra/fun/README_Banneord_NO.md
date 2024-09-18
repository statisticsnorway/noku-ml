# NØKU Banneordanalyse

English version:

- [English](README_Banneord.md)

![image](https://github.com/user-attachments/assets/585517da-5de8-44ac-995a-d4316b03c765)

#### Koden her bruker data som kun er tilgjengelig for ansatte i SSB. Konfidensiell data er skjult fra denne README-filen og fra koden. Selve koden ble utviklet i Statistisk sentralbyrå (SSB) sitt organisasjonsrepo, og senere ble en lettversjon kopiert hit for å reflektere mitt arbeid gjort for SSB.

Et verktøy for å analysere trender relatert til en kategori av svar (for eksempel banneord) i NØKU-statistikken.

En uhøytidelig analyse av spesifikke svar/kommentarer/tilbakemeldinger mottatt i OKI-skjemaet levert til respondentene av hovedundersøkelsen NØKU.

Selv om det er litt humoristisk ment, bør det forstås at denne tilbakemeldingen tas svært alvorlig, og selv om analysen her er gjort av kode, så leser mennesker faktisk all tilbakemeldingen som blir gitt. Dette er hvordan vi er i stand til å levere en omfattende ordliste som brukes i programmet.

Det finnes viktige meldinger i disse dataene. Negativ tilbakemelding er verdifull, uansett hvordan den uttrykkes. Vi i SSB har hørt meldingen høyt og tydelig, og vi gjør innsats for å forbedre opplevelsen for respondentene til undersøkelsene. Nåværende ideer som utforskes inkluderer bruk av AI for å hjelpe respondentene, bruk av maskinlæringsmodeller for å fylle ut fordelinger av forretningsstatistikk, og andre statistiske modeller er under utvikling med det endelige målet at vi ikke trenger å sende spørreskjema til like mange bedrifter.

## Instruksjoner

Å kjøre koden er ganske enkelt:

1. Skriv inn så mange banneord, uttrykk og fargerike kommentarer du kan komme på. Ha det gøy, vær fantasifull. Kjør deretter koden.

## Resultater:

Rangering av banneordbaserte svar basert på næringer på et 2-siffernivå:

![image](https://github.com/user-attachments/assets/9dfd7059-f0c2-45e3-a026-34d39ce3e13e)

Rangering av bruk av '!' på næringer på et 2-siffernivå (logaritmisk skala brukt):

![image](https://github.com/user-attachments/assets/c1845926-04b4-415e-a33f-d0bf7e11e3ce)

Rangering av banneord- og utropstegnbruk på et 3-siffernivå for næringer.

![image](https://github.com/user-attachments/assets/6b6871c7-2316-4674-b2a4-7a3d8ed1ea47)

![image](https://github.com/user-attachments/assets/ffb528ab-b879-4bac-86a3-17eb8d80f87c)

Er bruken av banneord og '!' korrelert? Det ser ikke ut til å være tilfelle. Det virker som om i Norge er det enten det ene eller det andre.

![image](https://github.com/user-attachments/assets/9b22ecfc-ee41-466e-8ff1-30377a683154)

Alle næringer over tid:

![image](https://github.com/user-attachments/assets/6a4a8d2c-14b5-46b5-b62e-9758f6352fff)

Interaktive plott for 2- og 3-siffernivå næringer over tid:

![image](https://github.com/user-attachments/assets/b622b33b-21bb-4ac0-b905-b40847bf259e)
