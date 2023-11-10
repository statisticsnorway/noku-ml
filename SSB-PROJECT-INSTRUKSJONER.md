# Om templaten

Valgene som er gjort i denne templaten dokumenteres nedover i denne readme-filen.
Etter at du har importert denne templaten inn i ditt prosjekt, står du fritt til å slette hele, eller deler av innholdet i denne filen, og erstatte med eget innhold.

## Plassering av notebooks og prosjektfunksjoner

Vi foreslår at funksjoner som skal brukes på tvers av notebooks legges i src/functions.\
Notebooks kan sorteres i mapper innenfor src/notebooks.\
I notebookene kan du da importere "de lokale funksjonene" med litt kode som ligger som eksempel øverst i src/notebooks/00_imports.ipynb\
Funksjoner som skal deles bredere enn innad i ett produksjonsløp, anbefaler vi at pakkes som en pakke og legges på PyPi. Pass på at ikke sikkerhetsdetaljer blir med i slike kodepubliseringer.

### Notat om importer og \_\_init\_\_.py-filene som ligger i noen mapper

Filene som heter "\_\_init\_\_.py" bør ligge der de er. Dette gjør at mappen de ligger i er "søkbar" når python leter etter funksjoner å hente inn.
Om du oppretter nye mapper, med kode du skal importere inn i andre filer, bør du nok også der opprette \_\_init\_\_.py-filer. Disse filene kan godt være tomme, om du ikke vil at noe kode skal kjøres under selve importen av funksjoner som ligger i denne mappen.

#### "Boilerplate"-kode for å finne prosjektmappen med hjelp av pyproject.toml

En forutsetning for mye av "hurtigstart"-koden er at denne filen (pyproject.toml) ligger kun en gang i prosjektet, og at det er der "grunnmappen" til prosjektet er.\
Denne lages av pakke-manageren "poetry", men er en av konfigurasjonsfilene python-miljøet beveger seg mot å legge mer og mer av "metadata om kodebasen" i.

### Instruksjoner for å innstallere nye pakker med Poetry

I terminalen:\
`poetry add pakke-navn`

### Kodetesting med Pytest

Vi anbefaler Pytest for kode-testing. Derfor ligger det en mappe for "tests" i grunnmappen for prosjektet.
For å kjøre pytest i terminalen, navigerer du først til prosjektmappen med
`cd prosjekt_mappe`
Så kjører du kommandoen
`pytest`
Pytest vil da lete etter test-mappen, finne .py-filer som starter med "test_". Inne i de, se etter funksjoner som har navn som starter med "\_test" og kjøre disse som en del av testsuiten.
Prøv det gjerne med koden du får inn som en del av templaten. Det er tre tester (tre prikker), en for hver av de tre funksjonene fizz, buzz og fizzbuzz.

### Innhold i .gitignore og .gitattributes

.gitignore bestemmer hvilke filer som skjules fra git-staging, altså filer git ignorerer, mtp. om de skal "bli med". Om dere bruker noen dataformater dere er nervøse for at kan ende sammen med koden, så kan det være nyttig å detaljere disse her.

.gitattributes forteller systemet bla. at den skal fjærne output fra under cellene i notebooks før de sendes opp til github. Dette er viktig da det ofte ligger data representert her. Pass gjerne på at disse filene er med ved å kjøre denne kommandoen i terminalen i bunnmappen av prosjektet deres:\
`ls -a`

### "MIT" som lisenstype

SSB har standardisert på [MIT-lisensen](https://github.com/statisticsnorway/adr/blob/main/docs/0006-aapen-kildekode-i-ssb.md)
som sin kildekodelisens.
