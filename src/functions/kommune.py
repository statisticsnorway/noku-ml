def kommune():
    import pandas as pd
    import sgis as sg

    testdatasti = "ssb-prod-dapla-felles-data-delt/GIS/testdata"
    kommuner = sg.read_geopandas(f"{testdatasti}/enkle_kommuner.parquet")
    pop = pd.read_excel("Befolkning.xlsx")
    inntekt = pd.read_excel("inntekt.xlsx")
    pop["KOMMUNENR"] = pop["KOMMUNENR"].astype("object")
    inntekt["KOMMUNENR"] = inntekt["KOMMUNENR"].astype("object")

    kommuner["KOMMUNENR"] = kommuner["KOMMUNENR"].str.replace('"', "").astype(str)
    pop["KOMMUNENR"] = pop["KOMMUNENR"].astype(str)
    pop["KOMMUNENR"] = pop["KOMMUNENR"].astype(str).str.zfill(4)

    inntekt["KOMMUNENR"] = inntekt["KOMMUNENR"].astype(str)
    inntekt["KOMMUNENR"] = inntekt["KOMMUNENR"].astype(str).str.zfill(4)

    kommuner = kommuner.merge(pop, on="KOMMUNENR", how="left")
    kommuner = kommuner.merge(inntekt, on="KOMMUNENR", how="left")

    return kommuner
