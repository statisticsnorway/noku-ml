import numpy as np
import pandas as pd


def rette_bedrifter(good_df):

    good_df["oms_share"] = good_df["omsetn_kr"] / good_df["tot_oms_fordelt"].round(5)

    # Round the values to whole numbers before assigning to the new columns
    good_df["new_oms"] = (
        (good_df["oms_share"] * good_df["foretak_omsetning"]).round(0).astype(int)
    )

    # Define the values for w_naring_vh, w_nace1_ikke_vh, and w_nace2_ikke_vh
    w_naring_vh = ("45", "46", "47")
    w_nace1_ikke_vh = "45.403"
    w_nace2_ikke_vh = ("45.2", "46.1")

    enhetene_brukes = good_df.copy()

    # Filter the DataFrame based on conditions and create vhbed variable
    enhetene_brukes["vhbed"] = 0

    # Check if the first two characters of 'naring' are in w_naring_vh
    enhetene_brukes.loc[
        enhetene_brukes["tmp_sn2007_5"].str[:2].isin(w_naring_vh), "vhbed"
    ] = 1

    # Check if 'naring' is in w_nace1_ikke_vh
    enhetene_brukes.loc[enhetene_brukes["tmp_sn2007_5"] == w_nace1_ikke_vh, "vhbed"] = 0

    # Check if the first four characters of 'naring' are in w_nace2_ikke_vh
    enhetene_brukes.loc[
        enhetene_brukes["tmp_sn2007_5"].str[:4].isin(w_nace2_ikke_vh), "vhbed"
    ] = 0

    salgsint_forbruk = enhetene_brukes[
        [
            "orgnr_n_1",
            "lopenr",
            "v_orgnr",
            "forbruk",
            "salgsint",
            "radnr",
            "nacef_5",
            "tmp_sn2007_5",
            "new_oms",
            "vhbed",
        ]
    ]

    har = salgsint_forbruk[
        salgsint_forbruk.groupby("orgnr_n_1")["vhbed"].transform("any")
    ]
    # Extract the 'orgnr_n_1' column
    har = har[["orgnr_n_1"]]

    # Remove duplicates
    har.drop_duplicates(inplace=True)

    ikke_har = salgsint_forbruk[
        ~salgsint_forbruk.groupby("orgnr_n_1")["vhbed"].transform("any")
    ]
    ikke_har = ikke_har[["orgnr_n_1"]]
    ikke_har.drop_duplicates(inplace=True)

    ikke_har["ikkevbed"] = 1

    # Merge ikke_har into salgsint_forbruk with a left join on the 'id' column
    salgsint_forbruk_update1 = pd.merge(
        salgsint_forbruk, ikke_har, on="orgnr_n_1", how="left"
    )

    # salgsint_forbruk_update1['ikkevbed'].fillna(0, inplace=True)

    # Update 'vhbed' to 1 where 'ikkevbed' is 1
    salgsint_forbruk_update1.loc[salgsint_forbruk_update1["ikkevbed"] == 1, "vhbed"] = 1

    # Create sum1 DataFrame for vhbed=1
    sum1 = (
        salgsint_forbruk_update1[salgsint_forbruk_update1["vhbed"] == 1]
        .groupby(["orgnr_n_1", "lopenr"])["new_oms"]
        .sum()
        .reset_index()
    )
    sum1.rename(columns={"new_oms": "sumoms_vh"}, inplace=True)

    # Create sum2 DataFrame for vhbed=0
    sum2 = (
        salgsint_forbruk_update1[salgsint_forbruk_update1["vhbed"] == 0]
        .groupby(["orgnr_n_1", "lopenr"])["new_oms"]
        .sum()
        .reset_index()
    )
    sum2.rename(columns={"new_oms": "sumoms_andre"}, inplace=True)

    sum3 = pd.merge(sum1, sum2, on=["orgnr_n_1", "lopenr"], how="outer")

    salgsint_forbruk_update2 = pd.merge(
        salgsint_forbruk_update1, sum3, on=["orgnr_n_1", "lopenr"], how="outer"
    )

    # Sort the DataFrame by 'orgnr_n_1', 'lopenr', and 'rad_nr'
    salgsint_forbruk_update2.sort_values(
        by=["orgnr_n_1", "lopenr", "radnr"], inplace=True
    )

    salgsint_forbruk_update2.sort_values(
        by=["orgnr_n_1", "lopenr", "vhbed"], inplace=True
    )

    # Sort the DataFrame by 'orgnr_foretak' and 'lopenr'

    salgsint_forbruk_update3 = salgsint_forbruk_update2.copy()

    salgsint_forbruk_update3.sort_values(by=["orgnr_n_1", "lopenr"], inplace=True)

    # Create a new variable 'vhf' based on the values of 'vhbed'
    salgsint_forbruk_update3["vhf"] = salgsint_forbruk_update3.groupby(
        ["orgnr_n_1", "lopenr"]
    )["vhbed"].transform("first")

    # Retain the value of 'vhf' from the first observation in each group
    salgsint_forbruk_update3["vhf"] = salgsint_forbruk_update3.groupby(
        ["orgnr_n_1", "lopenr"]
    )["vhf"].transform("first")

    # Apply labels to the variables
    salgsint_forbruk_update3["vhbed"] = salgsint_forbruk_update3["vhbed"].astype(str)
    salgsint_forbruk_update3["vhf"] = salgsint_forbruk_update3["vhf"].astype(str)

    label_map_vhbed = {"1": "varehandelsbedrift", "0": "annen type bedrift"}
    label_map_vhf = {
        "1": "foretaket har kun varehandelsbedrifter eller ingen",
        "0": "har varehandel og annen bedrift (blandingsnæringer)",
    }

    salgsint_forbruk_update3["vhbed"] = salgsint_forbruk_update3["vhbed"].map(
        label_map_vhbed
    )
    salgsint_forbruk_update3["vhf"] = salgsint_forbruk_update3["vhf"].map(label_map_vhf)

    # Filter rows where vhf is 'foretaket har kun varehandelsbedrifter eller ingen'
    vhf_condition = (
        salgsint_forbruk_update3["vhf"]
        == "foretaket har kun varehandelsbedrifter eller ingen"
    )
    vhf_df = salgsint_forbruk_update3.loc[vhf_condition]

    # Filter rows where vhf is not 'foretaket har kun varehandelsbedrifter eller ingen'
    andre_df = salgsint_forbruk_update3.loc[~vhf_condition]

    vhf_df["nokkel"] = vhf_df["new_oms"] / vhf_df["sumoms_vh"]

    # Convert 'salgsint' column to numeric
    vhf_df["salgsint"] = pd.to_numeric(vhf_df["salgsint"], errors="coerce")
    vhf_df["forbruk"] = pd.to_numeric(vhf_df["forbruk"], errors="coerce")

    vhf_df["bedr_salgsint"] = round(vhf_df["salgsint"] * vhf_df["nokkel"])
    vhf_df["bedr_forbruk"] = round(vhf_df["forbruk"] * vhf_df["nokkel"])

    andre_df["forbruk"] = pd.to_numeric(andre_df["forbruk"], errors="coerce")
    andre_df["salgsint"] = pd.to_numeric(andre_df["salgsint"], errors="coerce")

    # Assuming 'andre' is your DataFrame
    andre_df["avanse"] = andre_df["forbruk"] / andre_df["salgsint"]

    # Filter rows where vhbed is 1
    vh_bedriftene = andre_df[andre_df["vhbed"] == "varehandelsbedrift"].copy()

    # Calculate 'nokkel', 'bedr_salgsint', and 'bedr_forbruk' for vh-bedriftene
    vh_bedriftene["nokkel"] = vh_bedriftene["new_oms"] / vh_bedriftene["sumoms_vh"]
    vh_bedriftene["bedr_salgsint"] = round(
        vh_bedriftene["salgsint"] * vh_bedriftene["nokkel"]
    )
    vh_bedriftene.loc[
        vh_bedriftene["bedr_salgsint"] > vh_bedriftene["new_oms"], "bedr_salgsint"
    ] = vh_bedriftene["new_oms"]
    vh_bedriftene["bedr_forbruk"] = round(
        vh_bedriftene["bedr_salgsint"] * vh_bedriftene["avanse"]
    )

    # Summarize vh-bedriftene
    brukt1 = (
        vh_bedriftene.groupby(["orgnr_n_1", "lopenr"])
        .agg({"bedr_salgsint": "sum", "bedr_forbruk": "sum"})
        .reset_index()
    )

    # Merge summarized values back to 'andre'
    andre = pd.merge(andre_df, brukt1, on=["orgnr_n_1", "lopenr"], how="left")

    # Calculate 'resten1' and 'resten2'
    andre["resten1"] = andre["salgsint"] - andre["bedr_salgsint"]
    andre["resten2"] = andre["forbruk"] - andre["bedr_forbruk"]

    # Filter rows where vhbed is not 1
    blanding_av_vh_og_andre = andre[andre["vhbed"] != "varehandelsbedrift"].copy()

    # Calculate 'nokkel', 'bedr_salgsint', and 'bedr_forbruk' for blending of vh and other industries
    blanding_av_vh_og_andre["nokkel"] = (
        blanding_av_vh_og_andre["new_oms"] / blanding_av_vh_og_andre["sumoms_andre"]
    )
    blanding_av_vh_og_andre["bedr_salgsint"] = round(
        blanding_av_vh_og_andre["resten1"] * blanding_av_vh_og_andre["nokkel"]
    )
    blanding_av_vh_og_andre["bedr_forbruk"] = round(
        blanding_av_vh_og_andre["resten2"] * blanding_av_vh_og_andre["nokkel"]
    )

    # Combine the two subsets back into 'andre'
    andre = pd.concat([vh_bedriftene, blanding_av_vh_og_andre], ignore_index=True)

    andre.sort_values(by=["orgnr_n_1", "lopenr"], inplace=True)

    oppdatere_hv = pd.concat([vhf_df, andre], ignore_index=True)

    oppdatere_hv = oppdatere_hv[
        ["orgnr_n_1", "lopenr", "radnr", "bedr_forbruk", "bedr_salgsint"]
    ]

    enhetene_brukes2 = pd.merge(
        enhetene_brukes, oppdatere_hv, on=["orgnr_n_1", "lopenr", "radnr"]
    )

    rettes = enhetene_brukes2.copy()

    rettes["oms"] = rettes["new_oms"]
    rettes["driftsk"] = rettes["gjeldende_driftsk_kr"]

    # Convert columns to numeric
    rettes["tot_driftskost_fordelt"] = pd.to_numeric(
        rettes["tot_driftskost_fordelt"], errors="coerce"
    )
    rettes["driftsk"] = pd.to_numeric(rettes["driftsk"], errors="coerce")

    rettes["drkost_share"] = rettes["driftsk"] / rettes["tot_driftskost_fordelt"]

    rettes["new_drkost"] = rettes["drkost_share"] * rettes["foretak_driftskostnad"]

    rettes2 = rettes.copy()
    rettes2["drkost_temp"] = rettes2["new_drkost"]

    # Fill NaN in 'drkost_temp' with 0
    rettes2["drkost_temp"] = rettes2["drkost_temp"].fillna(0)

    rettes2["gjeldende_lonn_kr"] = pd.to_numeric(
        rettes2["gjeldende_lonn_kr"], errors="coerce"
    ).fillna(0)
    rettes2["bedr_forbruk"] = pd.to_numeric(
        rettes2["bedr_forbruk"], errors="coerce"
    ).fillna(0)

    rettes2["lonn_+_forbruk"] = rettes2["gjeldende_lonn_kr"] + rettes2["bedr_forbruk"]

    # Perform the if operation
    condition = rettes2["drkost_temp"] < rettes2["lonn_+_forbruk"]
    rettes2["drkost_temp"] = np.where(
        condition, rettes2["lonn_+_forbruk"], rettes2["drkost_temp"]
    )
    rettes2["theif"] = np.where(condition, 1, 0)

    dkvars = rettes2[rettes2.groupby("orgnr_n_1")["theif"].transform("any")]

    # Calculate 'utskudd'
    dkvars["utskudd"] = (
        dkvars["new_drkost"] - dkvars["gjeldende_lonn_kr"] - dkvars["bedr_forbruk"]
    )
    dkvars["utskudd"] = abs(dkvars["utskudd"])

    # Keep selected columns
    columns_to_keep = [
        "orgnr_n_1",
        "lopenr",
        "radnr",
        "utskudd",
        "new_drkost",
        "drkost_temp",
        "theif",
        "gjeldende_lonn_kr",
        "bedr_forbruk",
    ]
    dkvars = dkvars[columns_to_keep]

    sum7b = (
        dkvars.groupby(["orgnr_n_1", "lopenr", "theif"])["utskudd"].sum().reset_index()
    )

    # Transpose the result
    sum7b_transposed = sum7b.pivot(
        index=["orgnr_n_1", "lopenr"], columns="theif", values="utskudd"
    ).reset_index()

    # Rename columns as per SAS code
    sum7b_transposed.rename(columns={0: "thief0", 1: "thief1"}, inplace=True)

    sum7b_transposed = sum7b_transposed[["orgnr_n_1", "lopenr", "thief0", "thief1"]]

    # merge sums
    dkvars_2 = pd.merge(
        dkvars, sum7b_transposed, on=["orgnr_n_1", "lopenr"], how="inner"
    )

    # Apply conditional logic
    pd.set_option("display.float_format", "{:.2f}".format)
    dkvars_2["andel1"] = np.where(
        dkvars_2["theif"] == 0, dkvars_2["utskudd"] / dkvars_2["thief0"], np.nan
    )
    dkvars_2["andel2"] = np.where(
        dkvars_2["theif"] == 0,
        np.round(dkvars_2["andel1"] * dkvars_2["thief1"]),
        np.nan,
    )
    # dkvars_2['new_drkost'] = np.where(dkvars_2['theif'] == 0, np.sum(dkvars_2['drkost_temp'] - dkvars_2['andel2'], axis=0), dkvars_2['drkost_temp'])
    dkvars_2["new_drkost"] = np.where(
        dkvars_2["theif"] == 0,
        dkvars_2["drkost_temp"] - dkvars_2["andel2"],
        dkvars_2["drkost_temp"],
    )

    # Keep selected columns
    columns_to_keep = ["orgnr_n_1", "lopenr", "radnr", "new_drkost"]
    dkvars_3 = dkvars_2[columns_to_keep]

    # dkvars_2.head(50)

    good_final = dkvars_2.copy()

    return good_final
