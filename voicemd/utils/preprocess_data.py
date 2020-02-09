import pandas as pd
import re
import os


def clean_xlsx(fname):

    # open file
    df = pd.read_excel(fname)

    # clean_xlsx:

    # clean gender column
    df["Gender"] = df["Gender"].str.upper()
    df["Gender"] = df["Gender"].replace("FEMALE", "F")
    df["Gender"] = df["Gender"].replace("MALE", "M")

    # rename columns
    df = df.rename(columns={"Participant ID ": "uid"})
    df.columns = df.columns.str.lower()

    df["uid"] = df["uid"].str.upper()  # make uids lower case

    return df


def clean_filenames(data_path):
    original_filenames = [file.upper() for file in os.listdir(dpath) if "wav" in file]
    original_filenames.sort()

    reg = r"[\ \_ \.]*E_*NSS"
    cleaned_filenames = [
        re.sub(reg, "", file, re.I)
        for file in original_filenames
        if re.search(reg, file, re.I)
    ]
    cleaned_filenames = [
        re.sub(r"WAV", r"wav", file, re.I) for file in cleaned_filenames
    ]
    assert len(cleaned_filenames) == len(original_filenames), (
        "Something" "s prob wrong in your regex"
    )

    old_and_new_filenames = dict(zip(original_filenames, cleaned_filenames))
    return old_and_new_filenames


if __name__ == "__main__":

    dpath = "/home/jerpint/voicemd/data/"
    fname = "~/voicemd/data/first_sharing_demographics.xlsx"
    df = clean_xlsx(fname)

    old_and_new_filenames = clean_filenames(dpath)
