import re
import os
import zipfile

import pandas as pd


def clean_xlsx(fname, voice_clips_dir):

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

    # Align the uid with associated filenames
    df = align_uid_and_filename(df, voice_clips_dir)

    return df


def clean_filenames(dpath):
    original_filenames = [file for file in os.listdir(dpath) if "wav" in file]
    original_filenames.sort()
    tmp_fnames = [file.upper() for file in original_filenames if "wav" in file]

    reg = r"[\ \_ \.]*E_*NSS"
    cleaned_filenames = [
        re.sub(reg, "", file, re.I) for file in tmp_fnames if re.search(reg, file, re.I)
    ]
    cleaned_filenames = [
        re.sub(r"WAV", r"wav", file, re.I) for file in cleaned_filenames
    ]
    assert len(cleaned_filenames) == len(original_filenames), (
        "Something" "s prob wrong in your regex"
    )

    old_and_new_filenames = dict(zip(original_filenames, cleaned_filenames))
    return old_and_new_filenames


def align_uid_and_filename(df, dpath):
    # Make sure you have already renamed the files using rename_files()
    df["filename"] = None  # Add filename column
    for idx, uid in df["uid"].iteritems():
        for f in [file for file in os.listdir(dpath) if "wav" in file]:
            if uid == f.split(".")[0]:
                df.at[idx, "filename"] = f
                break
    return df


def rename_files(dpath):

    old_and_new_filenames = clean_filenames(dpath)
    for oldf, newf in old_and_new_filenames.items():
        os.rename(dpath + oldf, dpath + newf)


if __name__ == "__main__":

    # TODO: Add config file
    root_dir = "/home/jerpint/voicemd/"
    dpath = root_dir + "data/"
    zipfile_path = dpath + "voice files to share-20200201T172710Z-001.zip"
    fname = dpath + "voice_clips/first_sharing_demographics.xlsx"
    voice_clips_dir = dpath + "voice_clips/"

    if not os.path.exists(voice_clips_dir):
        print("Voice clips not found, extracting data to", voice_clips_dir)

        try:
            with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
                zip_ref.extractall(dpath)
            os.rename(dpath + "voice files to share", voice_clips_dir)
            rename_files(voice_clips_dir)
        except FileNotFoundError:
            "Download the .zip containing the voice clips"

        df = clean_xlsx(fname, voice_clips_dir)
        df.to_csv(voice_clips_dir + 'cleaned_metadata.csv')
