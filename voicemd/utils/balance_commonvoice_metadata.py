import pandas as pd


def balance_and_filter_commonvoice_tsv(tsv):
    """

    We remove entries with no associated gender.
    There is a big bias towards males in twenties.
    We sample and rebalance the dataset such that there are
    equal samples across males and females for all age categoires
    in valid_age_categories.
    """

    metadata = tsv.copy()
    seed = 42
    # final samples will be stored here
    male_metadata = pd.DataFrame(columns=metadata.columns)
    female_metadata = pd.DataFrame(columns=metadata.columns)

    # remove samples where gender is unidentified
    metadata = metadata[metadata["gender"].isin(["male", "female"])]

    # show stats before rebalanceing
    print("Breakdown before rebalance: \n")
    print_metadata_stats(metadata)

    # keep only valid age categories
    valid_age_categories = ["twenties", "thirties", "fourties", "fifties", "sixties"]
    metadata = metadata[metadata["age"].isin(valid_age_categories)]

    # Take the minimum number in females as the number to take
    n_samples = min(metadata.loc[metadata["gender"] == "female"]["age"].value_counts())

    # for each gender, for each age, sample n_samples at random
    for age in valid_age_categories:
        # separate by age
        tmp_metadata = metadata[metadata["age"] == age]

        # separate by gender
        tmp_male_metadata = tmp_metadata[tmp_metadata["gender"] == "male"]
        tmp_female_metadata = tmp_metadata[tmp_metadata["gender"] == "female"]

        # sample and add to all results
        male_metadata = male_metadata.append(
            tmp_male_metadata.sample(n=n_samples, random_state=seed)
        )
        female_metadata = female_metadata.append(
            tmp_female_metadata.sample(n=n_samples, random_state=seed)
        )

    metadata = male_metadata.append(female_metadata)

    print("Breakdown after rebalance: \n")
    print_metadata_stats(metadata)

    return metadata


def print_metadata_stats(metadata):
    print("Breakdown after processing \n")
    print("Gender breakdown: \n", metadata["gender"].value_counts())
    print(
        "Age breakdown by gender (male): \n",
        metadata[metadata["gender"] == "male"]["age"].value_counts(),
    )
    print(
        "Age breakdown by gender (female): \n",
        metadata[metadata["gender"] == "female"]["age"].value_counts(),
    )


if __name__ == "__main__":
    common_voice_path = "/data/mozilla/"
    tsv_fname = common_voice_path + "train.tsv"

    tsv = pd.read_csv(tsv_fname, sep="\t")
    print("reading ", tsv_fname)

    metadata = balance_and_filter_commonvoice_tsv(tsv)
    metadata.rename(columns={"path": "filename"}, inplace=True)
