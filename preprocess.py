import pandas as pd
import numpy as np
import os


def preprocess():
    responses = pd.read_csv("./raw/responses.csv")
    responses["time"] = pd.to_datetime(responses["time"])
    responses = responses.sort_values("time")
    responses = responses.rename(
        columns={"fact_check_method": "fact_check_source"})
    responses["fact_check_source"] = responses["fact_check_source"].fillna(
        "none")
    responses = responses.drop(columns=["id", "time", "session_id", "study_id"])
    responses = responses.drop_duplicates(subset=["prolific_pid", "post_id"])

    participants = pd.concat([
        pd.read_csv("./raw/participants1.csv"),
        pd.read_csv("./raw/participants2.csv"),
    ],
                             ignore_index=True)
    participants = participants[participants["Status"] == "APPROVED"]
    participants = participants.drop(columns=[
        "Submission id", "Status", "Custom study tncs accepted at",
        "Started at", "Completed at", "Reviewed at", "Archived at",
        "Time taken", "Completion code", "Total approvals", "Fluent languages",
        "Country of birth", "Country of residence", "Nationality", "Language"
    ])
    participants.columns = [
        "prolific_pid", "political_affiliation", "age", "gender", "ethnicity",
        "student_status", "employment_status"
    ]
    participants = participants.drop_duplicates(subset=["prolific_pid"])

    participants["employment_status"] = participants[
        "employment_status"].replace("Full-Time", True)
    participants["employment_status"] = participants[
        "employment_status"].replace("Part-Time", True)
    participants["employment_status"] = participants[
        "employment_status"].replace(
            "Due to start a new job within the next month", False)
    participants["employment_status"] = participants[
        "employment_status"].replace("Unemployed (and job seeking)", False)
    participants["employment_status"] = participants[
        "employment_status"].replace(
            "Not in paid work (e.g. homemaker', 'retired or disabled)", False)

    participants["student_status"] = participants["student_status"].replace(
        "Yes", True)
    participants["student_status"] = participants["student_status"].replace(
        "No", False)

    participants = participants.replace("DATA_EXPIRED", np.nan)
    participants = participants.replace("Other", np.nan)

    if not os.path.exists("./data/"):
        os.mkdir("./data/")

    responses.to_csv("./data/responses.csv", index=False)
    participants.to_csv("./data/participants.csv", index=False)


if __name__ == "__main__":
    preprocess()
