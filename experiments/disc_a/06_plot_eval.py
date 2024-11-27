# %%
from pathlib import Path

base_dir = Path("/nrs/funke/senetaire/duplex_paper_v2_eval/")
experiment = "disc_a"
expt_dir = base_dir / experiment
# %%
from quac.report import Report

reports = {}
for method in ["deeplift", "ig"]:
    report = Report(name=method)
    report.load(
        f"/nrs/funke/adjavond/projects/duplex/disc_a/reports/{method}_report.json/default.json"
    )
    reports[method] = report

for exp_dir in expt_dir.iterdir():
    # print(exp_dir)
    if "stylegan_folder_counterfactual" in exp_dir.name:
        if (exp_dir / "report.yaml").exists():
            reports[exp_dir.name] = Report(name=exp_dir.name)
            try:
                reports[exp_dir.name].load((exp_dir / "report.yaml") / "no_blur.json")
                reports[exp_dir.name + "_blur"].load(
                    (exp_dir / "report.yaml") / "default.json"
                )
            except:
                print(list((exp_dir / "report.yaml").iterdir()))
# %%

# %% Plot the curves
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for method, report in reports.items():
    report.plot_curve(ax=ax)
# Add the legend above the plot
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=1)

# %%
len(reports)

# %%
import pandas as pd


def create_metadata(name, report):
    metadata = []
    # The dataset name should be the first part of the name
    # Remove "disc_a_" from the name
    name = name.replace("disc_a_", "")
    if "stylegan_folder_counterfactual" in name:
        metadata.append("stylegan_folder_counterfactual")
        name = name.replace("stylegan_folder_counterfactual_", "")
    elif "stylegan_folder" in name:
        metadata.append("stylegan_folder")
        name = name.replace("stylegan_folder_", "")
    if "cste" in name:
        # Get cste + what is directly after
        cste_val = name.split("_")[0] + "_" + name.split("_")[1]
        metadata.append(cste_val)
        name = name.replace(cste_val + "_", "")
    if "uniform_noise" in name:
        metadata.append("uniform_noise")
        name = name.replace("uniform_noise_", "")
    if "reg" in name:
        # get reg + what is directly after
        reg_val = name.split("_")[0] + "_" + name.split("_")[1]
        metadata.append(reg_val)
        name = name.replace(reg_val + "_", "")
    # If anything left, it should be a date
    if name != "":
        date, time = name.split("-")
        metadata.append(date)
        metadata.append(time)
    # Add the report
    metadata.append(report)
    return metadata


metadata = [create_metadata(name) for name in reports.keys()]
df = pd.DataFrame(metadata)
df

# %%
