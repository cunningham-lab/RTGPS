import math
import numpy as np
import pandas as pd

dataset_sizes = {
    "PolTele": "$9.6$K",
    "Elevators": "$10.6$K",
    "Bike": "$11.1$K",
    "Kin40K": "$25.6$K",
    "Protein": "$25.6$K",
    "KEGG": "$31.2$K",
    "KEGGU": "$40.7$K",
    "3DRoad": "$278$K",
}
dataset_order = ["PolTele", "Elevators", "Bike", "Kin40K", "Protein", "KEGG", "KEGGU", "3DRoad"]
method_order = ["Cholesky", "RFF", "SVGP", "sgGP", "CG", "RR-CG"]
metric_order = ["RMSE", "NLL"]


if __name__ == "__main__":
    results = pd.read_csv("results.csv").set_index(["Dataset", "Method", "Seed"])
    results = results.rename({
        "pol": "PolTele",
        "elevators": "Elevators",
        "bike": "Bike",
        "kin40k": "Kin40K",
        "protein": "Protein",
        "keggdirected": "KEGG",
        "keggundirected": "KEGGU",
        "3droad": "3DRoad",
    })
    results.columns.name = "Metric"

    # Make table with mean and std err info
    mean = results.groupby(["Dataset", "Method"]).mean().stack(level="Metric").to_frame("Mean")
    stderr = results.groupby(["Dataset", "Method"]).sem().stack(level="Metric").to_frame("Stderr")
    table = pd.concat([mean, stderr], axis=1)

    # Determine best methods
    is_best = pd.Series(False, index=table.index)
    for idx, rows in table.groupby(level=(["Dataset", "Metric"]), sort=False):
        sub_means = rows["Mean"].fillna(math.inf)
        sub_stdvs = rows["Stderr"].fillna(0.)
        best_idx = sub_means.idxmin()
        min_val = sub_means[best_idx] + sub_stdvs[best_idx]
        is_best_values = sub_means - sub_stdvs <= min_val
        is_best.loc[is_best_values.index] = is_best_values

    # Make a column columns mean + std err
    table["Value"] = table.apply(
        lambda x: f'{x["Mean"]:.3f} \\pm {x["Stderr"]:.3f}' if x["Stderr"] > 0 else f'{x["Mean"]:.3f}',
        axis=1
    )
    table["Value"] = table["Value"].str.replace(r"0\.", r".")

    # Bold best values
    table["Value"] = np.where(
        is_best,
        table["Value"].map(lambda x: f"$\\mathbf{{ {x} }}$"),
        table["Value"].map(lambda x: f"${x}$")
    )

    # Table should now just be mean + stderr
    table = table["Value"].reindex(metric_order, level="Metric").unstack("Metric")

    # Turn methods into columns
    table = table.reindex(method_order, level="Method").unstack("Method")
    table = table.fillna("---")

    # Add column of dataset size:
    table.reset_index(inplace=True)
    table["$n$"] = table["Dataset"].map(lambda x: dataset_sizes[x])
    table = table.set_index(["Dataset", "$n$"]).reindex(dataset_order, level="Dataset")

    ###
    # Hacky post-processing of results
    ###

    # nothing here right now ;)

    ###
    # Make into LaTeX table
    ###

    # Add blank column in between NLL and RMSE separation
    # These will be removed when converted to LaTeX
    num_methods = len(table.columns.unique(level="Method"))
    table.insert(num_methods, ("BLANKCOL", "BLANKCOL"), value=" ", allow_duplicates=True)
    table.insert(0, ("BLANKCOL", "BLANKCOL"), value=" ", allow_duplicates=True)
    print(table)

    # Convert to LaTeX
    latex = table.to_latex(
        escape=False,
        bold_rows=False, sparsify=True,
        multicolumn_format="c",
        column_format=("cc" + ("c" * len(table.columns))),
    )
    latex = latex.replace("BLANKCOL", "")

    # Hack: get rid of "Metric" and "Method"
    latex = latex.replace("Metric", "").replace("Method", "")

    # Hack: add hline between small and large datasets
    latex = latex.replace("Kin40K", "\\hline\nKin40K")

    with open("rbf_ard_table.tex", "w") as f:
        f.write(latex)
