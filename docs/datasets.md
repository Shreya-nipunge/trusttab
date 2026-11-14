# Datasets

## Research benchmark pair

TrustTab's intended core dataset pair is German Credit and the processed
Cleveland subset of the UCI Heart Disease dataset. Adult Census Income is
retained as an additional selectable dataset; it does not replace Cleveland
in the core research design.

## UCI Heart Disease: processed Cleveland subset

The loader uses UCI Machine Learning Repository dataset ID 45 and reads only
the archive member `processed.cleveland.data`. It does not combine the four
databases in the repository. The source describes 303 instances and 13
features. Features are returned in this order:

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`,
`oldpeak`, `slope`, `ca`, `thal`.

The raw target is `num`. Its original values are retained in
`y.attrs["original_num"]`; the binary target is `0` for original `num = 0` and
`1` for original `num` values 1, 2, 3, or 4. No rows are dropped. `?` markers
in source feature values become `NaN`; the source's missing feature values
are allowed in `ca` and `thal` and are preserved by the loader. No
imputation occurs before splitting. Existing TrustTab preprocessing fits its
imputer only on the training partition.

The expected original `num` counts are 0: 164, 1: 55, 2: 36, 3: 35, and
4: 13. After binary mapping this is 164 no-disease and 139 disease-present
records. The loader validates this distribution to detect an unintended
dataset variant or source change.

The categorical code columns `sex`, `cp`, `fbs`, `restecg`, `exang`,
`slope`, `ca`, and `thal` are returned as object columns containing integer
codes; missing values use ordinary `NaN`. Numeric measurements remain
numeric. This makes the existing preprocessing pipeline one-hot encode
categorical variables and median-impute numeric ones. Missing categorical
values are imputed from the training data's most common category.

Passing `data_home` to `load_heart_disease` caches a local CSV named
`heart_disease_uci45_cleveland.csv`. The cache contains the 13 source features
and the original multiclass `num`, before TrustTab's binary conversion. When
the cache exists, it is used without a network request. Without `data_home`,
the UCI archive is fetched on each call.

### Citation and license

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart
Disease*. UCI Machine Learning Repository, dataset ID 45.
[https://doi.org/10.24432/C52P4X](https://doi.org/10.24432/C52P4X).

The dataset is licensed under the [Creative Commons Attribution 4.0
International license](https://creativecommons.org/licenses/by/4.0/). Suggested
attribution: “Heart Disease dataset by Janosi, Steinbrunn, Pfisterer, and
Detrano, hosted by the UCI Machine Learning Repository, used under CC BY 4.0.”

Source page: <https://archive.ics.uci.edu/dataset/45/heart>.

The unified benchmark forwards the optional `dataset.data_home` setting to
loaders. It may be a single path or a dataset-to-path mapping. The core
configuration maps only Cleveland to `data/raw`; German Credit retains
scikit-learn/OpenML's existing cache location, avoiding an accidental cache
relocation.
