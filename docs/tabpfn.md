# TabPFN optional model

TrustTab's `tabpfn` optional dependency is pinned to `tabpfn==10.0.0` in
`pyproject.toml`. Install it only when selecting TabPFN:

```bash
python -m pip install "trusttab[tabpfn]"
```

The core TrustTab dependencies do not install TabPFN or PyTorch. TabPFN 10.0.0
supports Python 3.10 and later, which includes TrustTab's Python 3.11 and 3.12
range. It depends on PyTorch and the model checkpoint may need to be downloaded
on first use. The default TabPFN-3.5 weights are licensed under Prior Labs'
TABPFN-3.5 Non-Commercial License v1.0. That license permits non-commercial,
non-production use and restricts commercial/production use and hosting. Users
must review the full current [checkpoint license](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE)
and obtain any required separate commercial license. Installing TrustTab or
the TabPFN Python package does not grant rights to use the weights.

The current Hugging Face license text (last revised September 9, 2026) also
states that the license is not available to consumers within the meaning of
Section 13 of the German Civil Code and that accepting it represents
professional-purpose status under Section 14. Its restrictions apply to model
outputs as well: outputs may be used only for non-commercial purposes, and
commercial or production use of outputs requires a separate license. Users
must review the complete current terms and determine their own eligibility
before accessing the weights. TrustTab distributes software only; it does not
include TabPFN weights, grant model rights, or authorize use of model outputs.

## Input sizes and hardware

TabPFN 10.0.0 uses the TabPFN-3.5 checkpoint by default. Its published
guidance recommends GPU inference because CPU inference is slow. The package
applies a 5,000-training-row CPU guardrail for TabPFN-3/3.5; its broader
checkpoint limit is much larger, but that limit is not a runtime or memory
guarantee. TrustTab's planned training partitions (800 German Credit rows and
242 Cleveland rows in the uncalibrated split) are below the CPU guardrail.
See the [TabPFN 10.0.0 guidance](https://pypi.org/project/tabpfn/10.0.0/)
for version-specific details.

With TrustTab's default 80/20 split, the planned data sizes are 800 training
rows and 20 source features for German Credit, and 242 training rows and 13
source features for processed Cleveland. These row counts are below the
documented CPU limit. TrustTab's shared preprocessing pipeline may expand
categorical features before the TabPFN adapter receives them; this should be
considered when interpreting model comparisons.

TrustTab follows the project blueprint for TabPFN preprocessing: categorical
features are imputed on the training set and one-hot encoded, while numeric
features are median-imputed but not standardized. This is model-specific;
Logistic Regression continues to receive standardized numeric features. The
current TabPFN 10.0 guidance recommends passing native categorical inputs
without one-hot encoding. TrustTab retains the blueprint's common categorical
encoding convention for this study, so this choice should be recorded as a
version-specific preprocessing limitation and revisited before any redesign
of the research protocol.

The research configuration selects CPU explicitly so it does not depend on a
GPU being available. Phase 8 verified a bounded real fit and prediction with
TabPFN 10.0.0 and CPU PyTorch. This integration check does not establish model
performance or grant other users access to the model weights.

## Checkpoint authorization and cache

The installed TabPFN 10.0.0 package uses two separate authorization paths for
the gated default checkpoint `Prior-Labs/tabpfn_3_5`:

1. Accept the TabPFN model license through Prior Labs. For a headless run,
   provide the account's Prior Labs API key as `TABPFN_TOKEN` in the local
   process environment; do not store it in TrustTab configuration or source.
2. Accept access to the gated Hugging Face model repository and authenticate
   this environment with `hf auth login` (or its supported `HF_TOKEN`
   environment variable). This is separate from the Prior Labs API key.

This distinction is supported by the installed package source: its license
checker reads `TABPFN_TOKEN`, while checkpoint metadata/download uses
`huggingface_hub`; the checkpoint download exception recommends Hugging Face
authentication. The [Prior Labs weight access guide](https://docs.priorlabs.ai/models/accessing-model-weights)
documents the license and headless-token step. In this project environment,
the authorized download and bounded CPU fit were completed in Phase 8; each
user must independently meet the current license and gated-repository access
requirements. The expected default checkpoint is
`tabpfn-v3.5-20260909.safetensors`. Set
`TABPFN_MODEL_CACHE_DIR` to the isolated environment's cache before importing
TabPFN. Once both authorization steps succeed and the checkpoint is present,
the bounded verifier can be run with:

```powershell
$env:PYTHONPATH = "src"
$env:SKB_DATA_DIRECTORY = ".venv/phase8-tabpfn/skrub-data"
$env:TABPFN_MODEL_CACHE_DIR = ".venv/phase8-tabpfn/tabpfn-cache"
.venv/phase8-tabpfn/Scripts/python.exe scripts/verify_tabpfn.py --seed 123
```

Do not put credential values in command arguments, files, logs, or experiment
configuration. The 200-condition research matrix has not been run. A
successful bounded fit verifies integration only; it does not establish
comparative performance or license rights for other users.
