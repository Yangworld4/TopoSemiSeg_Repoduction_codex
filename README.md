# TopoSemiSeg reproduction

This repository now contains an end-to-end reproduction of:

> Xu et al., **Semi-supervised Segmentation of Histopathology Images with
> Noise-Aware Topological Consistency**, ECCV 2024.

The implementation includes the paper's UNet++ student, EMA teacher, supervised
CE + Dice loss, Gaussian-ramped pixel consistency, persistence-diagram
signal/noise decomposition, Wasserstein signal matching, noisy-topology
removal, two-stage training, validation, and checkpointing.

## 1. Environment

The paper used Python 3.9.12, PyTorch 2.0.1, and CUDA 11.7. A matching setup is:

```bash
conda create -n toposemiseg python=3.9.12 -y
conda activate toposemiseg
pip install -r requirements.txt
```

`cripser` performs cubical persistent homology. GUDHI and POT perform optimal
Wasserstein matching. Persistent homology is computed on CPU; the selected
critical-pixel loss remains differentiable on GPU.

## 2. Data manifests

Images are not redistributed. Download CRAG, GlaS, or MoNuSeg from their
official sources and create CSV manifests. Paths may be absolute or relative to
the CSV file.

Labeled/validation manifest:

```csv
image,mask
images/train_001.png,masks/train_001.png
images/train_002.png,masks/train_002.png
```

Unlabeled manifest:

```csv
image
images/train_003.png
images/train_004.png
```

Masks may use `0/1`, `0/255`, or instance IDs; every nonzero value is treated as
foreground.

The supplementary material uses these splits:

| Dataset | Train / validation / test | 10% labeled | 20% labeled |
|---|---:|---:|---:|
| CRAG | 153 / 20 / 40 | 16 | 31 |
| GlaS | 68 / 17 / 80 | 7 | 14 |
| MoNuSeg | 24 / 6 / 14 | 3 | 5 |

For a semi-supervised run, put the labeled subset in `labeled_manifest` and the
remaining training images in `unlabeled_manifest`. Store the exact randomly
selected names so that a result is reproducible.

For the prepared CRAG/GLaS directory used in this workspace, deterministic
nested 10%/20% manifests can be regenerated with:

```bash
python tools/prepare_paper_splits.py --data-root data --seed 2026
```

The resulting files are under
`data/<dataset>/manifests/paper_seed_2026/`. GLaS is stratified by its
benign/malignant tag. Because the paper does not publish its random sample IDs,
the seed and all selected members are stored in `split_summary.json`.

## 3. Train

Edit one of the configs in `configs/`, especially the three manifest paths:

```bash
# CRAG 20% labeled
python train.py --config configs/crag.yaml

# CRAG 10% labeled
python train.py --config configs/crag_10.yaml

# GLaS 20% / 10% labeled
python train.py --config configs/glas.yaml
python train.py --config configs/glas_10.yaml
```

This executes both paper stages:

1. pretrain using supervised and pixel-consistency losses for 12,000 iterations
   on CRAG/GlaS (2,000 on MoNuSeg);
2. fine-tune for 500 epochs after adding the topology loss.

The pretrained checkpoint can also be reused explicitly:

```bash
python train.py --config configs/crag.yaml --stage pretrain
python train.py --config configs/crag.yaml \
  --stage finetune \
  --checkpoint runs/crag_20pct/pretrained.pt
```

Outputs are written under `output_dir`:

- `resolved_config.yaml`: exact run configuration;
- `metrics.jsonl`: machine-readable training/validation log;
- `pretrained.pt`: stage-one checkpoint;
- `last.pt`: latest fine-tuning checkpoint.

Resume an interrupted run by passing `--checkpoint path/to/last.pt`.

## 4. Evaluate

```bash
python evaluate.py \
  --config configs/crag.yaml \
  --checkpoint runs/crag_20pct/last.pt \
  --manifest data/CRAG/manifests/paper_seed_2026/test.csv
```

By default evaluation uses the student, consistent with the paper's training
objective. Pass `--weights teacher` to inspect the EMA teacher.

The included evaluator reports semantic Dice, area-weighted object Dice, IoU,
accuracy, foreground component-count (0-D Betti) error over 256-pixel windows,
and connected-component VOI. Betti Matching Error requires the induced-matching
evaluation package used by its original paper and is not mislabeled by this
generic evaluator.

## 5. Topological loss API

```python
import torch
from topo_consistency_loss import TopologyConsistencyLoss

criterion = TopologyConsistencyLoss(
    patch_size=100,
    persistence_threshold=0.7,
)

student_probability = torch.softmax(student_logits, dim=1)[:, 1]
teacher_probability = torch.softmax(teacher_logits, dim=1)[:, 1]
topology_loss = criterion(student_probability, teacher_probability)
loss = supervised_loss + pixel_weight * pixel_loss + 0.002 * topology_loss
```

`return_parts=True` returns `consistency` (Eq. 5) and `removal` (Eq. 7)
separately. The old `getTopoLoss(...)` and `calculate_topo_loss(...)` entry
points remain available.

## 6. Reproduction notes

The official supplement specifies random crop, rotation/flip, color change, and
"morphological shift", but does not define the latter algorithmically. This
repository uses a configurable one-pixel random min/max filter, which perturbs
stain morphology without breaking spatial correspondence between student and
teacher outputs. It also interprets the reported total batch size as an equal
labeled/unlabeled split. Both assumptions are explicit in the YAML files.

Key paper defaults are already encoded:

- UNet++ backbone;
- Adam optimizer;
- EMA decay `0.999`;
- topology weight `0.002`;
- persistence threshold `0.7`;
- pixel consistency maximum weight `0.1`;
- supervised CE/Dice weights `0.5/0.5`;
- CRAG/GlaS: crop `256`, batch `16`, learning rate `5e-4`;
- MoNuSeg: crop `416`, batch `8`, learning rate `1e-4`.

## Citation

```bibtex
@inproceedings{xu2024toposemiseg,
  title={Semi-supervised Segmentation of Histopathology Images with
         Noise-Aware Topological Consistency},
  author={Xu, Meilong and Hu, Xiaoling and Gupta, Saumya and
          Abousamra, Shahira and Chen, Chao},
  booktitle={ECCV},
  year={2024}
}
```
