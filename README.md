# C-LEAD: Contrastive Learning for Enhanced Adversarial Defense

Ghosh, S., Kumar, S. & Sur, A. C-LEAD: Contrastive Learning for Enhanced Adversarial Defense. SN COMPUT. SCI. 7, 494 (2026).

DOI: https://doi.org/10.1007/s42979-026-05066-6

## Abstract

C-LEAD introduces a contrastive representation learning objective into adversarial training to improve model robustness and generalization under adversarial perturbations. The method augments standard adversarial loss with a contrastive loss that encourages robust, semantically meaningful embeddings; experiments on CIFAR-10 show improvements in robust accuracy against FGSM, PGD and CW attacks while maintaining competitive clean accuracy.

## Quick citation

If you use this code, please cite:

Ghosh, S., Kumar, S. & Sur, A. C-LEAD: Contrastive Learning for Enhanced Adversarial Defense. SN COMPUT. SCI. 7, 494 (2026). DOI: 10.1007/s42979-026-05066-6

BibTeX:

```
@article{Ghosh2026CLEAD,
  title={C-LEAD: Contrastive Learning for Enhanced Adversarial Defense},
  author={Ghosh, Suklav and Kumar, S. and Sur, A.},
  journal={SN Computer Science},
  volume={7},
  pages={494},
  year={2026},
  doi={10.1007/s42979-026-05066-6}
}
```

## Key features

- Integration of contrastive loss into adversarial training.
- Example training scripts for ResNet architectures on CIFAR-10.
- Attack generation and evaluation scripts for FGSM, PGD, and CW.
- Notebooks and visualizations to reproduce figures and diagnostics.

## Repository layout

- `README_C-LEAD.md` — this file.
- `requirements.txt` — core Python dependencies (see root).
- `pytorch-cifar-master-atk/` — main training/evaluation/attack scripts and model definitions.
- `mm/` — development notebooks, `resnet.py` variants, checkpoints, and visualization assets.
- `Unsupervised-Classification/`, `Results/` — additional experiments and outputs.

Paths you will use frequently:

- Dataset: `pytorch-cifar-master-atk/data/cifar-10-batches-py/`
- Checkpoints: `pytorch-cifar-master-atk/checkpoint/`, `mm/checkpoint/`
- Attack images: `pytorch-cifar-master-atk/atk_images/`

## Requirements

- Python 3.8+ (3.9/3.10 recommended)
- PyTorch (choose the `torch` / `torchvision` pair compatible with your CUDA)
- numpy, tqdm, scikit-learn, matplotlib, pillow, seaborn

Install the minimal environment (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need me to pin exact package versions used in the paper experiments, I can add them to `requirements.txt`.

## Quick start: reproduce core experiments

1. Prepare the CIFAR-10 dataset:

```bash
# from repository root
cd pytorch-cifar-master-atk
python -c "from torchvision import datasets; datasets.CIFAR10(root='data', download=True)"
```

2. Train a baseline model (ResNet-50):

```bash
cd pytorch-cifar-master-atk
python main_50.py --dataset cifar10 --epochs 200 --batch-size 128 --lr 0.1
```

3. Train with the C-LEAD objective (example flags; check script headers for exact names):

```bash
python main_50.py --dataset cifar10 --adv-train clead --epochs 200 --batch-size 128 \
    --lr 0.1 --contrastive-weight 1.0 --save-path checkpoint/clead_resnet50.pth
```

4. Generate adversarial images (FGSM / PGD):

```bash
python generate_attack_images.py --attack fgsm --eps 8 --dataset cifar10 --model-path checkpoint/clead_resnet50.pth
python generate_attack_images.py --attack pgd --eps 8 --steps 10 --dataset cifar10 --model-path checkpoint/clead_resnet50.pth
```

5. Evaluate robust accuracy:

```bash
python 34_atk_accu.py --model checkpoint/clead_resnet50.pth --attack pgd --eps 8 --steps 20
```

Notes:
- The repository contains multiple scripts with slightly different CLIs; inspect the top of each script (or run `--help`) to confirm exact flags.
- Attack `eps` may be expressed as integer (e.g., `8`) in existing scripts; confirm units (often `8/255`) before reproducing exact numbers.

## Recommended hyperparameters (examples)

- Dataset: CIFAR-10
- Model: ResNet-50
- Optimizer: SGD, momentum=0.9, weight_decay=5e-4
- Learning rate: 0.1 initial, step or cosine schedule (paper uses schedule in Methods)
- Batch size: 128
- Epochs: 200
- Contrastive weight: 0.5–1.0 (tune per experiment)
- Attack (PGD): eps=8/255, steps=10–20, step-size=2/255

Match the exact values reported in the paper for reproducing numbers in tables/figures.

## Reproducibility notes

- Use matching `torch`/`torchvision` binaries for your CUDA version to avoid device errors.
- Ensure model architecture names match checkpoints (e.g., `resnet50` vs. custom `myresnet50`).
- Notebooks in `mm/` demonstrate analysis and plotting used to produce figures; they are a good starting point for verification.

## Results and artifacts

- Training logs and example accuracy outputs: `resnet50_acc.txt`, `resnet50_acc (copy)` in repo root.
- Visualizations and perturbed images: `mm/perturbed_image/` and notebooks under `mm/`.

## Troubleshooting

- CUDA / PyTorch mismatches are the most common issue — install `torch` matching your GPU drivers.
- If a script errors on a missing argument, run it with `--help` to discover required flags.

## License

See the top-level `LICENSE` file for licensing and redistribution terms.

## Contact

For questions about reproducing experiments or clarifications on the C-LEAD method, please contact the paper authors (see the paper for author emails). Issues and pull requests are welcome; include environment details and the commands you ran.
