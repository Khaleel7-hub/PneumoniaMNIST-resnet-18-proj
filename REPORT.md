# pneumonia mnist project report

## goal
classify chest x-ray images from the pneumonia mnist dataset with a resnet18 and explain the decisions with shap.

## setup steps
- open `pneumonia_resnet18_colab.ipynb` in google colab.
- change runtime to gpu (runtime > change runtime type > gpu).
- run the first cells to mount google drive and install libraries (torch, medmnist, shap).
- outputs (checkpoints + plots) are saved to `/gdrive/MyDrive/pneumonia_mnist_resnet18` by default.

## data
- dataset: `PneumoniaMNIST` from medmnist (preprocessed 28x28 grayscale).
- transforms: tensor + normalize to mean 0.5, std 0.5.
- splits: train/val/test loaders with batch size 128.
- quick sanity check: the notebook shows 10 sample images (labels 0 normal, 1 pneumonia).

## model + transfer learning plan
- base model: torchvision `resnet18` with imagenet weights.
- input change: first conv adjusted to accept one channel (weights averaged across rgb channels).
- head change: final fc -> 2 classes.
- transfer learning experiments coded:
  - `freeze_to_layer3`: freeze stem + layers 1-3, train layer4 + head.
  - `freeze_to_layer2`: freeze stem + layers 1-2, train layers 3-4 + head.
  - `full_finetune`: unfreeze everything.
- each run trains for 10 epochs (can increase if gpu time allows) with adam lr 1e-3, weight decay 1e-4.
- best checkpoint per config saved; best validation accuracy picked for test evaluation.

## evaluation steps
- load best checkpoint and score on test set.
- metrics: accuracy + roc auc.
- plots: confusion matrix (see `confusion_matrix.png`) and roc curve (`roc_curve.png`).
- use these figures in the write-up to talk about false positives/negatives and trade-off.

## shap explainability
- gradient explainer with 50 background images from train set.
- local: one correct normal case and one incorrect pneumonia case saved as `correct_normal_shap.png` and `incorrect_pneumonia_shap.png`.
- global: mean absolute shap map over 64 test images saved as `global_shap_mean.png`.
- interpretation tips:
  - bright red regions push prediction toward pneumonia; blue pushes toward normal.
  - if incorrect case highlights borders or noise, consider more augmentation or regularization.

## challenges + tips
- if validation accuracy stalls when many layers are frozen, try unfreezing layer3 or using a smaller learning rate for pretrained layers (two optimizers or param groups).
- class imbalance is mild but you can test weighted loss if false negatives stay high.
- keep seeds fixed (set in notebook) for repeatable runs.

## what to include in your final report
- mention hardware (colab gpu) and runtime duration.
- list the three freeze settings and note which one won (use numbers from notebook run).
- include test accuracy, roc auc, confusion matrix, roc curve, and the three shap figures.
- describe the two shap cases: where the model focused and whether it matched clinical lung regions.
- reflect on any misclassifications and how you might improve (more epochs, augmentation, learning rate scheduling).
