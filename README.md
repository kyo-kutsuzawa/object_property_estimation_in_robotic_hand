# Scripts for "Learning-based Object's Stiffness and Shape Estimation with Confidence Level in Multi-Fingered Hand Grasping"

## Collect datasets

```
python src/collect_dataset.py --n-workers 15 --num-episodes 10 --dataset 20 --data-name train --num-envs 150
python src/collect_dataset.py --n-workers 15 --num-episodes 10 --dataset 22 --data-name test --num-envs 21
python src/collect_dataset.py --n-workers 15 --num-episodes 10 --dataset 23 --data-name test_big --num-envs 21
python src/collect_dataset.py --n-workers 15 --num-episodes 10 --dataset 22 --data-name test_noise_2_2 --num-envs 21 --noise-pos 2e-4 --noise-vel 2e-4
python src/collect_dataset.py --n-workers 15 --num-episodes 10 --dataset 22 --data-name test_noise_8_8 --num-envs 21 --noise-pos 8e-4 --noise-vel 8e-4
python src/collect_dataset.py --n-workers 15 --num-episodes 10 --dataset 24 --data-name train_full --num-envs 450

```

* The raw data (generated datasets) supporting the conclusions of the article will be made available by the authors, without undue reservation.

## Show a recorded trajectory

```
python src/show_data.py
```

## Train neural networks

Train a proposed model with the _standard_ dataset:
```
python src/train.py --datadir dataset/train.pickle --outdir results/result_proposed_standard --augment-online --device cuda
```

Train baseline models with the _standard_ dataset varying a hyper-parameter:
```
python src/train.py --datadir dataset/train.pickle --outdir results/result_baseline_10e1 --method baseline --loss_weight 10.0 --augment-online --device cuda
python src/train.py --datadir dataset/train.pickle --outdir results/result_baseline_10e0 --method baseline --loss_weight 1.0 --augment-online --device cuda
python src/train.py --datadir dataset/train.pickle --outdir results/result_baseline_10e-3 --method baseline --loss_weight 0.001 --augment-online --device cuda
python src/train.py --datadir dataset/train.pickle --outdir results/result_baseline_10e-4 --method baseline --loss_weight 0.0001 --augment-online --device cuda
python src/train.py --datadir dataset/train.pickle --outdir results/result_baseline_10e-5 --method baseline --loss_weight 0.00001 --augment-online --device cuda
```

Train a proposed model with the _full_ dataset:
```
python src/train.py --datadir dataset/train_full.pickle --outdir results/result_proposed_full --augment-online --device cuda
```

* The raw data (trained neural networks) supporting the conclusions of the article will be made available by the authors, without undue reservation.

## Test trained neural networks (save results)

Evaluate the proposed model trained on the _standard_ dataset:
```
python src/evaluate.py results/result_proposed_standard/nn_latest.pth dataset/test.pickle results/result_proposed_standard/result_test.pickle
python src/evaluate.py results/result_proposed_standard/nn_latest.pth dataset/test_big.pickle results/result_proposed_standard/result_test_big.pickle
python src/evaluate.py results/result_proposed_standard/nn_latest.pth dataset/test_noise_2_2.pickle results/result_proposed_standard/result_test_noise_2_2.pickle
python src/evaluate.py results/result_proposed_standard/nn_latest.pth dataset/test_noise_8_8.pickle results/result_proposed_standard/result_test_noise_8_8.pickle
```

## Show results

```
python src/show_result.py results/result_proposed_standard/result_test.pickle --figname results/result_proposed_standard/result_test.pdf --no-plot
python src/show_result.py results/result_proposed_standard/result_test_big.pickle --figname results/result_proposed_standard/result_big.pdf --no-plot
python src/show_result.py results/result_proposed_standard/result_test_noise_2_2.pickle --figname results/result_proposed_standard/result_noise_2_2.pdf --no-plot
python src/show_result.py results/result_proposed_standard/result_test_noise_8_8.pickle --figname results/result_proposed_standard/result_noise_8_8.pdf --no-plot
```

```
python src/show_result_400.py results/result_proposed_standard/result_test.pickle --figname results/result_proposed_standard/result_test_400.pdf --no-plot
```

```
python src/show_variance.py --dirname results/result_proposed_standard --no-plot
python src/show_variance2.py --dirname results/result_proposed_standard --no-plot
python src/show_entropy.py --dirname results/result_proposed_standard --no-plot
```

## Take snapshots

```
python src/test_env.py
```

```
python src/collect_dataset.py --dataset 20 --num-envs 150
```

## Record videos

```
python src/evaluate_with_video.py results/result_proped_standard/result_test.pickle dataset/test.pickle --outdir video/iid_dataset_\(trained_on_standard_dataset\).mp4
python src/evaluate_with_video.py results/result_proped_standard/result_test_big.pickle dataset/test_big.pickle --outdir video/bigger_dataset_\(trained_on_standard_dataset\).mp4 --big
python src/evaluate_with_video.py results/result_proped_standard/result_test_noise_2_2.pickle dataset/test_noise_2_2.pickle --outdir video/small_noise_dataset_\(trained_on_standard_dataset\).mp4
python src/evaluate_with_video.py results/result_proped_standard/result_test_noise_8_8.pickle dataset/test_noise_8_8.pickle --outdir video/large_noise_dataset_\(trained_on_standard_dataset\).mp4

python src/evaluate_with_video.py results/result_proped_standard/result_test.pickle dataset/test.pickle --outdir video/iid_dataset_\(trained_on_full_dataset\).mp4
python src/evaluate_with_video.py results/result_proped_standard/result_test_big.pickle dataset/test_big.pickle --outdir video/bigger_dataset_\(trained_on_full_dataset\).mp4 --big
python src/evaluate_with_video.py results/result_proped_standard/result_test_noise_2_2.pickle dataset/test_noise_2_2.pickle --outdir video/small_noise_dataset_\(trained_on_full_dataset\).mp4
python src/evaluate_with_video.py results/result_proped_standard/result_test_noise_8_8.pickle dataset/test_noise_8_8.pickle --outdir video/large_noise_dataset_\(trained_on_full_dataset\).mp4
```
