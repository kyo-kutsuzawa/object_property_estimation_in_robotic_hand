# Scripts for "Learning-based Object's Stiffness and Shape Estimation with Confidence Level in Multi-Fingered Hand Grasping"

## Collect datasets

```
python src/collect_dataset.py --dataset 0 --data-name train --num-envs 600
python src/collect_dataset.py --dataset 2 --data-name test --num-envs 21
python src/collect_dataset.py --dataset 3 --data-name test_big --num-envs 21
python src/collect_dataset.py --dataset 2 --data-name test_noise_2_2 --num-envs 21 --noise-pos 2e-4 --noise-vel 2e-4
python src/collect_dataset.py --dataset 2 --data-name test_noise_8_8 --num-envs 21 --noise-pos 8e-4 --noise-vel 8e-4
```

* The raw data (generated datasets) supporting the conclusions of the article will be made available by the authors, without undue reservation.

## Show a recorded trajectory

```
python src/show_data.py
```

## Train a neural network

```
python src/train.py --datadir dataset/train.pickle
```

* The raw data (a trained neural network) supporting the conclusions of the article will be made available by the authors, without undue reservation.

## Test a trained neural network (save results)

```
python src/evaluate.py result/nn_latest.pth dataset/train.pickle result/result_train.pickle
python src/evaluate.py result/nn_latest.pth dataset/test.pickle result/result_test.pickle
python src/evaluate.py result/nn_latest.pth dataset/test_big.pickle result/result_test_big.pickle
python src/evaluate.py result/nn_latest.pth dataset/test_noise_2_2.pickle result/result_test_noise_2_2.pickle
python src/evaluate.py result/nn_latest.pth dataset/test_noise_8_8.pickle result/result_test_noise_8_8.pickle
```

## Show results

```
python src/show_result.py result/result_test.pickle --figname result/result_test.pdf --no-plot
python src/show_result.py result/result_test_big.pickle --figname result/result_big.pdf --no-plot
python src/show_result.py result/result_test_noise_2_2.pickle --figname result/result_noise_2_2.pdf --no-plot
python src/show_result.py result/result_test_noise_8_8.pickle --figname result/result_noise_8_8.pdf --no-plot
```

```
python src/show_result_400.py result/result_test.pickle --figname result/result_test_400.pdf --no-plot
```

```
python src/show_variance.py --no-plot
python src/show_variance2.py --no-plot
python src/show_entropy.py --no-plot
```

## Take snapshots

```
python src/test_env.py
```

```
python src/collect_dataset.py --dataset 0 --num-envs 600 --snapshot
```

## Record videos

```
python src/record_video.py result/result_test.pickle --outdir video/iid_dataset.mp4
python src/record_video.py result/result_test_big.pickle --outdir video/bigger_dataset.mp4 --big
python src/record_video.py result/result_test_noise_2_2.pickle --outdir video/small_noise_dataset.mp4
python src/record_video.py result/result_test_noise_8_8.pickle --outdir video/large_noise_dataset.mp4
```
