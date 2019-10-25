# dl4dial-bayesian-calibration

This code is running inference with human eval data using MCMC with NUTS sampler. It returns the mean and std of the empirical score distribution for each model.

Check our paper for more details:

https://uralik.github.io/beamdream/

https://arxiv.org/abs/1811.00907

## Dependencies
* Python 3
* Pyro (0.3.2)
* Pytorch
* Numpy

## Example dataset

In `mtbeam_human_eval.npz.npy` you can find an example matrix of human scores assigned to conversations. Each row in the matrix corresponds to a tuple: `(model_id, annotator_id, score)`.

`model_id` and `annotator_id` are integer non-negative values starting from 0. Score is an integer value from 0 to 3.

## Running the script

`python run.py --input-array ./mtbeam_human_eval.npz.npy --num-samples 50 --num-warmup-samples 100`

Sufficient number of samples depends on the size of your data.

## Citation

If you use this calibration method please cite this work as:
```
@article{kulikov2018importance,
  title={Importance of a search strategy in neural dialogue modelling},
  author={Kulikov, Ilya and Miller, Alexander H and Cho, Kyunghyun and Weston, Jason},
  journal={arXiv preprint arXiv:1811.00907},
  year={2018}
}
```
