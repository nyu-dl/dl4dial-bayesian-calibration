#!/usr/bin/env python
import logging
import argparse

import numpy
import torch
import pyro
import pyro.infer
import pyro.infer.mcmc
import pyro.distributions as dist


def get_arguments():

    parser = argparse.ArgumentParser(description='Bayesian Calibration arguments')
    parser.add_argument('--logging-level', default='WARNING', choices=['WARNING', 'DEBUG'])
    parser.add_argument('--input-array', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=150)
    parser.add_argument('--num-warmup-samples', type=int, default=100)

    args = parser.parse_args()

    return args


def sigmoid(x):
    return 1./(1. + numpy.exp(-x))


def model_single_score(data, config):
    """
    p(m_a) ~ U(0,3): prior for each model score
    p(s_a) ~ N(s_a; m_a, 1^2): prior for each score data point given model sampled mean
    """
    zm = []
    for mi in range(config['n_models']):
        mu_ = pyro.sample("model-mean-{}".format(mi), dist.Uniform(0., 3.))
        zm.append(pyro.sample("model-{}".format(mi), dist.Normal(mu_, 1.)))

    """
    p(s_t) ~ N(s_t; 0, 1^2): prior score for each annotator, no bias by default
    """
    tm = []
    for ti in range(config['n_turkers']):
        tm.append(pyro.sample("turker-mean-{}".format(ti), dist.Normal(0., 1.)))

    """
    p(s|a, t) = N(s, s_a + s_t, 1^2): likelihood mean for each score given by annotator
    t for model a
    """
    mu = []
    for ii, sc in enumerate(data):
        mu.append(zm[int(sc[0])] + tm[int(sc[1])]) # original
    mu_ = torch.stack(mu)

    return pyro.sample("scores", dist.Normal(mu_, 1.))


def infer(data, config):
    observed_single_scores = torch.Tensor([tup[2] for tup in data])
    single_score_condition = pyro.condition(model_single_score, data={'scores': observed_single_scores})
    nuts_kernel = pyro.infer.mcmc.NUTS(single_score_condition, adapt_step_size=True, step_size=0.1)
    mcmc_run = pyro.infer.mcmc.MCMC(nuts_kernel, num_samples=config['num-samples'], warmup_steps=config['warmup-steps']).run(data, config)
    score_marginal = pyro.infer.EmpiricalMarginal(mcmc_run, sites=["model-{}".format(mi) for mi in range(config['n_models'])])

    return score_marginal.mean, score_marginal.stddev


def prepare_data(args):
    data = numpy.load(args.input_array)

    # we assume models and annotator indexing from 0
    n_turkers = max([a[1] for a in data])+1
    n_models = max([a[0] for a in data])+1
    
    config = {
        'logging-level': args.logging_level,
        'num-samples': args.num_samples,
        'warmup-steps': args.num_warmup_samples,
        'n_models': n_models,
        'n_turkers': n_turkers,
    }
    return config, data


def main():
    args = get_arguments()
    config, data = prepare_data(args)

    mean, std = infer(data, config)
    print('Empirical mean: {}\n\n'.format(mean))
    print('Empirical std: {}\n\n'.format(std))


if __name__ == '__main__':
    main()


