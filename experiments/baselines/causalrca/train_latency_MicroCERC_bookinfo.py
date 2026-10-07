#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Sun Apr 10 20:33:33 2022"""

import time
from utils_microarcl import *
from typing import List
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import *
from modules import *
from config import CONFIG
import warnings
import argparse

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--indx', type=int, default=0, help='index')
parser.add_argument('--atype', type=str, default='cpu-hog1_', help='anomaly type')
parser.add_argument('--gamma', type=float, default=0.25, help='gamma')
parser.add_argument('--eta', type=int, default=10, help='eta')
args = parser.parse_args()

CONFIG.cuda = torch.cuda.is_available()
CONFIG.factor = not CONFIG.no_factor




idx = args.indx
atype = args.atype


class Simple:
    def __init__(self, global_now_time, global_end_time, label, root_cause, dir):
        self.global_now_time = global_now_time
        self.global_end_time = global_end_time
        self.label = label
        self.root_cause = root_cause
        self.dir = dir



def _h_A(A, m):
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


prox_plus = torch.nn.Threshold(0., 0.)


def stau(w, tau):
    w1 = prox_plus(torch.abs(w) - tau)
    return torch.sign(w) * w1


def update_optimizer(optimizer, original_lr, c_A):
    """related LR to c_A, whenever c_A gets big, reduce LR proportionally

Inputs: optimizer, original_lr, c_A. Outputs: result."""
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr


    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr





def train(epoch, best_val_loss, lambda_A, c_A, optimizer):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_trian = []

    encoder.train()
    decoder.train()
    scheduler.step()


    optimizer, lr = update_optimizer(optimizer, CONFIG.lr, c_A)

    for i in range(1):
        data = train_data[i * data_sample_size:(i + 1) * data_sample_size]
        data = torch.tensor(data.to_numpy().reshape(data_sample_size, data_variable_size, 1))
        if CONFIG.cuda:
            data = data.cuda()
        data = Variable(data).double()

        optimizer.zero_grad()

        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(
            data)
        edges = logits

        dec_x, output, adj_A_tilt_decoder = decoder(data, edges, data_variable_size * CONFIG.x_dims, origin_A,
                                                    adj_A_tilt_encoder, Wa)

        if torch.sum(output != output):
            print('nan error\n')

        target = data
        preds = output
        variance = 0.


        loss_nll = nll_gaussian(preds, target, variance)


        loss_kl = kl_gaussian_sem(logits)


        loss = loss_kl + loss_nll

        one_adj_A = origin_A
        sparse_loss = CONFIG.tau_A * torch.sum(torch.abs(one_adj_A))


        if CONFIG.use_A_connect_loss:
            connect_gap = A_connect_loss(one_adj_A, CONFIG.graph_threshold, z_gap)
            loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

        if CONFIG.use_A_positiver_loss:
            positive_gap = A_positive_loss(one_adj_A, z_positive)
            loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)


        h_A = _h_A(origin_A, data_variable_size)
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(
            origin_A * origin_A) + sparse_loss


        loss.backward()
        loss = optimizer.step()

        myA.data = stau(myA.data, CONFIG.tau_A * lr)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')


        graph = origin_A.data.clone().cpu().numpy()
        graph[np.abs(graph) < CONFIG.graph_threshold] = 0

        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())

    return np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A





if __name__ == '__main__':

    pps = ['p50']
    simples: List[Simple] = [
        Simple(
            1705125240, 1705125960, 'label-details-cpu-load-1', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-1'
        ),
        Simple(
            1705126080, 1705126800, 'label-details-cpu-load-2', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-2'
        ),
        Simple(
            1705126920, 1705127640, 'label-details-cpu-load-3', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-3'
        ),
        Simple(
            1706024880, 1706025660, 'label-details-cpu-load-4', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-4'
        ),
        Simple(
            1706025780, 1706026560, 'label-details-cpu-load-5', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-5'
        ),
        Simple(
            1706026680, 1706027460, 'label-details-cpu-load-6', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-6'
        ),
        Simple(
            1706027580, 1706028360, 'label-details-cpu-load-7', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-7'
        ),
        Simple(
            1706028480, 1706029260, 'label-details-cpu-load-8', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-8'
        ),
        Simple(
            1705129320, 1705130040, 'label-details-mem-load-1', 'details',
            'abnormal/bookinfo/details/bookinfo-details-mem-1'
        ),
        Simple(
            1705130160, 1705130880, 'label-details-mem-load-2', 'details',
            'abnormal/bookinfo/details/bookinfo-details-mem-2'
        ),
        Simple(
            1706029380, 1706030160, 'label-details-mem-load-3', 'details',
            'abnormal/bookinfo/details/bookinfo-details-mem-3'
        ),
        Simple(
            1706030280, 1706031060, 'label-details-mem-load-4', 'details',
            'abnormal/bookinfo/details/bookinfo-details-mem-4'
        ),
        Simple(
            1706031180, 1706031960, 'label-details-mem-load-5', 'details',
            'abnormal/bookinfo/details/bookinfo-details-mem-5'
        ),
        Simple(
            1706032080, 1706032860, 'label-details-mem-load-6', 'details',
            'abnormal/bookinfo/details/bookinfo-details-mem-6'
        ),
        Simple(
            1706032980, 1706033760, 'label-details-mem-load-7', 'details',
            'abnormal/bookinfo/details/bookinfo-details-mem-7'
        ),
        Simple(
            1705131180, 1705131900, 'label-details-net-latency-1', 'details',
            'abnormal/bookinfo/details/bookinfo-details-net-1'
        ),
        Simple(
            1705132020, 1705132740, 'label-details-net-latency-2', 'details',
            'abnormal/bookinfo/details/bookinfo-details-net-2'
        ),
        Simple(
            1705132860, 1705133580, 'label-details-net-latency-3', 'details',
            'abnormal/bookinfo/details/bookinfo-details-net-3'
        ),
        Simple(
            1706033880, 1706034660, 'label-details-net-latency-4', 'details',
            'abnormal/bookinfo/details/bookinfo-details-net-4'
        ),
        Simple(
            1706034780, 1706035560, 'label-details-net-latency-5', 'details',
            'abnormal/bookinfo/details/bookinfo-details-net-5'
        ),
        Simple(
            1706035680, 1706036460, 'label-details-net-latency-6', 'details',
            'abnormal/bookinfo/details/bookinfo-details-net-6'
        )
    ]
    namespaces = ['bookinfo', 'hipster', 'cloud-sock-shop', 'horsecoder-test']
    for pp in pps:
        for simple in simples:
            print(simple.label)
            all_data = pd.DataFrame()
            for namespace in namespaces:
                all_data_ns = pd.read_csv(
                    '/Users/zhuyuhan/Documents/391-WHU/experiment/researchProject/MicroCERC/data/' + simple.dir + '/' + namespace + '/latency.csv')
                all_data_ns = df_time_limit_normalization(all_data_ns, simple.global_now_time, simple.global_end_time)
                if all_data.empty:
                    all_data = all_data_ns
                else:
                    all_data = pd.merge(all_data, all_data_ns, on='timestamp', how='outer')
            name = [i for i in all_data.columns if i != 'timestamp' and pp in i]
            data = all_data[name]

            data_sample_size = data.shape[0]
            data_variable_size = data.shape[1]







            train_data = data





            off_diag = np.ones([data_variable_size, data_variable_size]) - np.eye(data_variable_size)


            num_nodes = data_variable_size
            adj_A = np.zeros((num_nodes, num_nodes))

            if CONFIG.encoder == 'mlp':
                encoder = MLPEncoder(data_variable_size * CONFIG.x_dims, CONFIG.x_dims, CONFIG.encoder_hidden,
                                     int(CONFIG.z_dims), adj_A,
                                     batch_size=CONFIG.batch_size,
                                     do_prob=CONFIG.encoder_dropout, factor=CONFIG.factor).double()
            elif CONFIG.encoder == 'sem':
                encoder = SEMEncoder(data_variable_size * CONFIG.x_dims, CONFIG.encoder_hidden,
                                     int(CONFIG.z_dims), adj_A,
                                     batch_size=CONFIG.batch_size,
                                     do_prob=CONFIG.encoder_dropout, factor=CONFIG.factor).double()

            if CONFIG.decoder == 'mlp':
                decoder = MLPDecoder(data_variable_size * CONFIG.x_dims,
                                     CONFIG.z_dims, CONFIG.x_dims, encoder,
                                     data_variable_size=data_variable_size,
                                     batch_size=CONFIG.batch_size,
                                     n_hid=CONFIG.decoder_hidden,
                                     do_prob=CONFIG.decoder_dropout).double()
            elif CONFIG.decoder == 'sem':
                decoder = SEMDecoder(data_variable_size * CONFIG.x_dims,
                                     CONFIG.z_dims, 2, encoder,
                                     data_variable_size=data_variable_size,
                                     batch_size=CONFIG.batch_size,
                                     n_hid=CONFIG.decoder_hidden,
                                     do_prob=CONFIG.decoder_dropout).double()




            if CONFIG.optimizer == 'Adam':
                optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=CONFIG.lr)
            elif CONFIG.optimizer == 'LBFGS':
                optimizer = optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()),
                                        lr=CONFIG.lr)
            elif CONFIG.optimizer == 'SGD':
                optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()),
                                      lr=CONFIG.lr)

            scheduler = lr_scheduler.StepLR(optimizer, step_size=CONFIG.lr_decay,
                                            gamma=CONFIG.gamma)


            triu_indices = get_triu_offdiag_indices(data_variable_size)
            tril_indices = get_tril_offdiag_indices(data_variable_size)

            if CONFIG.prior:
                prior = np.array([0.91, 0.03, 0.03, 0.03])
                print("Using prior")
                print(prior)
                log_prior = torch.DoubleTensor(np.log(prior))
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = Variable(log_prior)

                if CONFIG.cuda:
                    log_prior = log_prior.cuda()

            if CONFIG.cuda:
                encoder.cuda()
                decoder.cuda()
                triu_indices = triu_indices.cuda()
                tril_indices = tril_indices.cuda()
            gamma = args.gamma
            eta = args.eta

            t_total = time.time()
            best_ELBO_loss = np.inf
            best_NLL_loss = np.inf
            best_MSE_loss = np.inf
            best_epoch = 0
            best_ELBO_graph = []
            best_NLL_graph = []
            best_MSE_graph = []

            c_A = CONFIG.c_A
            lambda_A = CONFIG.lambda_A
            h_A_new = torch.tensor(1.)
            h_tol = CONFIG.h_tol
            k_max_iter = int(CONFIG.k_max_iter)
            h_A_old = np.inf

            E_loss = []
            N_loss = []
            M_loss = []
            start_time = time.time()
            try:
                for step_k in range(k_max_iter):

                    while c_A < 1e+20:
                        for epoch in range(CONFIG.epochs):

                            ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(epoch, best_ELBO_loss, lambda_A, c_A,
                                                                                   optimizer)
                            E_loss.append(ELBO_loss)
                            N_loss.append(NLL_loss)
                            M_loss.append(MSE_loss)
                            if ELBO_loss < best_ELBO_loss:
                                best_ELBO_loss = ELBO_loss
                                best_epoch = epoch
                                best_ELBO_graph = graph

                            if NLL_loss < best_NLL_loss:
                                best_NLL_loss = NLL_loss
                                best_epoch = epoch
                                best_NLL_graph = graph

                            if MSE_loss < best_MSE_loss:
                                best_MSE_loss = MSE_loss
                                best_epoch = epoch
                                best_MSE_graph = graph



                        if ELBO_loss > 2 * best_ELBO_loss:
                            break


                        A_new = origin_A.data.clone()
                        h_A_new = _h_A(A_new, data_variable_size)
                        if h_A_new.item() > gamma * h_A_old:
                            c_A *= eta
                        else:
                            break



                    h_A_old = h_A_new.item()
                    lambda_A += c_A * h_A_new.item()

                    if h_A_new.item() <= h_tol:
                        break









                graph = origin_A.data.clone().cpu().numpy()
                graph[np.abs(graph) < 0.1] = 0
                graph[np.abs(graph) < 0.2] = 0
                graph[np.abs(graph) < 0.3] = 0

            except KeyboardInterrupt:
                print('Done!')

            end_time = time.time()

            adj = graph
            if not np.any(adj):
                print(simple.label + ' is absent')
                continue
            org_G = nx.from_numpy_array(adj, parallel_edges=True, create_using=nx.DiGraph)










            from sknetwork.ranking import PageRank

            pagerank = PageRank()
            scores = pagerank.fit_transform(np.abs(adj.T))

            score_dict = {}
            for i, s in enumerate(scores):
                score_dict[name[i]] = s
            sorted_scores = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
            count = 0
            with open('MicroCERC/bookinfo/service/' + simple.label + '-' + pp + '.log', "a") as output_file:
                print('root cause: ' + simple.root_cause, file=output_file)
                for sorted_score in sorted_scores:
                    count += 1
                    print(sorted_score, file=output_file)
                    if ('edge' in simple.root_cause and simple.root_cause in sorted_score[0]) or (
                            'edge' not in simple.root_cause and simple.root_cause in sorted_score[0] and 'edge' not in
                            sorted_score[0]):
                        print("topK: " + str(count), file=output_file)
                        break
