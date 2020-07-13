# -*- coding: utf-8 -*-

"""

source code for APLC which is based on the Pytorch implementation of AdaptiveLogSoftmaxWithLoss

"""

import torch
from torch.nn import Module, BatchNorm1d, LayerNorm
from torch.nn import Sequential, ModuleList, Linear, BCEWithLogitsLoss, Sigmoid, BCELoss



class AdaptiveBCEWithLogitsLoss(Module):
    r"""

    Adaptive Probabilistic Label Clusters is an approximate strategy for training models with large
    outputs. It is most effective when the label distribution is highly
    imbalanced.


    * :attr:`cutoffs` should be an ordered Sequence of integers sorted
      in the increasing order.
      It controls number of clusters and the partitioning of targets into
      clusters. For example setting ``cutoffs = [10, 100, 1000]``
      means that first `10` targets will be assigned
      to the 'head' of the Adaptive Probabilistic Label Clusters, targets `11, 12, ..., 100` will be
      assigned to the first cluster, and targets `101, 102, ..., 1000` will be
      assigned to the second cluster, while targets
      `1001, 1002, ..., n_classes - 1` will be assigned
      to the last, third cluster.

    * :attr:`div_value` is used to compute the dimension of each tail cluster,
      which is given as
      :math:`\left\lfloor\frac{in\_features}{div\_value^{idx}}\right\rfloor`,
      where :math:`idx` is the cluster index (with clusters
      for less frequent words having larger indices,
      and indices starting from :math:`1`).

    * :attr:`head_bias` if set to True, adds a bias term to the 'head' of the
      Adaptive Probabilistic Label Clusters .
    .. warning::
        Labels passed as inputs to this module should be sorted accoridng to
        their frequency. This means that the most frequent label should be
        represented by the index `0`, and the least frequent
        label should be represented by the index `n_classes - 1`.



    Args:
        in_features (int): Number of features in the input tensor
        n_classes (int): Number of classes in the dataset
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets
        div_value (float, optional): value used as an exponent to compute hidden sizes
            of the clusters. Default: 2.0
        head_bias (bool, optional): If ``True``, adds a bias term to the 'head' of the
            Adaptive Probabilistic Label Clusters. Default: ``False``


    """

    def __init__(self, in_features, n_classes, cutoffs, div_value=2., head_bias=False):
        super(AdaptiveBCEWithLogitsLoss, self).__init__()

        cutoffs = list(cutoffs)

        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) > (n_classes - 1)) \
                or (len(set(cutoffs)) != len(cutoffs)) \
                or any([int(c) != c for c in cutoffs]):

            raise ValueError("cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and n_classes-1")

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.cluster_size = []

        self.head = Linear(self.in_features, self.head_size, bias=self.head_bias)
        self.tail = ModuleList()

        for i in range(self.n_clusters):

            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = Sequential(
                Linear(self.in_features, hsz, bias=False),
                LayerNorm(hsz),
                torch.nn.ReLU(),
                # torch.nn.Dropout(p=0.1),
                Linear(hsz, osz, bias=False)
            )

            self.tail.append(projection)
            self.cluster_size.append(osz)

    def reset_parameters(self):
        self.head.reset_parameters()
        for i2h, h2o in self.tail:
            i2h.reset_parameters()
            h2o.reset_parameters()

    def forward(self, input, target):
        if input.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        # used_rows = 0
        batch_size = target.size(0)

        # output = input.new_zeros(batch_size)
        # gather_inds = target.new_empty(batch_size)

        total_cluster_loss = input.new_zeros(batch_size)

        head_onehot = target.new_zeros(batch_size, self.cutoffs[0])
        cluster_onehot = target.new_zeros(batch_size,self.n_clusters)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):

            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]
            num_idx = high_idx - low_idx

            target_mask = (target >= low_idx) & (target < high_idx)
            target_mask_row = torch.sum(target_mask,dim=1)
            row_indices = target_mask_row.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue

            input_subset = input.index_select(0, row_indices)
            target_onehot = self.get_multi_hot_label(target, target_mask, row_indices, low_idx, num_idx).detach()

            if i == 0:
                # indices =  row_indices.repeat(num_idx, 1).transpose(1,0)
                head_onehot.index_copy_(0, row_indices, target_onehot)

            else:
                head_output = self.head(input_subset)
                cluster_root_output = head_output[:,self.shortlist_size + i - 1]

                sig_func = Sigmoid()
                # test = sig_func(cluster_root_output)
                cluster_root_output = torch.diag(sig_func(cluster_root_output))

                cluster_output = self.tail[i - 1](input_subset)

                # cluster_output = cluster_output * cluster_root_output
                cluster_output = torch.mm(cluster_root_output,sig_func(cluster_output))

                # cluster_index = self.shortlist_size + i - 1

                temp_onehot = target.new_zeros(batch_size).index_fill_(0, row_indices, 1)
                cluster_onehot[:,i-1] = temp_onehot

                # loss_fct = BCEWithLogitsLoss(reduction='none')
                loss_fct = BCELoss(reduction='none')

                loss = loss_fct(cluster_output.view(-1, num_idx), target_onehot.view(-1, num_idx).float())
                loss = torch.sum(loss,dim=1)
                # total_cluster_loss = total_cluster_loss.scatter_add(0,row_indices,loss)
                temp_loss = input.new_zeros(batch_size)
                total_cluster_loss += temp_loss.index_copy_(0,row_indices,loss)

        head_output = self.head(input)
        head_onehot = torch.cat((head_onehot,cluster_onehot),dim=1)
        loss_fct = BCEWithLogitsLoss(reduction='none')
        head_loss = loss_fct(head_output.view(-1, self.head_size), head_onehot.view(-1, self.head_size).float())

        cluster_root_loss = head_loss[:,self.shortlist_size:]
        # temp_mask = head_onehot[:,self.shortlist_size:]
        multiplier = (cluster_onehot == 0).long()
        # multiplier += cluster_onehot * torch.tensor(self.cluster_size)
        cluster_root_loss = cluster_root_loss * multiplier.float()

        head_loss[:,self.shortlist_size:] = cluster_root_loss
        head_loss = torch.sum(head_loss, dim=1)

        multiplier += cluster_onehot * torch.tensor(self.cluster_size).cuda()
        num_loss = torch.sum(multiplier, dim=1) + self.shortlist_size

        # loss = (head_loss + total_cluster_loss) / num_loss.float()
        loss = ((head_loss + total_cluster_loss) / num_loss.float()).mean()

        return loss


    def predict(self, input):
        r"""

        Args:
            input (Tensor): a minibatch of examples

        """
        head_output = self.head(input)
        sig_func = Sigmoid()
        head_output = sig_func(head_output)
        output = head_output[:, :self.shortlist_size]

        for i in range(self.n_clusters):
            cluster_root_output = head_output[:, self.shortlist_size + i]
            cluster_root_output = torch.diag(cluster_root_output)

            cluster_output = self.tail[i](input)
            cluster_output = torch.mm(cluster_root_output, sig_func(cluster_output))

            output = torch.cat((output, cluster_output), dim=1)

        return output

    def get_multi_hot_label(self, target, target_mask, row_indices, low_idx, num_idx):

        target_subset = target.index_select(0, row_indices)

        target_mask_subset = target_mask.index_select(0, row_indices)

        target_prime = target_subset * (target_mask_subset.long())

        target_max = torch.max(target_prime, 1)[0]
        target_pad = target_prime.new_ones(target_prime.shape) * target_max.view(-1, 1)
        target_pad = target_pad * ((~ target_mask_subset).long())

        relative_target = target_prime + target_pad - low_idx

        target_onehot = target.new_zeros(target_subset.shape[0], num_idx).scatter(1, relative_target, 1)

        return target_onehot
