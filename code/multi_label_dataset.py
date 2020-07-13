from torch.utils.data import Dataset
import torch
import numpy as np

class MultiLabelDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, args):
        # assert all(len(lists[0]) == len(list) for list in lists)
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.num_labels = args.num_labels
        self.pos_label = args.pos_label

    def __getitem__(self, index):
        # temp_tuple = tuple(list[index] for list in self.lists)
        input_ids = torch.as_tensor(self.input_ids[index], dtype=torch.long)
        input_mask = torch.as_tensor(self.input_mask[index], dtype=torch.long)
        segment_ids = torch.as_tensor(self.segment_ids[index], dtype=torch.long)
        # label_ids = self._get_multi_hot_label(self.label_ids[index])
        label_ids = torch.as_tensor(self._get_pad_label(self.label_ids[index]), dtype=torch.long)

        return input_ids, input_mask, segment_ids, label_ids

    def __len__(self):
        return len(self.input_ids)

    def _get_multi_hot_label(self, doc_labels):

        temp_array = np.zeros(self.num_labels,dtype= np.float32 )
        temp_array[doc_labels] = 1
        return torch.from_numpy(temp_array)

    def _get_pad_label(self, doc_labels):

        num_pad = self.pos_label - len(doc_labels)
        # if num_pad > 0:
        if len(doc_labels) > 0:
            idx_pad = [doc_labels[0]] * num_pad
        else:
            idx_pad = [-1] * num_pad
        doc_labels.extend(idx_pad)

        return doc_labels