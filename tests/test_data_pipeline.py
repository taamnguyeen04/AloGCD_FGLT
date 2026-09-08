import sys
import os
import argparse
import numpy as np
import collections
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cub import get_cub_200_datasets

def test_data_pipeline():
    class Args:
        pass
    
    args = Args()
    args.labeled_classes = 100
    args.imb_ratio = 10
    args.use_lt_split = True
    
    datasets = get_cub_200_datasets(None, None, args=args)
    
    l_set = datasets['train_labelled']
    u_set = datasets['train_unlabelled']
    
    l_targets = np.array(l_set.targets)
    
    # Handle ConcatDataset
    u_targets = np.concatenate([np.array(d.targets) for d in u_set.datasets])
    u_uq_idxs = np.concatenate([d.uq_idxs for d in u_set.datasets])
    
    l_classes = np.unique(l_targets)
    assert len(l_classes) == 100, f"Expected 100 known classes, got {len(l_classes)}"
    
    l_idxs = set(l_set.uq_idxs)
    u_idxs = set(u_uq_idxs)
    assert len(l_idxs.intersection(u_idxs)) == 0, "Overlap found between labelled and unlabelled indices!"
    
    print(f"Total labelled samples: {len(l_set)}")
    print(f"Total unlabelled samples: {len(u_set)}")
    
    l_counts = collections.Counter(l_targets)
    sorted_l_counts = l_counts.most_common()
    print("\nLabelled set class distribution (top 5, bottom 5):")
    print("Head:", sorted_l_counts[:5])
    print("Tail:", sorted_l_counts[-5:])
    
    u_counts = collections.Counter(u_targets)
    sorted_u_counts = u_counts.most_common()
    print("\nUnlabelled set class distribution (top 5, bottom 5):")
    print("Head:", sorted_u_counts[:5])
    print("Tail:", sorted_u_counts[-5:])

if __name__ == '__main__':
    test_data_pipeline()
    print("\nAll Phase 0 tests passed!")
