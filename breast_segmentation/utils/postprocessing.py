"""Post-processing utilities for breast segmentation."""

import numpy as np
import torch
import cv2
from skimage.measure import label as label_fn, regionprops
from skimage import morphology
from typing import Optional, Tuple, Union
from monai.transforms import KeepLargestConnectedComponent, RemoveSmallObjects


def fuse_segmentations(model1_prob, model2_prob, prob_threshold=0.5, boost_factor=1.5, penalty_factor=0.5, kernel_size=3):
    """
    Fuse segmentations from two models by combining their probability maps.
    
    Args:
        model1_prob: Probability map from first model
        model2_prob: Probability map from second model  
        prob_threshold: Threshold for considering agreement
        boost_factor: Factor to boost probabilities where models agree
        penalty_factor: Factor to penalize probabilities where models disagree
        kernel_size: Size of dilation kernel for enlarging agreement areas
        
    Returns:
        Fused probability map
    """
    model1_prob = np.squeeze(model1_prob)
    model2_prob = np.squeeze(model2_prob)
    
    # Step 1: Check where both models agree above the probability threshold
    agreement = np.logical_and(model1_prob > prob_threshold, model2_prob > prob_threshold)
    
    # Step 2: Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Step 3: Dilate the agreement area to enlarge it
    enlarged_agreement = cv2.dilate(agreement.astype(np.uint8), kernel)
    
    # Step 4: Sum the probabilities of both models
    prob_sum = model1_prob + model2_prob
    
    # Step 5: Boost the probability sum where there is agreement
    prob_sum[enlarged_agreement > 0] *= boost_factor
    
    # Step 6: Identify disagreement (where there is no enlarged agreement)
    disagreement = enlarged_agreement == 0
    
    # Step 7: Apply penalty factor where there is disagreement
    prob_sum[disagreement] *= penalty_factor
    
    # Step 8: Normalize the probability sum to get the fused probability, ensuring it's within [0, 1]
    fused_prob = np.clip(prob_sum / 2.0, 0, 1)
    
    return fused_prob
