import torch
import torch.nn.functional as F

def info_nce_loss(features, temperature=0.1, reduction='mean'):

    N = int(features.shape[0]/2)

    labels = torch.cat([torch.arange(N) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / temperature
    loss = F.cross_entropy(logits, labels, reduction=reduction)

    return loss

def supervised_contrastive_loss(features, labels, temperature=0.07, reduction='mean'):
    '''
    The original supervised contrastive loss: OE samples are taken as an extra class.
    Args:
        features: features of ID_tail and OE samples. Tensor. Shape=(N,2,d)
        labels: labels of ID_tail and OE samples. Tensor. Shape=(N)
    '''

    features = features.view(features.shape[0], features.shape[1], -1) # shape=(N,2,d), i.e., 2 views

    features = F.normalize(features, dim=-1)

    batch_size = features.shape[0]
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(labels.device) # shape=(N,N). 1 -> positive pair

    contrast_count = features.shape[1] # = 2
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # shape=(N*2,d)
    anchor_feature = contrast_feature
    anchor_count = contrast_count # = 2

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature) # shape=(2N,2N+N_ood)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # dim=1 is the KL dim.
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count) # shape=(2*N,2*N)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(labels.device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - mean_log_prob_pos
    if reduction == 'mean':
        loss = loss.view(anchor_count, batch_size).mean()
    elif reduction == 'none':
        loss = loss.view(anchor_count, batch_size)

    return loss

def jsd_loss(logits1, logits2, logits3):
    p1, p2, p3 = F.softmax(logits1, dim=1), F.softmax(logits2, dim=1), F.softmax(logits3, dim=1)
    log_p_mixture = torch.clamp((p1 + p2 + p3) / 3., 1e-7, 1).log()
    loss = (
        F.kl_div(log_p_mixture, p1, reduction='batchmean') +
        F.kl_div(log_p_mixture, p2, reduction='batchmean') +
        F.kl_div(log_p_mixture, p3, reduction='batchmean')
        ) / 3.
    return loss