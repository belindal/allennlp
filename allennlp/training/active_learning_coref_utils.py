import pdb
import torch

def translate_to_indA(edges, output_dict, all_spans, translation_reference = None) -> torch.LongTensor:
    """
    :param edges: Tensor (Nx3) of N edges, each of form [instance in batch, indB of proform, indC of antecedent]
    output_dict: holds information for translation
    :param translation_reference: Tensor (BxS) of B batches and S top_spans, whereby number at [b,s] represents index of
        to translate sth *top* span in batch b in *all* spans
    :return: Tensor (Nx3) of N edges, which indices translated to indA. Each edge has form
    [instance in batch, indA of proform, indA of antecedent]
    """
    if len(edges) == 0:
        return edges

    # NOTE in below code-- each span has 3 indices, will refer to them as A,B,C respectively:
    #   indA. index in all spans (i.e. in the training data)
    #   indB. index in output_dict['top_spans']
    #   indC. index in output_dict['predicted_antecedents'] (antecedents only)
    # note output_dict['antecedent_indices'] translates indB <-> indC by:
    # indB = output_dict['antecedent_indices'][instance, proform idx, indC]
    if translation_reference is not None:
        indA_edges = edges.clone()
        indA_edges[:,2] = output_dict['antecedent_indices'][indA_edges[:,0], indA_edges[:,1], indA_edges[:,2]]
        indA_edges[:,1] = translation_reference[indA_edges[:,0], indA_edges[:,1]]
        indA_edges[:,2] = translation_reference[indA_edges[:,0], indA_edges[:,2]]
        return indA_edges

    instances = edges[:, 0]
    ind_proforms = edges[:, 1]
    ind_antecedents = edges[:, 2]
    ind_antecedents = output_dict['antecedent_indices'][instances, ind_proforms, ind_antecedents] #indB

    proform_spans = output_dict['top_spans'][instances, ind_proforms]
    antecedent_spans = output_dict['top_spans'][instances, ind_antecedents]
    chunk_size = 10000
    if len(proform_spans) > chunk_size:
        # too big for cuda, break into chunks
        indA_proforms = torch.empty(instances.size(), dtype=torch.long, device=proform_spans.device)
        indA_antecedents = torch.empty(instances.size(), dtype=torch.long, device=proform_spans.device)
        i = 0
        while i < len(proform_spans):
            try:
                instances_chunk = instances[i:i+chunk_size]
                proform_span_chunk = proform_spans[i:i+chunk_size]
                antecedent_span_chunk = antecedent_spans[i:i+chunk_size]
                indA_proforms[i:i+chunk_size] = ((proform_span_chunk.unsqueeze(1) - all_spans[instances_chunk]).abs().sum(-1) == 0).nonzero()[:, 1]
                indA_antecedents[i:i+chunk_size] = ((antecedent_span_chunk.unsqueeze(1) - all_spans[instances_chunk]).abs().sum(-1) == 0).nonzero()[:, 1]
                i += chunk_size
                # should use < 75% of GPU memory
                assert(torch.cuda.memory_cached(instances.device.index) + torch.cuda.memory_allocated(instances.device.index)
                       < 0.75 * torch.cuda.get_device_properties(instances.device.index).total_memory)
            except:
                    torch.cuda.empty_cache()
                    chunk_size = int(chunk_size / 2)
        else:
            indA_proforms = ((proform_spans.unsqueeze(1) - all_spans[instances]).abs().sum(-1) == 0).nonzero()[:, 1]
            indA_antecedents = ((antecedent_spans.unsqueeze(1) - all_spans[instances]).abs().sum(-1) == 0).nonzero()[:, 1]
        return torch.stack([instances, indA_proforms, indA_antecedents], dim=-1)

def filter_gold_cluster_edges(chosen_edges, span_labels):
    """
    :param chosen_edges: edges chosen to verify
    :param span_labels: from batch['span_labels']
    Filter out edges for which both ends are in the same, or different, gold clusters already
    """
    proform_gold_labels = span_labels[chosen_edges[:, 0], chosen_edges[:, 1]]
    antecedent_gold_labels = span_labels[chosen_edges[:, 0], chosen_edges[:, 2]]
    edges_both_ends_in_gold_clusters_mask = (proform_gold_labels != -1) & (antecedent_gold_labels != -1)
    chosen_edges = chosen_edges[~edges_both_ends_in_gold_clusters_mask]
    return chosen_edges

def update_clusters_with_edge(span_labels, edge, delete=False, all_edges=None):
    '''
    delete: whether we are deleting or adding the edge to the clusters
    if delete true, all_edges must be true to allow us to know how to split the cluster
        also, all_edges can be before/after deletion of edge
    '''
    proform_label = span_labels[edge[0], edge[1]].item()
    antecedent_label = span_labels[edge[0], edge[2]].item()

    if delete:
        assert all_edges is not None
        if proform_label != -1 and proform_label == antecedent_label:
            # ONLY make changes after having affirmed both belong to same cluster
            # find edges involving spans in cluster
            spans_in_cluster_mask = (span_labels[edge[0]] == proform_label)
            spans_in_cluster = spans_in_cluster_mask.nonzero()
            edges_in_cluster_mask = ((all_edges[:,1] == spans_in_cluster) | (all_edges[:,2] == spans_in_cluster)).max(0)[0]
            edges_in_cluster = all_edges[edges_in_cluster_mask]

            # delete original cluster
            span_labels[edge[0]][spans_in_cluster_mask] = -1
            decrement_mask = -(span_labels[edge[0]] > proform_label).type(torch.long)
            span_labels[edge[0]] += decrement_mask

            # construct new clusters by adding in edges used to create original cluster
            for remaining_edge in edges_in_cluster:
                if (remaining_edge != edge).sum() > 0:  # (don't add the edge we want to delete)
                    span_labels = update_clusters_with_edge(span_labels, remaining_edge)
        return span_labels

    # NOTE: Do not modify num_gold_clusters field in metadata, which is used to keep track of
    # the original, gold clusters
    if proform_label != -1 and antecedent_label != -1:
        # Case 0: both in clusters (merge clusters iff were newly created, non-gold clusters)
        if proform_label == antecedent_label:
            # If already in same clusters, no need to merge
            return span_labels
        # Otherwise, merge clusters: merge larger cluster id into smaller cluster id
        min_cluster_id = min(proform_label, antecedent_label)
        max_cluster_id = max(proform_label, antecedent_label)

        span_labels[edge[0]][span_labels[edge[0]] == max_cluster_id] = min_cluster_id
        # decrease by 1 the index of all clusters > removed max_cluster in span_labels
        decrement_mask = -(span_labels[edge[0]] > max_cluster_id).type(torch.long)
        span_labels[edge[0]] += decrement_mask
    elif antecedent_label != -1:
        # Case 1: antecedent in cluster, proform not (update proform's label,
        # add proform to cluster)
        span_labels[edge[0], edge[1]] = antecedent_label
    elif proform_label != -1:
        # Case 2: proform in cluster, antecedent not (update antecedent's label,
        # add antecedent to cluster)
        span_labels[edge[0], edge[2]] = proform_label
    else:
        # Case 3: neither in cluster (create new cluster with both)
        cluster_id = span_labels.max() + 1
        span_labels[edge[0], edge[2]] = cluster_id
        span_labels[edge[0], edge[1]] = cluster_id
    return span_labels

