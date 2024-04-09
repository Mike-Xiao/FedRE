import torch
def aggregate_clients(w_locals, psi_scores, global_model):
    """
    根据PSI分数聚合客户端模型参数。
    """
    w_avg = {name: torch.zeros_like(param) for name, param in global_model.named_parameters()}
    
    # 计算每个客户端每层的PSI比例
    layer_psis = {}
    for client_id, client_model in enumerate(w_locals):
        for name, param in client_model.named_parameters():
            if param.requires_grad:
                # 假设psi_scores[client_id][name]是该客户端该层的PSI分数
                psi_score = psi_scores[client_id][name]
                if name not in layer_psis:
                    layer_psis[name] = []
                layer_psis[name].append(psi_score)
    
    # 计算权重并聚合
    for name, param in global_model.named_parameters():
        if param.requires_grad:
            # 计算该层所有客户端的PSI分数总和
            layer_psi_sum = sum(layer_psis[name])
            # 计算每个客户端的PSI比例
            psi_ratios = [score / layer_psi_sum for score in layer_psis[name]]
            # 计算权重（使用比例的倒数）
            weights = [1.0 / ratio if ratio > 0 else 0 for ratio in psi_ratios]
            # 归一化权重
            total_weight = sum(weights)
            normalized_weights = [weight / total_weight for weight in weights]
            
            # 根据权重累加客户端的参数
            for i, client_param in enumerate(w_locals):
                w_avg[name] += client_param[name] * normalized_weights[i]
    
    return w_avg