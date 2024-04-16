# Copyright (c) 2024, SliceX AI, Inc.


def count_parameters(model):
    """Count the number of parameters in the model."""
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params+=params
    
    print(f"Total Trainable Params: {total_params}")
    return total_params
    

def batchify(lst, n):
    """Divide a list into chunks of size n."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]

