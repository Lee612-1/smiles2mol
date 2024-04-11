
def split_for_gpu(data, num_gpus):
    # Calculate the amount of text that needs to be allocated to each GPU
    if num_gpus >4:
        num_gpus = 4
    texts_per_gpu_floor = len(texts_to_generate) // num_gpus
    texts_per_gpu_remainder = len(texts_to_generate) % num_gpus
    texts_per_gpu_list = [texts_per_gpu_floor + 1 if i < texts_per_gpu_remainder else texts_per_gpu_floor for i in range(num_gpus)]
    start = 0
    sublists = []
    for i in range(num_gpus):
        end = start + texts_per_gpu_list[i]
        sublist = texts_to_generate[start:end]
        sublists.append(sublist)
        start = end
    
    return sublists


