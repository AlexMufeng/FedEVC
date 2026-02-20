import numpy as np

# 全局字典缓存 {mask_ratio: masked_nodes_list}
MASK_CACHE = {}


def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def Add_Window_Horizon_Mask(args, data, window=3, horizon=1, mask_ratio=0.0, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    X = X.copy()
    
    # ====== 节点级 Mask，支持缓存 ======
    if mask_ratio > 0:
        num_nodes = X.shape[2]
        num_mask_nodes = int(num_nodes * mask_ratio)

        # 如果该 mask_ratio 已生成过，则复用，否则生成并记录
        if mask_ratio not in MASK_CACHE:
            MASK_CACHE[mask_ratio] = np.random.choice(num_nodes, num_mask_nodes, replace=False)

        mask_nodes = MASK_CACHE[mask_ratio]
        X[:, :, mask_nodes] = 0

        args.logger.info(f"[MASK FIXED] Using same mask nodes for ratio={mask_ratio}: {mask_nodes}")
    return X, Y

def Add_Window_Hyper(data, window, stride, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length -window + 1
    X = []      #windows
  
    index = 0
    if single:
        while index < end_index:
           
            X.append(list(data[index:index+window]))

            index = index + stride
    else:
        while index < end_index:
            
            X.append(list(data[index:index+window]))
          
            index = index + stride
  
    X = np.array(X)
   
    return X