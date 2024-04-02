import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict 
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from tqdm import tqdm
def read_file(file_name):
    data=[]
    with open(file_name,"r") as file:
        lines=file.readlines()
        for line in lines:
            line=line.split()
            for i in range(len(line)):
                line[i]=float(line[i])
            data.append(line)
    return data


# def K_medoids(data,K,max_iterations):
#     data=np.array(data)
#     # cnt=1

#     for iter in range(max_iterations):
#         clusters=[[] for _ in range(K)]
#         centers=data[np.random.choice(range(len(data)),K,replace=False)]

#         for point in data:
#             distances=[np.linalg.norm(point-center) for center in centers]
#             cluster_index = np.argmin(distances)
#             clusters[cluster_index].append(point)

#         new_centers=[]
#         for cluster in clusters:
#             cluster=np.array(cluster)
#             distances_sum=[np.sum(np.linalg.norm(cluster-cluster[i],axis=1)) for i in range(len(cluster))]
#             new_cneter_index=np.argmin(distances_sum)
#             new_centers.append(cluster[new_cneter_index])
#         if np.all(centers==new_centers):
#             break
#         centers=new_centers
#         # cnt+=1
#     # print(f"iter = {iter}")
#     return centers,clusters
def calculate_Euclidean_distance_matrix(data):
    EDM=[]
    for point1 in data:
        tmp=[]
        for point2 in data:
            tmp.append(np.linalg.norm(point1-point2))
        EDM.append(tmp)
    return EDM

    

def DB_scan(data,eps,minPts):
    data=np.array(data)
    distance_matrix=calculate_Euclidean_distance_matrix(data)
    # print(np.array(distance_matrix))
    n,m=data.shape
    # print(n,m)
    core_pt_idxs=[]
    for idx_start in range(len(distance_matrix)):
        cnt=0
        for dis in distance_matrix[idx_start]:
            if dis<=eps:
                cnt+=1
        if cnt>=minPts:
            core_pt_idxs.append(idx_start)
    # print(np.array(distance_matrix))
    # print(len(core_pt_idx),len(data))
    labels=[-1]*n
    cluster_id=0
    for core_pt_idx in core_pt_idxs:
        if labels[core_pt_idx]==-1:
            labels[core_pt_idx]=cluster_id
            neighbor=[]
            for idx,dis in enumerate(distance_matrix[core_pt_idx]):
                if dis<=eps and labels[idx]==-1:
                    neighbor.append(idx)
            neighbor=set(neighbor)
            while len(neighbor)>0:
                new_pt=neighbor.pop()
                labels[new_pt]=cluster_id
                neighbor_neighbor=[]
                for idx,dis in enumerate(distance_matrix[new_pt]):
                    if dis<=eps:
                        neighbor_neighbor.append(idx)
                if len(neighbor_neighbor)>=minPts:
                    for nbnb in neighbor_neighbor:
                        if labels[nbnb]==-1:
                            neighbor.add(nbnb)
            cluster_id+=1
    return labels
        
    
def get_labels(label):
    if label == -1:
        return "noise"
    else:
        return label



def plot_before_and_after_clustering(data, before_labels, after_labels):
    # 創建一個新的圖形
    plt.figure(figsize=(12, 6))

    # 在第一個子圖上繪製分群前的資料分布
    plt.subplot(1, 2, 1)
    plt.scatter([point[0] for point in data], [point[1] for point in data], color='gray', label='Before Clustering')
    plt.title('Before Clustering')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')

    # 在第二個子圖上繪製分群後的資料分布
    plt.subplot(1, 2, 2)
    unique_labels = list(set(after_labels))
    colors = [mcolors.to_rgba(f'C{i}') for i in range(len(unique_labels))]
    print(f"unique label = {unique_labels}")
    print(f"colors = {colors}")
    color_dict = {label: colors[i] for i, label in enumerate(unique_labels)}
    label_legend = defaultdict(bool)
    for i in range(len(data)):
        x, y = data[i]
        group = after_labels[i]
        plt.scatter(x, y, color=color_dict[group], label=f'Group {group}')
    plt.title('After Clustering')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    legend_handles = [Patch(color=color, label=get_labels(label)) for color, label in zip(colors, unique_labels)]

    plt.legend(handles=legend_handles, loc='upper left')


    plt.show()

if __name__ == "__main__":
    directory="Clustering_testdata"
    file_names=os.listdir(directory)
    print(file_names)
    arguments=[[8,39],[8,50],[10,100],[10,100],[10,100]]
    for i in range(len(file_names)):
        file_name=file_names[i]
        full_path=os.path.join(directory,file_name)
        data=read_file(full_path)
        eps=arguments[i][0]
        minPts=arguments[i][1]
        #start_time=time.time()
        arg=[]
        if i==1:
            for e in tqdm(range(5,31),desc='outer loop'):
                for m in tqdm(range(30,81),desc='inner loop'):
                    labels=DB_scan(data,e,m)
                    if len(set(labels))==5:
                        arg.append([e,m])
                        print(f"arg = {arg}")
                        print(f"len of data = {len(data)}")
                        print(f"count_of_minus_ones = {labels.count(-1)}")
            print(arg)
        else:
            labels=DB_scan(data,eps,minPts)
        # end_time=time.time()
        # for j in range(len(data)):
        #     print(f"Data = {data[j]}")
        #     print(f"Label = {labels[j]}")
        # print(f"len of data = {len(data)}")
        # print(labels)
        # print(f"num of labels = {len(set(labels))}")
        # print(f"len of labels = {len(labels)}")
        # print(f"len of data = {len(data)}")
        # print(f"Perform DB-scan clustering algorithm in {file_name} dataset spend {round(end_time-start_time,6)} sec.")    
        # print()
        # plot_before_and_after_clustering(data,[-1]*len(data),labels)
        # input("press Enter to continue...")
        # os.system('cls' if os.name == 'nt' else 'clear')
