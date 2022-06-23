# vectors = []
# cons = []
# true_labels = []
# del_idx = []
# for idx, temp in enumerate(train_set):
#     try:
#         data, con = temp
#         true_labels.append(label_dic[con])
#     except KeyError:
#         del_idx.append(idx)
#     vectors.append(data.cpu().detach().numpy())
# true_labels = np.array(true_labels)
# vectors = np.stack(vectors, axis = 0)
# temp = np.zeros(vectors.shape[0]) 
# temp[del_idx] = 1   
# vectors = vectors[np.zeros(vectors.shape[0]) == temp]
# X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(vectors)
# plot_latent_space(X_embedded, true_labels,True)