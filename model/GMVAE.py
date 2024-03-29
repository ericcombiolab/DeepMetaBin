"""
---------------------------------------------------------------------
-- Author: RAO Mingxing
---------------------------------------------------------------------

DeepmetaBin for Semi-supervised Clustering

"""
from cmath import isnan
from unicodedata import category
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from networks.Networks import *
from losses.LossFunctions import *
from metrics.Metrics import *
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import logging
from tqdm import tqdm
import _pickle as pickle


class GMVAE:

  def __init__(self, args):
    self.num_epochs = args.epochs
    self.cuda = args.cuda
    self.verbose = args.verbose

    self.batch_size = args.batch_size
    self.batch_size_val = args.batch_size_val
    self.learning_rate = args.learning_rate
    self.decay_epoch = args.decay_epoch
    self.lr_decay = args.lr_decay
    self.w_cat = args.w_categ
    self.w_gauss = args.w_gauss    
    self.w_rec = args.w_rec
    self.rec_type = args.rec_type 

    self.num_classes = args.num_classes
    self.gaussian_size = args.gaussian_size
    self.input_size = args.input_size

    # gumbel
    self.init_temp = args.init_temp
    self.decay_temp = args.decay_temp
    self.hard_gumbel = args.hard_gumbel
    self.min_temp = args.min_temp
    self.decay_temp_rate = args.decay_temp_rate
    self.gumbel_temp = self.init_temp

    self.network = GMVAENet(self.input_size, self.gaussian_size, self.num_classes)
    self.losses = LossFunctions()
    self.metrics = Metrics()

    self.logger = self.logger_config(log_path='log.txt', logging_name='paperCrawler')
    self.cutoff = args.cutoff
    self.ground_truth = []

    self.best_metric_results = (0.0, 0.0, 0.0, 0.0)
    self.best_result = None
    # self.best_model = None

    with open(args.truth, 'r') as f:
      for l in f.readlines():
          items = l.split(',')
          if len(items) == 3:
              continue
          temp = items[0].split('_')
          if int(temp[3]) >= self.cutoff:
            self.ground_truth.append((items[0], items[1]))

    if self.cuda:
      self.network = self.network.cuda() 
  

  def unlabeled_loss(self, data1, data2, out_net1, out_net2, weight):
    """Method defining the loss functions derived from the variational lower bound
    Args:
        data: (array) corresponding array containing the input data
        out_net: (dict) contains the graph operations or nodes of the network output

    Returns:
        loss_dic: (dict) contains the values of each loss function and predictions
    """
    # obtain network variables
    z1, data_recon1 = out_net1['gaussian'], out_net1['x_rec'] 
    logits1, prob_cat1 = out_net1['logits'], out_net1['prob_cat']
    y_mu1, y_var1 = out_net1['y_mean'], out_net1['y_var']
    mu1, var1 = out_net1['mean'], out_net1['var']
    # reconstruction loss
    loss_rec1 = self.losses.reconstruction_loss(data1, data_recon1, self.rec_type)

    # gaussian loss
    loss_gauss1 = self.losses.gaussian_loss(z1, mu1, var1, y_mu1, y_var1)

    # categorical loss
    loss_cat1 = -self.losses.entropy(logits1, prob_cat1) - np.log(0.1)

    # total loss
    loss_total1 = self.w_rec * loss_rec1 + self.w_gauss * loss_gauss1 + self.w_cat * loss_cat1
    # print(loss_cat1)
    # sys.exit(0)
    
    z2, data_recon2 = out_net2['gaussian'], out_net2['x_rec'] 
    logits2, prob_cat2 = out_net2['logits'], out_net2['prob_cat']
    y_mu2, y_var2 = out_net2['y_mean'], out_net2['y_var']
    mu2, var2 = out_net2['mean'], out_net2['var']
    
    # reconstruction loss
    loss_rec2 = self.losses.reconstruction_loss(data1, data_recon2, self.rec_type)

    # gaussian loss
    loss_gauss2 = self.losses.gaussian_loss(z2, mu2, var2, y_mu2, y_var2)

    # categorical loss
    loss_cat2 = -self.losses.entropy(logits2, prob_cat2) - np.log(0.1)

    # total loss
    loss_total2 = self.w_rec * loss_rec2 + self.w_gauss * loss_gauss2 + self.w_cat * loss_cat2
    
    loss = loss_total1 + loss_total2
    # print(loss)
    total = 0.5 * torch.mean(loss * weight)

    loss_dic = {'total': total, 
                'reconstruction': torch.mean(loss_rec1),
                'gaussian': torch.mean(loss_gauss1),
                'categorical': torch.mean(loss_cat1)}
    return loss_dic

  def test_loss(self, data, out_net):
    """Method defining the loss functions derived from the variational lower bound
    Args:
        data: (array) corresponding array containing the input data
        out_net: (dict) contains the graph operations or nodes of the network output

    Returns:
        loss_dic: (dict) contains the values of each loss function and predictions
    """
    # obtain network variables
    z, data_recon = out_net['gaussian'], out_net['x_rec'] 
    logits, prob_cat = out_net['logits'], out_net['prob_cat']
    y_mu, y_var = out_net['y_mean'], out_net['y_var']
    mu, var = out_net['mean'], out_net['var']
    
    # reconstruction loss
    loss_rec = torch.mean(self.losses.reconstruction_loss(data, data_recon, self.rec_type))

    # gaussian loss
    loss_gauss = torch.mean(self.losses.gaussian_loss(z, mu, var, y_mu, y_var))

    # categorical loss
    loss_cat = torch.mean(-self.losses.entropy(logits, prob_cat) - np.log(0.1))

    # total loss
    loss_total = self.w_rec * loss_rec + self.w_gauss * loss_gauss + self.w_cat * loss_cat
    
    predicted_clusters = prob_cat.argmax(-1)
    highest_probs = prob_cat.max(-1).values

    loss_dic = {'total': loss_total, 
                'predicted_clusters': predicted_clusters,
                'reconstruction': loss_rec,
                'gaussian': loss_gauss,
                'categorical': loss_cat,
                'highest_prob': highest_probs}
    return loss_dic
    
  
  def train_epoch(self, optimizer, data_loader):
    """Train the model for one epoch

    Args:
        optimizer: (Optim) optimizer to use in backpropagation
        data_loader: (DataLoader) corresponding loader containing the training data

    Returns:
        average of all loss values, accuracy, nmi
    """
    self.network.train()
    total_loss = 0.
    recon_loss = 0.
    cat_loss = 0.
    gauss_loss = 0.

    accuracy = 0.
    nmi = 0.
    num_batches = 0.
    
    true_labels_list = []
    predicted_labels_list = []

    # iterate over the dataset
    for (data1, data2, weight, _) in data_loader:
      if self.cuda == 1:
        data1 = data1.cuda()
        data2 = data2.cuda()
        weight = weight.cuda()

      optimizer.zero_grad()
      # flatten data
      # data = data.view(data.size(0), -1)

      # separate xi and xj

      # forward call
      out_net1 = self.network(data1, self.gumbel_temp, self.hard_gumbel) 
      out_net2 = self.network(data2, self.gumbel_temp, self.hard_gumbel) 
      unlab_loss_dic = self.unlabeled_loss(data1, data2, out_net1, out_net2, weight) 
      total = unlab_loss_dic['total']

      # accumulate values
      total_loss += total.item()
      recon_loss += unlab_loss_dic['reconstruction'].item()
      gauss_loss += unlab_loss_dic['gaussian'].item()
      cat_loss += unlab_loss_dic['categorical'].item()
      total.backward()
      optimizer.step()  
   
      num_batches += 1. 

    # average per batch
    total_loss /= num_batches
    recon_loss /= num_batches
    gauss_loss /= num_batches
    cat_loss /= num_batches
    

    return total_loss, recon_loss, gauss_loss, cat_loss

  def test(self, data_loader, return_loss=False, output_result = False):
    """Test the model with new data

    Args:
        data_loader: (DataLoader) corresponding loader containing the test/validation data
        return_loss: (boolean) whether to return the average loss values
          
    Return:
        accuracy and nmi for the given test data

    """
    self.network.eval()
    total_loss = 0.
    recon_loss = 0.
    cat_loss = 0.
    gauss_loss = 0.

    accuracy = 0.
    nmi = 0.
    num_batches = 0.
    
    # true_labels_list = []
    # predicted_labels_list = []
    clusters = []

    with torch.no_grad():
      for data, contigs in data_loader:
        if self.cuda == 1:
          data = data.cuda()
      
        # flatten data
        data = data.view(data.size(0), -1)

        # forward call
        out_net = self.network(data, self.gumbel_temp, self.hard_gumbel) 
        unlab_loss_dic = self.test_loss(data, out_net)  

        # accumulate values
        total_loss += unlab_loss_dic['total'].item()
        recon_loss += unlab_loss_dic['reconstruction'].item()
        gauss_loss += unlab_loss_dic['gaussian'].item()
        cat_loss += unlab_loss_dic['categorical'].item()

        # save predicted and true labels
        predicted_clusters = unlab_loss_dic['predicted_clusters']
        highest_probs = unlab_loss_dic['highest_prob']

        for cluster, highest_prob, contig in zip(predicted_clusters, highest_probs, contigs):
          clusters.append((cluster.item(), highest_prob.item(), contig))
        # true_labels_list.append(labels)
        # predicted_labels_list.append(predicted)   
   
        num_batches += 1. 

    # average per batch
    if return_loss:
      total_loss /= num_batches
      recon_loss /= num_batches
      gauss_loss /= num_batches
      cat_loss /= num_batches

    if output_result:
      temp = []
      for cluster, _, contig in clusters:
        temp.append((cluster, contig))
      clusters = temp
      return self.calculate_accuracy(clusters, self.ground_truth)
    
    if return_loss:
      temp = []
      for cluster, _, contig in clusters:
        temp.append((cluster, contig))
      clusters = temp
      return total_loss, recon_loss, gauss_loss, cat_loss, clusters
    else:
      return clusters


  def train(self, train_loader, val_loader, test_loader):
    """Train the model

    Args:
        train_loader: (DataLoader) corresponding loader containing the training data
        val_loader: (DataLoader) corresponding loader containing the validation data

    Returns:
        output: (dict) contains the history of train/val loss
    """
    optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)


    # val_results = []
    for epoch in range(1, self.num_epochs + 1):
      train_loss, train_rec, train_gauss, train_cat = self.train_epoch(optimizer, train_loader)
      val_loss, val_rec, val_gauss, val_cat, predicts = self.test(test_loader, True)
      precision, recall, f1_score, ari = self.calculate_accuracy(predicts, self.ground_truth)
      contignames, predicts, metric_results = self.fit_gmm(test_loader)
      precision, recall, f1_score, ari = metric_results
      # if epoch > 50:
        # precision, recall, f1_score, ari = self.fit_gmm(test_loader)

      # precision1, recall1, f1_score1, ari1 = self.fit_gmm(test_loader)     
      # val_results.append((epoch, f1_score, ari))
      if f1_score > self.best_metric_results[2]:
        self.best_metric_results = (precision, recall, f1_score, ari)
        self.best_result = (contignames, predicts)

      self.logger.info("(Epoch %d / %d)" % (epoch, self.num_epochs))

      self.logger.info("Train - Loss: %.5lf; REC: %.5lf;  Gauss: %.5lf;  Cat: %.5lf;" % \
            (train_loss, train_rec, train_gauss, train_cat))

      self.logger.info("Valid - Loss: %.5lf; REC: %.5lf;  Gauss: %.5lf;  Cat: %.5lf;" % \
            (val_loss, val_rec, val_gauss, val_cat))

      self.logger.info("Valid - Precision: {:.5f}; Recall: {:.5f}; F1_score: {:.5f}; ARI: {:.5f}".format(precision, recall, f1_score, ari))


      # decay gumbel temperature
      if self.decay_temp == 1:
        self.gumbel_temp = np.maximum(self.init_temp * np.exp(-self.decay_temp_rate * epoch), self.min_temp)
        if self.verbose == 1:
          print("Gumbel Temperature: %.3lf" % self.gumbel_temp)
    return self.best_result, self.best_metric_results


  def calculate_precision(self, predicts, ground_truth):
    predicts_dict = {}
    for clusterNo, contig in predicts:
        if clusterNo in predicts_dict.keys():
            predicts_dict[clusterNo].append(contig)
        else:
            predicts_dict[clusterNo] = [contig]

    ground_truth_dict = {}
    for contig, label in ground_truth:
        ground_truth_dict[contig] = label
    precision_dict = {}
    for key, value in predicts_dict.items():
        precision_dict[key] = {}
        for contig in value:
            if contig not in ground_truth_dict.keys():
                continue
            if ground_truth_dict[contig] in precision_dict[key].keys():
                precision_dict[key][ground_truth_dict[contig]] += 1
            else:
                precision_dict[key][ground_truth_dict[contig]] = 1
    correct_predicts = 0
    total_predicts = 0
    for label_dict in precision_dict.values():
        if len(label_dict.values()) != 0:
            correct_predicts += max(label_dict.values())
            total_predicts += sum(label_dict.values())
    return correct_predicts / total_predicts

      

  def calculate_recall(self, predicts, ground_truth):
    predicts_dict = {}
    for clusterNo, contig in predicts:
        predicts_dict[contig] = clusterNo
    ground_truth_dict = {}
    for contig, label in ground_truth:
        if label in ground_truth_dict.keys():
            ground_truth_dict[label].append(contig)
        else:
            ground_truth_dict[label] = [contig]

    recall_dict = {}
    total_recalls = 0
    for key, value in ground_truth_dict.items():
        recall_dict[key] = {}
        for contig in value:
            if contig not in predicts_dict.keys():
                total_recalls += 1
                continue
            if predicts_dict[contig] in recall_dict[key].keys():
                recall_dict[key][predicts_dict[contig]] += 1
            else:
                recall_dict[key][predicts_dict[contig]] = 1
    correct_recalls = 0
    for cluster_dict in recall_dict.values():
        if len(cluster_dict.values()) != 0:
            correct_recalls += max(cluster_dict.values())
            total_recalls += sum(cluster_dict.values())
    return correct_recalls / total_recalls


  def calculate_ari(self, predicts, ground_truth):
    ground_truth_dict = {}
    for contig, label in ground_truth:
        ground_truth_dict[contig] = label
    clusters = []
    labels_true = []
    for clusterNo, contig in predicts:
        if contig not in ground_truth_dict.keys():
            continue
        clusters.append(clusterNo)
        labels_true.append(ground_truth_dict[contig])
    return adjusted_rand_score(clusters, labels_true)

  def calculate_accuracy(self, predicts, ground_truth):
    precision = self.calculate_precision(predicts, ground_truth)
    recall = self.calculate_recall(predicts, ground_truth)
    f1_score = 2 * (precision * recall) / (precision + recall)
    ari = self.calculate_ari(predicts, ground_truth)
    return precision, recall, f1_score, ari
  

  def latent_features_with_true_labels(self, data_loader, return_labels=False):
    """Obtain latent features learnt by the model

    Args:
        data_loader: (DataLoader) loader containing the data
        return_labels: (boolean) whether to return true labels or not

    Returns:
       features: (array) array containing the features from the data
    """
    self.network.eval()
    N = len(data_loader.dataset)
    features = np.zeros((N, self.gaussian_size))
    if return_labels:
      label_dic = {}
      label_types = set()
      types_dict = {}
      for contigname, category in self.ground_truth:
        label_types.add(category)
        label_dic[contigname] = category
      for idx, val in enumerate(label_types):
        types_dict[val] = idx
      for key, value in label_dic.items():
        label_dic[key] = types_dict[value]
      true_labels = np.zeros(N, dtype=np.int64)
    start_ind = 0
    with torch.no_grad():
      for (data, labels) in data_loader:
        if self.cuda == 1:
          data = data.cuda()
        # flatten data
        data = data.view(data.size(0), -1)  
        out = self.network.inference(data, self.gumbel_temp, self.hard_gumbel)
        latent_feat = out['mean']
        end_ind = min(start_ind + data.size(0), N+1)

        # return true labels
        if return_labels:
          labels_numpy = []
          del_idx = []
          for idx, contig in enumerate(labels):
            try:
              labels_numpy.append(label_dic[contig])
            except KeyError:
              end_ind -= 1
              del_idx.append(idx)
          temp = torch.zeros(latent_feat.size(0)) 
          temp[del_idx] = 1   
          latent_feat = latent_feat[torch.zeros(latent_feat.size(0)) == temp]
          labels_numpy = np.array(labels_numpy)
          true_labels[start_ind:end_ind] = labels_numpy

        features[start_ind:end_ind] = latent_feat.cpu().detach().numpy() 
        start_ind = end_ind

    true_labels = true_labels[:end_ind]
    features = features[:end_ind] 
    if return_labels:
      return features, true_labels
    return features



  def plot_latent_space(self, data_loader, save=False):
    """Plot the latent space learnt by the model

    Args:
        data: (array) corresponding array containing the data
        labels: (array) corresponding array containing the labels
        save: (bool) whether to save the latent space plot

    Returns:
        fig: (figure) plot of the latent space
    """
    # obtain the latent features
    features, labels = self.latent_features_with_true_labels(data_loader, True)
    # reduce the dimention
    features = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(features)
    # plot only the first 2 dimensions
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0], features[:, 1], c=labels, marker='.',
            edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)

    plt.colorbar()
    if(save):
        fig.savefig('latent_space.png')
    return fig
  


  def generate_latent(self, data_loader):
    """Test the model with new data

    Args:
        data_loader: (DataLoader) corresponding loader containing the test/validation data
        return_loss: (boolean) whether to return the average loss values
          
    Return:
        accuracy and nmi for the given test data

    """
    self.network.eval()
    num_batches = 0.
    
    contignames = []
    latents = []
    with torch.no_grad():
      for data, contigs in data_loader:
        if self.cuda == 1:
          data = data.cuda()
      
        # flatten data
        data = data.view(data.size(0), -1)

        # forward call
        out_net = self.network(data, self.gumbel_temp, self.hard_gumbel)
        mus= out_net['mean']


        for mu, contig in zip(mus, contigs):
          latents.append(mu.cpu().detach().numpy() )
          contignames.append(contig)
   
        num_batches += 1. 
    latents = np.float32(np.stack(latents, axis=0))
    return latents, contignames

    # average per batch

  def fit_gmm(self, data_loader):
    latents, contignames = self.generate_latent(data_loader)
  
    gmm = GaussianMixture(n_components=self.num_classes, covariance_type='full', random_state = 0)
    predicts = gmm.fit_predict(latents)
    results = []
    for cluster, contig in zip(predicts, contignames):
      results.append((cluster, contig))
    return contignames, predicts, self.calculate_accuracy(results, self.ground_truth)
    

  def k_means(self, data_loader):
    latents, contignames = self.generate_latent(data_loader)
    kmeans_model = KMeans(n_clusters=self.num_classes, random_state=0).fit(X=latents.astype('float32'))
    predicts = kmeans_model.predict(latents)
    results = []
    for cluster, contig in zip(predicts, contignames):
      results.append((cluster, contig))
    return self.calculate_accuracy(results, self.ground_truth)

  def logger_config(self, log_path, logging_name):
    '''
    config log
    :param log_path: output log path
    :param logging_name: record name，optional
    :return:
    '''

    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)

    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger





