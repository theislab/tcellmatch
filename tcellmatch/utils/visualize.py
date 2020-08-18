from tcellmatch.estimators.estimator_base import *
import os
import numpy as np
from sklearn import manifold
import seaborn as sns
from matplotlib import pyplot as plt
import umap.umap_ as umap


class Visualization:

    def __init__(
            self,
            indir
    ):
        # Path of input directory.
        self.indir = indir
        self.blosumfile = self.indir + "/blosum/BLOSUM50.txt"
        # Path to 10x raw files.
        self.fns = [
            "vdj_v1_hs_aggregated_donor1_binarized_matrix.csv",
            "vdj_v1_hs_aggregated_donor2_binarized_matrix.csv"
        ]

        # Path to preprocessed covariates files.
        # covariates: pd.DataFrame with covariates including TCR CDR3 sequence by cell and chain [observations, covariates]
        self.fns_covar = [
            "vdj_v1_hs_aggregated_donor1_binarized_matrix_extended_covariates.csv",
            "vdj_v1_hs_aggregated_donor2_binarized_matrix_extended_covariates.csv"
        ]
        # Path to preprocessed clonotypes files.
        self.fns_clonotype = [
            "vdj_v1_hs_aggregated_donor1_clonotypes.csv",
            "vdj_v1_hs_aggregated_donor2_clonotypes.csv"
        ]
        # Path to iedb files.
        self.fn_iedb = self.indir + "tcell_receptor_table_export_1558607498.csv"

        #path for saving files.
        self.tsne_path = self.indir+"figures/tsne/"
        self.umap_path = self.indir+"figures/umap/"
        if not os.path.exists(self.tsne_path) and not os.path.exists(self.tsne_path):
            os.makedirs(self.tsne_path)
            os.makedirs(self.umap_path)

        self.nc_cols = [
            'A0101_SLEGGGLGY_NC_binder',
            'A0101_STEGGGLAY_NC_binder',
            'A0201_ALIAPVHAV_NC_binder',
            'A2402_AYSSAGASI_NC_binder',
            'B0702_GPAESAAGL_NC_binder',
            'NR(B0801)_AAKGRGAAL_NC_binder'
        ]

    def read_and_plot_10x(self,blosum_encoding = False,):
        target_ids = [
            'A0201_GILGFVFTL_Flu-MP_Influenza_binder',
            'A0201_GLCTLVAML_BMLF1_EBV_binder',
            'A0301_KLGGALQAK_IE-1_CMV_binder',
            'A1101_IVTDFSVIK_EBNA-3B_EBV_binder',
            'A1101_AVFDRKSDAK_EBNA-3B_EBV_binder',
            'B0801_RAKFKQLL_BZLF1_EBV_binder'
        ]

        base = EstimatorBase()
        base.read_binarized_matrix(
            fns=[self.indir + x for x in self.fns],
            fns_clonotype=[self.indir + x for x in self.fns_clonotype],
            fns_covar=[self.indir + x for x in self.fns_covar],
            fn_blosum=self.blosumfile,
            blosum_encoding=blosum_encoding,
            is_train=False,
            covariate_formula_categ=["donor"],
            covariate_formula_numeric=[],
            add_non_binder_for_softmax=True,
            chains="trb",
            label_cols=target_ids,
            nc_cols=self.nc_cols
        )
        #Choose samples w.r.t the target_ids.
        length_target = len(target_ids)
        labels = np.where(base.y_test == 1)[1]
        labels_idx = np.where(labels != length_target)[0]
        base.x_test = base.x_test[labels_idx]
        labels = labels[labels_idx]
        #Reshape to feed into tsne.
        base.x_test = np.reshape(base.x_test, (int(base.x_test.shape[0]), -1))
        file_name = "10x"
        for x in target_ids:
            file_name = file_name + "_" + x.split()[0]
        sns_plot_tsne = self.plot_tsne_and_umap(base.x_test, labels, model_name="tsne")
        sns_plot_tsne.savefig(self.tsne_path + file_name)
        sns_plot_umap = self.plot_tsne_and_umap(base.x_test, labels, model_name="umap")
        sns_plot_umap.savefig(self.umap_path + file_name)

    def read_and_plot_iedb(self,blosum_encoding = False,):
        target_ids = [
                "GILGFVFTL",
                "NLVPMVATV",
                "GLCTLVAML",
                "LLWNGPMAV",
                "VLFGLGFAI"
            ]

        base = EstimatorBase()
        base.read_iedb_as_categorical_model(
            fns=self.fn_iedb,
            antigen_ids=target_ids,
            fn_blosum=self.blosumfile,
            blosum_encoding=blosum_encoding,
            add_non_binder_for_softmax=True,
            is_train=False
        )
        #Choose samples w.r.t the target_ids.
        length_target = len(target_ids)
        labels = np.where(base.y_test == 1)[1]
        labels_idx = np.where(labels != length_target)[0]
        base.x_test = base.x_test[labels_idx]
        labels = labels[labels_idx]
        #Reshape to feed into tsne.
        base.x_test = np.reshape(base.x_test, (int(base.x_test.shape[0]), -1))

        file_name = "iedb"
        for x in target_ids:
            file_name = file_name + "_" + x.split()[0]
        _ = self.plot_tsne_and_umap(training_set=base.x_test,
                                    labels=labels,
                                    save_path=self.tsne_path + file_name,
                                    model_name="tsne")

        _ = self.plot_tsne_and_umap(
            training_set=base.x_test,
            labels=labels,
            save_path=self.umap_path + file_name,
            model_name="umap"
        )

    def plot_tsne_and_umap(self,training_set,labels,save_path,model_name = "tsne"):
        """
        Run T-SNE on training set and save plots.
        :param training_set:
        :param labels: labels of training set
        :param model_name: "tsne" or "umap"
        :return:
        """
        if model_name == "tsne":
            tsne_train = manifold.TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0,
                                 n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean',
                                 init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5)
            results = tsne_train.fit_transform(training_set)
        elif model_name == "umap":
            umap_train = umap.UMAP(n_neighbors=10,
                                  min_dist=0.3,
                                  metric='correlation')
            results = umap_train.fit_transform(training_set)
        else:
            raise ValueError("Model name could not be recognized")
        df_subset_up = {}
        df_subset_up['tsne-2d-one'] = results[:, 0]
        df_subset_up['tsne-2d-two'] = results[:, 1]
        df_subset_up['y'] = labels
        plt.figure(figsize=(16, 10))
        sns_plot = sns.scatterplot(
            x="tsne-2d-one",
            y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", max(labels)+1),
            data=df_subset_up,
            legend="full",
            alpha=0.3
        )
        fig = sns_plot.get_figure()
        fig.savefig(save_path)
        return sns_plot
