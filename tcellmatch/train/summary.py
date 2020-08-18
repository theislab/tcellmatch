import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import pickle
import scipy.sparse
import seaborn as sns

from tcellmatch.estimators import deviation_label, deviation_global


class SummaryContainer:

    def __init__(
            self,
            indir,
            gs_ids
    ):
        self.indirs = [
            indir + x
            for x in gs_ids
        ]

    def load_all(
            self,
            data="",
            load_peptide_seqs_by_dataset: bool = False
    ):
        run_ids = [
            np.sort(np.unique([
                x.split("_runparams.pickle")[0]
                for x in os.listdir(indir + '/summaries/')
                if "_runparams.pickle" in x
            ]))
            for i, indir in enumerate(self.indirs)
        ]
        model_args = {}
        model_settings = {}
        runparams = {}
        labels = {}
        histories = {}
        evaluations = {}
        evaluations_custom = {}
        run_ids_proc = []
        dirs = {}
        peptide_seqs_unique = {}
        labels = {}
        for i, indir in enumerate(self.indirs):
            for x in run_ids[i]:
                if data != "":
                    x_string = x + "_" + data
                else:
                    x_string = x
                fn = indir + '/summaries/' + x + "_labels.csv"
                if os.path.isfile(fn):
                    labels[x] = pandas.read_csv(fn)["label"].values
                fn = indir + '/models/' + x + "_model_args.pkl"
                if os.path.isfile(fn):
                    with open(fn, 'rb') as f:
                        model_args[x] = pickle.load(f)
                fn = indir + '/models/' + x + "_model_settings.pkl"
                if os.path.isfile(fn):
                    with open(fn, 'rb') as f:
                        model_settings[x] = pickle.load(f)
                fn = indir + '/summaries/' + x + "_runparams.pickle"
                if os.path.isfile(fn):
                    with open(fn, 'rb') as f:
                        runparams[x] = pickle.load(f)
                fn = indir + '/summaries/' + x_string + "_history.pkl"
                if os.path.isfile(fn):
                    with open(fn, 'rb') as f:
                        histories[x] = pickle.load(f)
                    run_ids_proc.append(x)
                fn = indir + '/summaries/' + x_string + "_evaluations.pkl"
                if os.path.isfile(fn):
                    with open(fn, 'rb') as f:
                        evaluations[x] = pickle.load(f)
                fn = indir + '/summaries/' + x_string + "_evaluations_custom.pkl"
                if os.path.isfile(fn):
                    with open(fn, 'rb') as f:
                        evaluations_custom[x] = pickle.load(f)
                dirs[x] = indir
                if load_peptide_seqs_by_dataset:
                    fn = indir + '/summaries/' + x_string + "_peptide_seqs_unique.pkl"
                    if os.path.isfile(fn):
                        with open(fn, 'rb') as f:
                            peptide_seqs_unique[x] = pickle.load(f)
                    else:
                        peptide_seqs_unique[x] = None
                else:
                    fn = indir + '/models/' + x + "_peptide_seqs_unique.csv"
                    if os.path.isfile(fn):
                        peptide_seqs_unique[x] = pandas.read_csv(fn)["antigen"].values
                    else:
                        peptide_seqs_unique[x] = None
        self.model_args = model_args
        self.model_settings = model_settings
        self.runparams = runparams
        self.labels = labels
        self.histories = histories
        self.evaluations = evaluations
        self.evaluations_custom = evaluations_custom
        self.run_ids_proc = run_ids_proc
        self.dirs = dirs
        self.y = {}
        self.yhat = {}
        self.peptide_seqs = {}
        self.peptide_seqs_unique = peptide_seqs_unique
        self.labels = labels

    def choose_best(
            self,
            partition: str = "val",
            metric: str = "loss",
            groups_keep=[],
            subset=[]
    ):
        results_tab = self.load_table(
            subset=subset
        )
        results_tab = self.reduce_table(
            tab=results_tab,
            metric=metric,
            partition=partition,
            groups_keep=groups_keep
        )
        return {
            "run_group": dict([
                (x[0], "_".join(x[1:]))
                for x in results_tab[["run_group"] + groups_keep].drop_duplicates().values
            ]),
            "run": dict([
                (x[0], "_".join(x[1:]))
                for x in results_tab[["run"] + groups_keep].values
            ])
        }

    def plot_training_overview(
            self,
            ids,
            remove_legend=False,
            xrot=45,
            log_y=True
    ):
        """ Average training loss across cross-validations. """
        sns_data = pandas.concat([pandas.DataFrame({
            "epochs": np.arange(0, len(self.histories[k]['loss'])),
            "loss_train": self.histories[k]['loss'],
            "loss_val": self.histories[k]['val_loss'],
            "lr": self.histories[k]['lr'],
            "run": k,
            "model": v,
            "cv": k.split("_")[-1],
        }) for k, v in ids.items()])

        fig, axs = plt.subplots(1, 3, figsize=(3 * 5, 5))
        sns.lineplot(
            x="epochs",
            y="loss_train",
            hue="model",
            style="cv",
            data=sns_data,
            ax=axs[0]
        )
        axs[0].legend_.remove()
        if log_y:
            axs[0].set_yscale('log')
        axs[0].tick_params(axis='x', labelrotation=xrot)
        axs[0].set_xlabel("epochs")
        axs[0].set_ylabel("training loss")

        sns.lineplot(
            x="epochs",
            y="loss_val",
            hue="model",
            style="cv",
            data=sns_data,
            ax=axs[1]
        )
        axs[1].legend_.remove()
        if log_y:
            axs[1].set_yscale('log')
        axs[1].tick_params(axis='x', labelrotation=xrot)
        axs[1].set_xlabel("epochs")
        axs[1].set_ylabel("validation loss")

        sns.lineplot(
            x="epochs",
            y="lr",
            hue="model",
            style="cv",
            data=sns_data,
            ax=axs[2]
        )
        if remove_legend:
            axs[2].legend_.remove()
        else:
            box = axs[2].get_position()
            axs[2].set_position([box.x0, box.y0, box.width * 0.95, box.height])
            axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if log_y:
            axs[2].set_yscale('log')
        axs[2].tick_params(axis='x', labelrotation=xrot)
        axs[2].set_xlabel("epochs")
        axs[2].set_ylabel("learning rate")
        return fig, axs

    def load_table(
            self,
            subset=[]
    ):
        """
        :param metrics_custom: global or local
        """
        res = [  # ids
            pandas.concat([  # objects
                pandas.DataFrame({
                    "run": id_x,
                    "run_group": id_x.split("_cv")[0]
                }, index=[id_x]),
                pandas.DataFrame(dict([
                    (k, [str(v)])
                    for k, v in self.model_args[id_x].items()
                    if k in ["loss", "val_loss"]
                ]), index=[id_x]),
                pandas.DataFrame(dict([
                    (k, [str(v)])
                    for k, v in self.model_settings[id_x].items()
                ]), index=[id_x]),
                pandas.DataFrame(dict([
                    (k, [str(v)])
                    for k, v in self.runparams[id_x].items()
                ]), index=[id_x]),
                pandas.concat([  # partitions
                    pandas.concat([
                        pandas.DataFrame(dict([
                            # need split here because tensorflow adds _i for number of execution of metric across partitions
                            (k.split("_1")[0].split("_2")[0].split("_3")[0].split("_4")[0] + "_" + p, [v])
                            for k, v in self.evaluations[id_x][p].items()
                        ]), index=[id_x]),
                        pandas.DataFrame(dict([
                            (k + "_" + p, v[0].flatten())
                            for k, v in self.evaluations_custom[id_x][p]["global"].items()
                        ]), index=[id_x])
                    ], axis=1)
                    for p in ["train", "val", "test"]
                ], axis=1)
            ], axis=1)
            for id_x in self.run_ids_proc
        ]
        # Pad non-overlapping columns names
        all_cols = np.unique([item for x in res for item in x.columns.tolist()]).tolist()
        for i in range(len(res)):
            for x in all_cols:
                if x not in res[i].columns.tolist():
                    res[i][x] = ["nan"]
            res[i] = res[i].loc[:, all_cols].copy()
        res = pandas.concat(res, axis=0)
        # Modify columns
        if "prec_test" in res.columns:
            res["f1_test"] = 2 * res["prec_test"].values * res["rec_test"].values / \
                             (res["prec_test"].values + res["rec_test"].values + 1e-10)
        res["chain_key"] = pandas.Categorical(
            res["chain_key"].values,
            ordered=True,
            categories=np.sort(np.unique(res["chain_key"]))
        )
        # Subset:
        if len(subset) > 0:
            for i, si in enumerate(subset):
                res = res[res[si[0]].values == si[1]].copy()
        return res

    def load_y(self, ids):
        y = {}
        yhat = {}
        peptide_seqs = {}
        for x in ids:
            fn = self.dirs[x] + "/models/" + x + "_y_test.npz"
            if os.path.isfile(fn):
                y[x] = np.asarray(scipy.sparse.load_npz(file=fn).todense())
            else:
                y[x] = None
            fn = self.dirs[x] + "/models/" + x + "_yhat_test.npy"
            if os.path.isfile(fn):
                yhat[x] = np.load(file=fn)
            else:
                yhat[x] = None
            fn = self.dirs[x] + '/models/' + x + "_peptide_seqs_test.csv"
            if os.path.isfile(fn):
                if os.path.isfile(fn):
                    peptide_seqs[x] = pandas.read_csv(fn)["antigen"].values
            else:
                peptide_seqs[x] = None
        self.y.update(y)
        self.yhat.update(yhat)
        self.peptide_seqs.update(peptide_seqs)

    def del_y(self):
        self.y = {}
        self.yhat = {}
        self.peptide_seqs = {}

    def load_deviation_metrics(self, ids):
        self.load_y(ids=ids)
        for x in ids:
            mse_global, msle_global, r2_global, r2log_global = deviation_global(
                y_hat=[self.y[x]],
                y_obs=[self.yhat[x]],
                labels=[self.peptide_seqs[x]],
            )
            self.evaluations_custom[x]["test"]["global"]["mse"] = mse_global
            self.evaluations_custom[x]["test"]["global"]["msle"] = msle_global
            self.evaluations_custom[x]["test"]["global"]["r2"] = r2_global
            self.evaluations_custom[x]["test"]["global"]["r2log"] = r2log_global

            mse_label, msle_label, r2_label, r2log_label = deviation_label(
                y_hat=[self.y[x]],
                y_obs=[self.yhat[x]],
                labels=[self.peptide_seqs[x]],
            )
            self.evaluations_custom[x]["test"]["local"]["mse"] = mse_label
            self.evaluations_custom[x]["test"]["local"]["msle"] = msle_label
            self.evaluations_custom[x]["test"]["local"]["r2"] = r2_label
            self.evaluations_custom[x]["test"]["local"]["r2log"] = r2log_label
        self.del_y()

    def reduce_table(
            self,
            tab,
            metric,
            partition,
            groups_keep=[]
    ):
        """
        Choose the best over categories based on mean loss in CV.

        Keep variation across CV.
        """
        metric_str = metric + "_" + partition
        if metric in [
            "loss", "keras_ce",
            "keras_mse", "keras_msle", "keras_rmse", "keras_cosine",
        ]:
            temp = tab.groupby(["run_group"] + groups_keep).transform("mean")
            for x in groups_keep:
                temp[x] = tab[x].values
            if len(groups_keep) > 0:
                idx = temp.groupby(groups_keep)[metric_str].idxmin().values
            else:
                idx = temp[metric_str].idxmin()
        elif metric in [
            "f1", "auc", "precision", "recall",
            "keras_f1", "keras_auc", "keras_precision", "keras_recall",
            "keras_poisson"
        ]:
            temp = tab.groupby([] + groups_keep).transform("mean")
            for x in groups_keep:
                temp[x] = tab[x].values
            if len(groups_keep) > 0:
                idx = temp.groupby(groups_keep)[metric_str].idxmax().values
            else:
                idx = temp[metric_str].idxmax()
        else:
            assert False
        if len(groups_keep) > 0:
            tab = tab.loc[[x in tab.loc[idx, "run_group"].values for x in tab["run_group"].values], :].copy()
        else:
            tab = tab.loc[[x in tab.loc[idx, "run_group"] for x in tab["run_group"].values], :].copy()
        return tab

    def rename_levels(
            self,
            tab,
            rename_levels=[]
    ):
        if len(rename_levels) > 0:
            for y in np.unique(np.asarray([x[0] for x in rename_levels])):
                levels_new = tab[y].values
                if not isinstance(levels_new, list):
                    levels_new = levels_new.tolist()
                for i, xx in enumerate(levels_new):
                    for rename_level in rename_levels:
                        if xx == rename_level[1]:
                            levels_new[i] = rename_level[2]
                levels_new = pandas.Categorical(
                    values=levels_new,
                    categories=[
                        x for i, x in enumerate([
                            xx[2] for xx in rename_levels
                            if xx[0] == y
                        ])
                        if x not in [xxx[2] for xxx in rename_levels[:i]]
                    ]
                )
                tab[y] = levels_new
        return tab

    def plot_metrics_topmodel(
            self,
            x,
            hue,
            select_metric,
            show_partition="test",
            select_partition="val",
            plot_f1=True,
            plot_roc_only=False,
            plot_fp_only=False,
            plot_tp_only=False,
            subset=[],
            rename_levels=[],
            ylim=None,
            xrot=0,
            show_swarm=False,
            width_fig=7,
            height_fig=7
    ):
        """ Average training loss across cross-validations. """
        results_tab = self.load_table(
            subset=subset
        )
        results_tab = self.reduce_table(
            tab=results_tab,
            metric=select_metric,
            partition=select_partition,
            groups_keep=[x, hue] if hue is not None else [x]
        )
        results_tab = self.rename_levels(
            tab=results_tab,
            rename_levels=rename_levels
        )
        return self._plot_best_model(
            sns_data=results_tab,
            x=x,
            hue=hue,
            show_partition=show_partition,
            width_fig=width_fig,
            height_fig=height_fig,
            plot_f1=plot_f1,
            plot_roc_only=plot_roc_only,
            plot_tp_only=plot_tp_only,
            plot_fp_only=plot_fp_only,
            xrot=xrot,
            ylim=ylim,
            show_swarm=show_swarm
        )

    def _plot_best_model(
            self,
            sns_data,
            x,
            hue,
            show_partition,
            plot_f1,
            plot_roc_only,
            plot_fp_only,
            plot_tp_only,
            xrot,
            ylim,
            show_swarm,
            width_fig,
            height_fig
    ):
        # Build figure.
        if plot_roc_only or plot_fp_only or plot_tp_only:
            if ylim is None:
                n_subplots = 1
                fig, axs = plt.subplots(n_subplots, 1, figsize=(width_fig, height_fig))
                axs = [axs]
            else:
                n_subplots = 2
                fig, axs = plt.subplots(n_subplots, 1, figsize=(width_fig, 4 * 1.5),
                                        gridspec_kw={'height_ratios': [1, 2]})
        else:
            if hue is not None and hue.lower() == "label" or x.lower() == "label":
                n_subplots = 3  # do not show loss
            else:
                n_subplots = 4
            if plot_f1:
                n_subplots = n_subplots - 1
            fig, axs = plt.subplots(n_subplots, 1, figsize=(width_fig, 3 * n_subplots))

        if plot_fp_only or plot_tp_only:
            if ylim is not None:
                sns.boxplot(
                    x=x,
                    y="fp_" + show_partition if plot_fp_only else "tp_" + show_partition,
                    hue=hue,
                    data=sns_data,
                    ax=axs[0]
                )
                if show_swarm:
                    sns.swarmplot(
                        x=x,
                        y="fp_" + show_partition if plot_fp_only else "tp_" + show_partition,
                        hue=hue,
                        data=sns_data,
                        ax=axs[0]
                    )
                if hue is not None:
                    axs[0].legend_.remove()
                axs[0].set_xticks([])
                axs[0].set_xlabel("")
                axs[0].set_ylabel("false-positive rate" if plot_fp_only else "true-positive rate")
                axs[0].set(ylim=ylim)
                i = 1
            else:
                i = 0
            sns.boxplot(
                x=x,
                y="fp_" + show_partition if plot_fp_only else "tp_" + show_partition,
                hue=hue,
                data=sns_data,
                ax=axs[i]
            )
            h, l = axs[i].get_legend_handles_labels()
            if show_swarm:
                sns.swarmplot(
                    x=x,
                    y="fp_" + show_partition if plot_fp_only else "tp_" + show_partition,
                    hue=hue,
                    data=sns_data,
                    ax=axs[i]
                )
            axs[i].legend(h, l, fontsize='10', labelspacing=0.2)
            axs[i].tick_params(axis='x', labelrotation=xrot)
            axs[i].set(ylim=[-0.01, 1.01])
            axs[i].set_ylabel("false-positive rate" if plot_fp_only else "true-positive rate")
        elif not plot_roc_only:
            if (hue is None or hue.lower() != "label") and x.lower() != "label" and not plot_roc_only:
                sns.boxplot(
                    x=x,
                    y="loss_" + show_partition,
                    hue=hue,  # if hue.lower() != "label" else None,
                    data=sns_data,
                    ax=axs[0]
                )
                if show_swarm:
                    sns.swarmplot(
                        x=x,
                        y="loss_" + show_partition,
                        hue=hue,  # if hue.lower() != "label" else None,
                        data=sns_data,
                        ax=axs[0]
                    )
                if hue is not None:
                    axs[0].legend_.remove()
                axs[0].set_xticks([])
                axs[0].set_xlabel("")
                axs[0].set_ylabel(show_partition + " loss")
                i = 1
            else:
                i = 0
            if plot_f1:
                sns.boxplot(
                    x=x,
                    y="f1_" + show_partition,
                    hue=hue,
                    data=sns_data,
                    ax=axs[i]
                )
                if show_swarm:
                    sns.swarmplot(
                        x=x,
                        y="f1_" + show_partition,
                        hue=hue,
                        data=sns_data,
                        ax=axs[i]
                    )
                if hue is not None:
                    axs[i].legend_.remove()
                axs[i].set_xticks([])
                axs[i].set_xlabel("")
                axs[i].set_ylabel(show_partition + " F1 score")
                i = i + 1
            else:
                sns.boxplot(
                    x=x,
                    y="prec_" + show_partition,
                    hue=hue,
                    data=sns_data,
                    ax=axs[i]
                )
                if show_swarm:
                    sns.swarmplot(
                        x=x,
                        y="prec_" + show_partition,
                        hue=hue,
                        data=sns_data,
                        ax=axs[i]
                    )
                if hue is not None:
                    axs[i].legend_.remove()
                axs[i].set_xticks([])
                axs[i].set_xlabel("")
                axs[i].set_ylabel(show_partition + " precision")
                i = i + 1

                sns.boxplot(
                    x=x,
                    y="rec_" + show_partition,
                    hue=hue,
                    data=sns_data,
                    ax=axs[i]
                )
                if show_swarm:
                    sns.swarmplot(
                        x=x,
                        y="rec_" + show_partition,
                        hue=hue,
                        data=sns_data,
                        ax=axs[i]
                    )
                if hue is not None:
                    axs[i].legend_.remove()
                axs[i].set_xticks([])
                axs[i].set_xlabel("")
                axs[i].set_ylabel(show_partition + " recall")
                i = i + 1
        else:
            i = 0

        if plot_roc_only and ylim is not None:
            sns.boxplot(
                x=x,
                y="auc_" + show_partition,
                hue=hue,
                data=sns_data,
                ax=axs[0]
            )
            if show_swarm:
                sns.swarmplot(
                    x=x,
                    y="auc_" + show_partition,
                    hue=hue,
                    data=sns_data,
                    ax=axs[0]
                )
            axs[0].set_xticks([])
            axs[0].set_xlabel("")
            axs[0].set_ylabel(show_partition + " AUC ROC")
            axs[0].set(ylim=ylim)
            if hue is not None:
                axs[0].legend_.remove()
            i = 1

        if not plot_fp_only and not plot_tp_only:
            sns.boxplot(
                x=x,
                y="auc_" + show_partition,
                hue=hue,
                data=sns_data,
                ax=axs[i]
            )
            h, l = axs[i].get_legend_handles_labels()
            if show_swarm:
                sns.swarmplot(
                    x=x,
                    y="auc_" + show_partition,
                    hue=hue,
                    data=sns_data,
                    ax=axs[i]
                )
            if hue is not None and hue.lower() == "label":
                axs[i].legend_.remove()
            else:
                axs[i].legend(h, l, fontsize='10', labelspacing=0.2)
            axs[i].tick_params(axis='x', labelrotation=xrot)
            axs[i].set_xlabel(x)
            axs[i].set_ylabel(show_partition + " AUC ROC")
            axs[i].set(ylim=[0.5, 1])
            i = i + 1
        return fig, axs

    def plot_confusion_mat(
            self,
            y_hat,
            y_obs,
            labels=None
    ):
        import sklearn

        if labels is None:
            labels = np.arange(0, y_obs.shape[1])
        y_hat = np.argmax(y_hat, axis=1)
        y_obs = np.argmax(y_obs, axis=1)
        cmat = sklearn.metrics.confusion_matrix(
            y_true=y_obs,
            y_pred=y_hat,
            labels=labels,
            sample_weight=None
        )
        sns_cmat = pandas.DataFrame(
            cmat,
            labels,
            labels
        )
        fig, axs = plt.subplots(1, 1, figsize=(7, 7))
        sns.heatmap(
            sns_cmat,
            annot=False,
            ax=axs
        )
        return fig, axs

    def plot_mse(
            self,
            x,
            hue,
            select_metric,
            show_partition="test",
            select_partition="val",
            subset=[],
            rename_levels=[],
            plot_only_r2=False,
            y_lim=None,
            xrot=0,
            width_fig=7,
            show_swarm=False
    ):
        """ Average training loss across cross-validations. """
        # Format all data.
        results_tab = self.load_table(
            subset=subset
        )
        results_tab = self.reduce_table(
            tab=results_tab,
            metric=select_metric,
            partition=select_partition,
            groups_keep=[x, hue] if hue is not None else [x]
        )
        results_tab = self.rename_levels(
            tab=results_tab,
            rename_levels=rename_levels
        )
        return self._plot_mse(
            sns_data=results_tab,
            x=x,
            hue=hue,
            plot_only_r2=plot_only_r2,
            y_lim=y_lim,
            xrot=xrot,
            show_swarm=show_swarm,
            width_fig=width_fig
        )

    def _plot_mse(
            self,
            sns_data,
            x,
            hue,
            y_lim,
            plot_only_r2,
            xrot,
            show_swarm,
            width_fig
    ):
        # Build figure.
        if plot_only_r2:
            n_subplots = 2 if y_lim is not None else 1
            if y_lim is None:
                fig, axs = plt.subplots(n_subplots, 1, figsize=(width_fig, 4))
                axs = [axs]
            else:
                fig, axs = plt.subplots(n_subplots, 1, figsize=(width_fig, 4 * 1.5),
                                        gridspec_kw={'height_ratios': [1, 2]})
        else:
            n_subplots = 2
            fig, axs = plt.subplots(n_subplots, 1, figsize=(width_fig, 4 * n_subplots))

        if not plot_only_r2:
            sns.boxplot(
                x=x,
                y="msle_test",
                hue=hue,
                data=sns_data,
                ax=axs[0]
            )
            if show_swarm:
                sns.swarmplot(
                    x=x,
                    y="msle_test",
                    hue=hue,
                    data=sns_data,
                    ax=axs[0]
                )
            if hue is not None:
                axs[0].legend_.remove()
            axs[0].set_xticks([])
            axs[0].set_xlabel("")
            axs[0].set_ylabel("test MSLE")
            i = 1
        else:
            if y_lim is not None:
                sns.boxplot(
                    x=x,
                    y="r2log_test",
                    hue=hue,
                    data=sns_data,
                    ax=axs[0]
                )
                if show_swarm:
                    sns.swarmplot(
                        x=x,
                        y="r2log_test",
                        hue=hue,
                        data=sns_data,
                        ax=axs[0]
                    )
                axs[0].set_xticks([])
                axs[0].set_xlabel("")
                axs[0].set_ylabel("test R2 (log)")
                axs[0].set(ylim=y_lim)
                axs[0].legend_.remove()
                i = 1
            else:
                i = 0

        sns.boxplot(
            x=x,
            y="r2log_test",
            hue=hue,
            data=sns_data,
            ax=axs[i]
        )
        h, l = axs[i].get_legend_handles_labels()
        if show_swarm:
            sns.swarmplot(
                x=x,
                y="r2log_test",
                hue=hue,
                data=sns_data,
                ax=axs[i]
            )
        axs[i].legend(h, l, fontsize='10', labelspacing=0.2)
        axs[i].tick_params(axis='x', labelrotation=xrot)
        axs[i].set_xlabel(x)
        axs[i].set_ylabel("test R2 (log)")
        axs[i].set(ylim=[0, 1])
        return fig, axs

    def plot_metrics_topmodel_split(
            self,
            hue,
            select_metric,
            show_partition="test",
            select_partition="val",
            drop_last_label=True,
            plot_f1=True,
            plot_roc_only=False,
            subset=[],
            rename_levels=[],
            ylim=None,
            xrot=0,
            show_swarm=False,
            width_fig=7,
            height_fig=7
    ):
        """ Average training loss across cross-validations. """
        results_tab = self.load_table(
            subset=subset
        )
        results_tab = self.reduce_table(
            tab=results_tab,
            metric=select_metric,
            partition=select_partition,
            groups_keep=[hue] if hue is not None else []
        )
        # Divide table with selected models into label-wise performance:
        if len(self.labels[results_tab["run"].values[0]]) == 1 and \
                self.labels[results_tab["run"].values[0]] == ["bound"]:
            # sampled indexed labels with scalar prediction
            results_tab_label = pandas.concat([
                pandas.DataFrame(dict([
                    (k, [v[i] for x in self.peptide_seqs_unique[run]])
                    for k, v in results_tab.items()
                ] + [
                    ("label", self.peptide_seqs_unique[run])
                ] + [
                    (k + "_" + show_partition, v.flatten())
                    for k, v in self.evaluations_custom[run][show_partition]["local"].items()
                ]))
                for i, run in enumerate(results_tab["run"].values)
            ])
        else:
            # label indexed prediction vector
            results_tab_label = pandas.concat([
                pandas.DataFrame(dict([
                    (hue, [results_tab[hue].values[i] for j in
                         range(len(self.labels[run]) + int(not drop_last_label))]),
                    (
                        "label",
                        self.labels[run].tolist() if drop_last_label else
                        self.labels[run].tolist() + ["bound"]
                    )
                ] + [
                    (k + "_" + show_partition,
                    v.flatten()[:-1] if drop_last_label else v.flatten())
                    for k, v in self.evaluations_custom[run][show_partition]["local"].items()
                ]))
                for i, run in enumerate(results_tab["run"].values)
            ])
        results_tab_label = self.rename_levels(
            tab=results_tab_label,
            rename_levels=rename_levels
        )

        return self._plot_best_model(
            sns_data=results_tab_label,
            x="label",
            hue=hue,
            show_partition=show_partition,
            width_fig=width_fig,
            height_fig=height_fig,
            plot_f1=plot_f1,
            plot_roc_only=plot_roc_only,
            plot_fp_only=False,
            plot_tp_only=False,
            xrot=xrot,
            ylim=ylim,
            show_swarm=show_swarm
        )

    def plot_mse_split(
            self,
            hue,
            select_metric,
            show_partition="test",
            select_partition="val",
            drop_last_label=True,
            subset=[],
            rename_levels=[],
            plot_only_r2=False,
            y_lim=None,
            xrot=0,
            width_fig=7,
            show_swarm=False
    ):
        results_tab = self.load_table(
            subset=subset
        )
        results_tab = self.reduce_table(
            tab=results_tab,
            metric=select_metric,
            partition=select_partition,
            groups_keep=[hue] if hue is not None else []
        )
        results_tab = self.rename_levels(
            tab=results_tab,
            rename_levels=rename_levels
        )
        # Divide table with selected models into label-wise performance:
        if len(self.labels[results_tab["run"].values[0]]) == 1 and \
                self.labels[results_tab["run"].values[0]] == ["bound"]:
            # sampled indexed labels with scalar prediction
            results_tab_label = pandas.concat([
                pandas.DataFrame(dict([
                                          (k, [v[i] for x in self.peptide_seqs_unique[run]])
                                          for k, v in results_tab.items()
                                      ] + [
                                          ("label", self.peptide_seqs_unique[run])
                                      ] + [
                                          (k + "_" + show_partition, v.flatten())
                                          for k, v in self.evaluations_custom[run][show_partition]["local"].items()
                                      ]))
                for i, run in enumerate(results_tab["run"].values)
            ])
        else:
            # label indexed prediction vector
            results_tab_label = pandas.concat([
                pandas.DataFrame(dict([
                    (hue, [results_tab[hue].values[i] for j in range(len(self.labels[run]))]),
                    ("label", self.labels[run].tolist())
                ] + [
                    (k + "_" + show_partition, v.flatten())
                    for k, v in self.evaluations_custom[run][show_partition]["local"].items()
                ]))
                for i, run in enumerate(results_tab["run"].values)
            ])

        return self._plot_mse(
            sns_data=results_tab_label,
            x="label",
            hue=hue,
            plot_only_r2=plot_only_r2,
            y_lim=y_lim,
            xrot=xrot,
            show_swarm=show_swarm,
            width_fig=width_fig
        )
        return fig, axs
