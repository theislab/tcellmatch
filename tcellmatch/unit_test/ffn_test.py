import numpy as np
import tensorflow as tf
from typing import Union
import unittest

import tcellmatch.api as tm


class _TestFfn_Base(unittest.TestCase):

    # DEFINE THESE PATHS TO RUN UNIT TESTS
    fn_tmp = "."
    indir = "."

    def _test_ffn(
            self,
            model,
            chains,
            aa_embedding_dim: Union[int, None] = 0,
            cost: str = "cce",
            epochs: int = 1,
            use_clono: bool = True,
            split_labels: bool = False,
            log1p_labels: bool = False,
            covariate_formula_numeric: list = [],
            nc: bool = False,
            downsample: bool = True,
            train_set: str = "10x",
            test_set: str = "subset",
            split_idx: int = 2,
            blosum: bool = False,
    ):
        """
        :param model: model name, include "linear", "conv", "bilstm", "bigru", "sa" and "inception" models.
        :param chains: {"tra",  "trb", "separate", "concat"} Which TCR CDR chain(s) to keep.

            - "tra": Only keep TRA chain and discard TRB chain. Not supported for IEDB.
            - "trb": Only keep TRB chain and discard TRA chain.
            - "separate": Keep both TRA and TRB chain as separate entries, ie. with a TCR chain dimension
                of length 2 in the input data tensor. This can be used to compute independent embeddings of
                both chains with RNN models.
            - "concat": Keep both TRA and TRB chain as a single entry, ie. with a TCR chain dimension
                of length 1 in the input data tensor. This concatenation allows the computation of a single
                embedding of both chains with a RNN model.
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param cost: cost function name

            - "categorical_crossentropy", "cce" for single boolean binding events with binary crossentropy loss.
            - "binary_crossentropy", "bce" for multiple boolean binding events with binary crossentropy loss.
            - "weighted_binary_crossentropy", "wbce" for multiple boolean binding events with
                weighted binary crossentropy loss.
            - "mean_squared_error", "mse" for continuous value prediction with mean squared error loss.
            - "mean_squared_logarithmic_error", "msle" for continuous value prediction with mean squared
                logarithmic error loss.
        :param epochs: number of epochs for training
        :param use_clono: whether to use preprocessed clonotypes as addition inputs.
        :param split_labels: split certain labels in target_ids for training.
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param log1p_labels: whether to use log transform training and test labels
        :param covariate_formula_numeric: Terms for patsy formula to build numeric covariate matrix from based on input table.
        :param nc: whether to use negative controls.
        :param extra_covar: whether to use extra covariates for training
        :param downsample: Downsample clonotypes to data stored in x_train, this avoids training, evaluation or test set to be too biased to a subset of TCRs.
        :param train_set: training dataset
            - "10x" for 10x database.
            - "iedb" for IE database.
        :param test_set: testing dataset
            - "subset": split test set from training set.
            - "vdjdb" for VDJ database.
        :param split_idx: index for splitting labels.
        :param blosum: whether to use blosum encoding for TCR sequences.
        :return:
        """
        print(
            "Make sure that the files vdj_v1_hs_aggregated_donor1_binarized_matrix.csv,"
            " vdj_v2_hs_aggregated_donor1_binarized_matrix.csv,"
            " vdj_v1_hs_aggregated_donor1_clonotypes.csv and "
            " vdj_v1_hs_aggregated_donor2_clonotypes.csv are copied into ./data/example_files "
            " to execute these tests. Refer to the publication for download instructions."
        )
        # This does not belong into a unit test but we keep it here for now to make sure
        # that tf versions are not an issue:
        print(tf.__version__)

        # Path of input directory.
        blosumfile = self.indir + "blosum/BLOSUM50.txt"
        fn_iedb = self.indir + "tcell_receptor_table_export_1558607498.csv"
        fn_vdjdb = self.indir + "vdjdb.tsv"
        fns = ["vdj_v1_hs_aggregated_donor1_binarized_matrix.csv",
               "vdj_v1_hs_aggregated_donor2_binarized_matrix.csv"]
        fns_clonotype = [
            "vdj_v1_hs_aggregated_donor1_clonotypes.csv",
            "vdj_v1_hs_aggregated_donor2_clonotypes.csv"
        ]

        # List of column names in .csv dataset of lables to predicts.
        target_ids = [
            'A0101_VTEHDTLLY_IE-1_CMV_binder',
            'A0201_KTWGQYWQV_gp100_Cancer_binder',
            'A0201_ELAGIGILTV_MART-1_Cancer_binder',
            'A0201_CLLWSFQTSA_Tyrosinase_Cancer_binder',
            'A0201_IMDQVPFSV_gp100_Cancer_binder',
            'A0201_SLLMWITQV_NY-ESO-1_Cancer_binder',
            'A0201_KVAELVHFL_MAGE-A3_Cancer_binder',
            'A0201_KVLEYVIKV_MAGE-A1_Cancer_binder',
            'A0201_CLLGTYTQDV_Kanamycin-B-dioxygenase_binder',
            'A0201_LLDFVRFMGV_EBNA-3B_EBV_binder',
            'A0201_LLMGTLGIVC_HPV-16E7_82-91_binder',
            'A0201_CLGGLLTMV_LMP-2A_EBV_binder',
            'A0201_YLLEMLWRL_LMP1_EBV_binder',
            'A0201_FLYALALLL_LMP2A_EBV_binder',
            'A0201_GILGFVFTL_Flu-MP_Influenza_binder',
            'A0201_GLCTLVAML_BMLF1_EBV_binder',
            'A0201_NLVPMVATV_pp65_CMV_binder',
            'A0201_ILKEPVHGV_RT_HIV_binder',
            'A0201_FLASKIGRLV_Ca2-indepen-Plip-A2_binder',
            'A2402_CYTWNQMNL_WT1-(235-243)236M_Y_binder',
            'A0201_RTLNAWVKV_Gag-protein_HIV_binder',
            'A0201_KLQCVDLHV_PSA146-154_binder',
            'A0201_LLFGYPVYV_HTLV-1_binder',
            'A0201_SLFNTVATL_Gag-protein_HIV_binder',
            'A0201_SLYNTVATLY_Gag-protein_HIV_binder',
            'A0201_SLFNTVATLY_Gag-protein_HIV_binder',
            'A0201_RMFPNAPYL_WT-1_binder',
            'A0201_YLNDHLEPWI_BCL-X_Cancer_binder',
            'A0201_MLDLQPETT_16E7_HPV_binder',
            'A0301_KLGGALQAK_IE-1_CMV_binder',
            'A0301_RLRAEAQVK_EMNA-3A_EBV_binder',
            'A0301_RIAAWMATY_BCL-2L1_Cancer_binder',
            'A1101_IVTDFSVIK_EBNA-3B_EBV_binder',
            'A1101_AVFDRKSDAK_EBNA-3B_EBV_binder',
            'B3501_IPSINVHHY_pp65_CMV_binder',
            'A2402_AYAQKIFKI_IE-1_CMV_binder',
            'A2402_QYDPVAALF_pp65_CMV_binder',
            'B0702_QPRAPIRPI_EBNA-6_EBV_binder',
            'B0702_TPRVTGGGAM_pp65_CMV_binder',
            'B0702_RPPIFIRRL_EBNA-3A_EBV_binder',
            'B0702_RPHERNGFTVL_pp65_CMV_binder',
            'B0801_RAKFKQLL_BZLF1_EBV_binder',
            'B0801_ELRRKMMYM_IE-1_CMV_binder',
            'B0801_FLRGRAYGL_EBNA-3A_EBV_binder',
            'A0101_SLEGGGLGY_NC_binder',
            'A0101_STEGGGLAY_NC_binder',
            'A0201_ALIAPVHAV_NC_binder',
            'A2402_AYSSAGASI_NC_binder',
            'B0702_GPAESAAGL_NC_binder',
            'NR(B0801)_AAKGRGAAL_NC_binder'
        ]
        # List of column names in .csv dataset file fn of negative controls.
        nc_cols = [
            'A0101_SLEGGGLGY_NC_binder',
            'A0101_STEGGGLAY_NC_binder',
            'A0201_ALIAPVHAV_NC_binder',
            'A2402_AYSSAGASI_NC_binder',
            'B0702_GPAESAAGL_NC_binder',
            'NR(B0801)_AAKGRGAAL_NC_binder'
        ]
        # List of antigens for training ffn with iedb.
        iedb_categ_ids = [
            "GILGFVFTL",
            "NLVPMVATV",
            "GLCTLVAML",
            "LLWNGPMAV",
            "VLFGLGFAI"
        ]

        # Set different target set according to loss funciton
        # For binary classification
        if cost.lower() in ["cce", "bce", "wbce"]:
            target_ids = target_ids
            label_smoothing = 0.1
            transform = None
        # For continuous value prediction
        elif cost.lower() in ["mse", "msle"]:
            target_ids = [x.split("_binder")[0] for x in target_ids]
            nc_cols = [x.split("_binder")[0] for x in nc_cols] if nc else None
            label_smoothing = 0
            if train_set.lower() == "10x":
                transform = "10x_cd8_v1"
            else:
                assert False
        else:
            raise ValueError("cost %s not recognized" % cost)

        # Create model object.
        ffn = tm.models.EstimatorFfn()

        #Create training dataset.
        if train_set.lower() == "10x":
            if split_labels:
                target_ids = target_ids[split_idx]
            ffn.read_binarized_matrix(
                fns=[self.indir + x for x in fns],
                fns_clonotype=[self.indir + x for x in fns_clonotype] if use_clono else None,
                fn_blosum=blosumfile,
                blosum_encoding=blosum,
                is_train=True,
                covariate_formula_categ=["donor"],
                covariate_formula_numeric=covariate_formula_numeric,
                add_non_binder_for_softmax=True if cost.lower() == "cce" else False,
                chains=chains,
                label_cols=target_ids,
                nc_cols=nc_cols if nc else None,
                add_nc_to_covariates=nc
            )
            if downsample:
                ffn.downsample_clonotype(max_obs=1)
            if log1p_labels:
                ffn.log1p_labels()
        elif train_set.lower() == "iedb":
            ffn.read_iedb_as_categorical_model(
                fns=fn_iedb,
                antigen_ids=iedb_categ_ids if not split_labels else [iedb_categ_ids[0]],
                fn_blosum=blosumfile,
                blosum_encoding=blosum,
                add_non_binder_for_softmax=not split_labels,
                is_train=True
            )
            ffn.assign_clonotype(flavor="manhatten")
            ffn.downsample_clonotype(max_obs=10)
        else:
            assert False

        # Create test dataset.
        # "subset": splitting testset from training set.
        if test_set.lower() == "subset":
            if split_labels:
                ffn.downsample_labels(data="train", multiple_of_min=2)
                ffn.sample_test_set(test_split=0.1)
                ffn.remove_overlapping_tcrs(data="test")
            else:
                ffn.sample_test_set(test_split=0.1)
        elif test_set.lower() == "vdjdb":
            ffn.read_vdjdb(
                fns=[fn_vdjdb],
                fn_blosum=blosumfile,
                blosum_encoding=blosum,
                is_train=False
            )
        elif test_set.lower() == "none":
            pass
        else:
            assert False

        #Padding zeros to tcr sequences in both training and testing set to make sure they have same size.
        tcr_padding = 80 if chains.lower() == "concat" else 40
        ffn.pad_sequence(target_len=tcr_padding, sequence="tcr")

        optimizer = "adam"

        # Create different models in ffn
        if model == "bilstm":
            ffn.build_bilstm(
                residual_connection=True,
                aa_embedding_dim=aa_embedding_dim,
                topology=[5, 5],
                optimizer=optimizer,
                lr=0.001,
                loss=cost,
                label_smoothing=label_smoothing,
                optimize_for_gpu=False
            )
        elif model == "bigru":
            ffn.build_bigru(
                residual_connection=True,
                aa_embedding_dim=aa_embedding_dim,
                topology=[5, 5],
                optimizer=optimizer,
                lr=0.001,
                loss=cost,
                label_smoothing=label_smoothing,
                optimize_for_gpu=False
            )
        elif model == "sa":
            ffn.build_self_attention(
                residual_connection=True,
                aa_embedding_dim=aa_embedding_dim,
                attention_size=[5, 5],
                attention_heads=[4, 4],
                optimizer=optimizer,
                lr=0.001,
                loss=cost,
                label_smoothing=label_smoothing
            )
        elif model == "conv":
            ffn.build_conv(
                aa_embedding_dim=aa_embedding_dim,
                activations=["relu", "relu", "relu"],
                filter_widths=[4, 4, 4],
                filters=[10, 10, 6],
                strides=[1, 1, 1],
                pool_sizes=[None, None, 2],
                batch_norm=False,
                optimizer=optimizer,
                lr=0.001,
                loss=cost,
                label_smoothing=label_smoothing
            )
        elif model == "inception":
            ffn.build_inception(
                aa_embedding_dim=aa_embedding_dim,
                split=False,
                n_filters_1x1=[10, 10, 10],
                n_filters_out=[20, 20, 20],
                residual_connection=True,
                optimizer=optimizer,
                lr=0.001,
                loss=cost,
                label_smoothing=label_smoothing
            )
        elif model == "linear":
            ffn.build_linear(
                aa_embedding_dim=aa_embedding_dim,
                optimizer=optimizer,
                loss=cost,
                label_smoothing=label_smoothing
            )
        elif model == "noseq":
            ffn.build_noseq(
                optimizer=optimizer,
                loss=cost,
                label_smoothing=label_smoothing
            )
        else:
            assert False

        # Downsample data to given number of observations.
        if downsample:
            ffn.downsample_data(n=200, data="train")
            ffn.downsample_data(n=200, data="test")

        # Training!
        ffn.train(
            epochs=epochs,
            batch_size=128
        )
        if test_set.lower() != "none":
            ffn.evaluate()
            ffn.evaluate_custom(transform=transform)
            ffn.predict()
        print(ffn.model.training_model.summary())
        return ffn


class TestFfn_TestSets(_TestFfn_Base):

    def _test_external_test_10x_trained(
            self,
            blosum: bool,
            split_labels: bool,
            indir="~/Desktop/tcellmatch/data/"
    ):
        """ Test external testset on trained ffn models with 10x dataset """
        blosumfile = indir + "blosum/BLOSUM50.txt"
        fns_vdjdb = ["example_files/vdjdb.tsv"]
        fns_iedb = ["example_files/iedb.csv"]
        # Train model on 10x:
        ffn = self._test_ffn(
            model="linear",
            chains="concat",
            cost="bce",
            blosum=blosum,
            test_set="subset",
            split_labels=split_labels,
            split_idx=2 if split_labels else None,
            covariate_formula_numeric=["CD3"]
        )

        # Predict with model on VDJdb entries unmatched to 10x.
        ffn.clear_test_data()
        ffn.read_vdjdb_matched_to_categorical_model(
            fns=[indir + x for x in fns_vdjdb],
            fn_blosum=blosumfile,
            blosum_encoding=blosum,
            is_train=False,
            same_antigen=False
        )
        ffn.assign_clonotype(flavor="manhatten", data="test")
        ffn.downsample_clonotype(max_obs=10, data="test")
        ffn.evaluate()
        ffn.predict()

        # Predict with model on IEDB entries unmatched to 10x.
        ffn.clear_test_data()
        ffn.read_iedb_matched_to_categorical_model(
            fns=[indir + x for x in fns_iedb],
            fn_blosum=blosumfile,
            blosum_encoding=blosum,
            is_train=False,
            same_antigen=False
        )
        ffn.assign_clonotype(flavor="manhatten", data="test")
        ffn.downsample_clonotype(max_obs=10, data="test")
        ffn.evaluate()
        ffn.predict()

    def _test_external_test_iedb_trained(
            self,
            blosum: bool,
            split_labels: bool,
            indir="~/Desktop/tcellmatch/data/"
    ):
        """ Test external testset on trained ffn models with iedb dataset """
        blosumfile = indir + "blosum/BLOSUM50.txt"
        fns_vdjdb = [indir + "example_files/vdjdb.tsv"]
        # Train model on 10x:
        ffn = self._test_ffn(
            model="linear",
            chains="concat",
            cost="cce",
            blosum=blosum,
            train_set="iedb",
            test_set="subset",
            split_labels=split_labels,
            split_idx=2 if split_labels else None
        )

        # Predict with model on VDJdb entries matched to IEDB categorical subset.
        ffn.clear_test_data()
        ffn.read_vdjdb_matched_to_categorical_model(
            fns=fns_vdjdb,
            fn_blosum=blosumfile,
            blosum_encoding=blosum,
            is_train=False,
            same_antigen=True
        )
        ffn.assign_clonotype(flavor="manhatten", data="test")
        ffn.downsample_clonotype(max_obs=10, data="test")
        ffn.evaluate()
        ffn.predict()

    def test_external_test_10x_trained(self):
        """ Test external testset on trained ffn models with 10x dataset """
        self._test_external_test_10x_trained(split_labels=True, blosum=False)
        self._test_external_test_10x_trained(split_labels=True, blosum=True)
        self._test_external_test_10x_trained(split_labels=False, blosum=False)
        self._test_external_test_10x_trained(split_labels=False, blosum=True)

    def test_iedb_categorical_iedb_trained(self):
        """ Test external testset on trained ffn models with iedb dataset """
        self._test_external_test_iedb_trained(split_labels=False, blosum=False)

    def test_test_mode(self):
        """ Test that different test sets work. """
        self._test_ffn(model="bilstm", chains="trb", cost="cce", test_set="subset")
        self._test_ffn(model="bilstm", chains="trb", cost="cce", test_set="vdjdb")


class TestFfn_Input(_TestFfn_Base):

    def test_split(self):
        """ Test label splitting. """
        _ = self._test_ffn(model="linear", chains="trb", epochs=0, train_set="iedb", test_set="subset", split_labels=True)
        _ = self._test_ffn(model="linear", chains="trb", epochs=0, train_set="10x", test_set="subset", split_labels=True)

    def test_nc_covar(self):
        """ Test inclusion of negative controls as a covariate."""
        _ = self._test_ffn(model="bigru", chains="tra", epochs=0, train_set="10x", test_set="subset", nc=True)

    def test_chain_style(self):
        """ Test input chain configuration."""
        _ = self._test_ffn(model="bigru", chains="tra", epochs=0, train_set="10x", test_set="subset")
        _ = self._test_ffn(model="bigru", chains="trb", epochs=0, train_set="10x", test_set="subset")
        _ = self._test_ffn(model="bigru", chains="concat", epochs=0, train_set="10x", test_set="subset")
        #_ = self._test_ffn(model="bigru", chains="separate", epochs=0, train_set="10x", test_set="subset")

    def test_covariates(self):
        """ Test that inclusion of covariates works. """
        _ = self._test_ffn(model="bilstm", chains="trb", covariate_formula_numeric=[])
        ffn = self._test_ffn(
            model="bilstm", chains="trb",
            covariate_formula_numeric=[
                "CD3", "CD19", "CD45RA", "CD4", "CD8a", "CD14",
                "CD45RO", "CD279_PD-1", "IgG1", "IgG2a", "IgG2b", "CD127", "CD197_CCR7"
            ]
        )
        assert ffn.covariates_train.shape[1] == 15
        assert ffn.covariates_test.shape[1] == 15

    def test_clonotypes(self):
        """ Test that usage of clonotypes in preprocessing works. """
        _ = self._test_ffn(model="bilstm", chains="trb", cost="cce", use_clono=True)
        _ = self._test_ffn(model="bilstm", chains="trb", cost="cce", use_clono=False)


class TestFfn_Models(_TestFfn_Base):

        def test_losses(self):
            """ Tests different cost functions. """
            _ = self._test_ffn(model="bilstm", chains="trb", cost="cce")
            _ = self._test_ffn(model="bilstm", chains="trb", cost="bce")
            _ = self._test_ffn(model="bilstm", chains="trb", cost="mse", log1p_labels=True)
            _ = self._test_ffn(model="bilstm", chains="trb", cost="msle", log1p_labels=False)

        def test_bilstm(self):
            """ Tests whether BiLSTMs work. """
            _ = self._test_ffn(model="bilstm", chains="concat")

        def test_bigru(self):
            """ Tests whether BiGRUs work. """
            _ = self._test_ffn(model="bigru", chains="concat")

        def test_sa(self):
            """ Tests whether self-attention networks work. """
            _ = self._test_ffn(model="sa", chains="concat")

        def test_conv(self):
            """ Tests whether convolution networks work. """
            _ = self._test_ffn(model="conv", chains="concat")

        def test_inception(self):
            """ Tests whether inception networks work. """
            _ = self._test_ffn(model="inception", chains="concat")

        def test_dense(self):
            """ Tests whether dense networks work. """
            _ = self._test_ffn(model="linear", chains="concat")

        def test_noseq(self):
            """ Tests whether dense networks work. """
            _ = self._test_ffn(model="noseq", chains="concat")

        def test_aa_embedding(self):
            """ Tests whether different amino acid embeddings work. """
            _ = self._test_ffn(model="linear", chains="trb", aa_embedding_dim=None, blosum=True)
            _ = self._test_ffn(model="linear", chains="trb", aa_embedding_dim=None, blosum=False)
            _ = self._test_ffn(model="linear", chains="trb", aa_embedding_dim=5, blosum=False)


class TestFfn_Saving(_TestFfn_Base):

    def _test_model_saving(
            self,
            model: str,
            reduce_size: bool = False,
            fn_tmp: str = "/Users/david.fischer/Desktop/temp/temp"
    ):
        """ Tests whether entire model saving and loading works. """
        # Fit, evaluate and save a model.
        ffn = self._test_ffn(model=model, chains="trb", epochs=3)
        ffn.save_model_full(fn_tmp, reduce_size=reduce_size)
        eval1 = list(ffn.evaluations["train"].values())[0]
        ffn.predict()
        pred1 = ffn.predictions

        # Try to reproduce evaluation in a new instance of model that receives same weights.
        ffn2 = tm.models.EstimatorFfn()
        ffn2.load_model_full(fn_tmp)
        if reduce_size:
            ffn2.x_test = ffn.x_test
            ffn2.covariates_test = ffn.covariates_test
            ffn2.y_test = ffn.y_test
        ffn2.evaluate()
        ffn2.predict()
        pred2 = ffn2.predictions
        eval2 = list(ffn2.evaluations["train"].values())[0]

        # Compare results and predictions of both models.
        mad_eval = np.mean(np.abs(eval1 - eval2))
        mad_pred = np.mean(np.abs(pred1 - pred2))
        assert mad_eval < 1e-12 and mad_pred < 1e-12, "mad_eval %f, mad_pred %f" % (mad_eval, mad_pred)

    def test_model_saving(self):
        """ Tests whether entire model saving and loading works. """
        self._test_model_saving(model="conv", reduce_size=False)
        self._test_model_saving(model="bilstm", reduce_size=False)
        self._test_model_saving(model="bigru", reduce_size=False)
        self._test_model_saving(model="sa", reduce_size=False)

    def test_model_saving_reduced(self):
        """ Tests whether entire model saving and loading works if only reduced data are saved. """
        self._test_model_saving(model="bigru", reduce_size=True)

    def test_recover_data_split(self):
        """ Fit, evaluate and save the train-test split of a model."""
        ffn = self._test_ffn(model="bilstm", chains="trb", epochs=1, downsample=True, test_set="subset")
        ffn.save_idx(fn=self.fn_tmp)
        w = ffn.model.training_model.get_weights()
        pred1 = ffn.predictions

        # Run predictions on test set in second model, that has same weights and was loaded with same split.
        ffn2 = self._test_ffn(model="bilstm", chains="trb", epochs=1, downsample=True, test_set="none")
        ffn2.subset_from_saved(fn=self.fn_tmp)
        ffn2.model.training_model.set_weights(w)
        ffn2.predict()
        pred2 = ffn2.predictions

        # Compare results and predictions of both models.
        mad_pred = np.mean(np.abs(pred1 - pred2))
        assert mad_pred < 1e-12, "mad_pred %f" % mad_pred

    def test_results_saving(self):
        """ Tests whether entire model saving and loading works. """
        ffn = self._test_ffn(model="conv", chains="trb")
        ffn.save_results(self.fn_tmp)
        ffn.load_results(self.fn_tmp)


if __name__ == '__main__':
    unittest.main()
