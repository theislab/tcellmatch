import numpy as np
import tensorflow as tf
import unittest

import tcellmatch.api as tm


class _TestFfnBinary_Base(unittest.TestCase):

    def _test_ffn(
            self,
            model: str,
            cost: str = "bce",
            label_smoothing: float = 0.1,
            aa_embedding_dim=None,
            training_set: str = "iedb",
            test_set: str = "subset",
            epochs = 1,
            blosum=False,
            use_iedb_targetted=False,
            indir="~/gitDevelopment/tcellmatch_private/data/"
    ):
        """
        :param model: model name, include "linear", "conv", "bilstm", "bigru", "sa", "inception" and "nettcr" models.
        :param cost: cost function name, should only be "binary_crossentropy" or "bce"
        :param label_smoothing: Fraction of the label interval to take out during smoothing. The labels are mapped
            from [0, 1] into [label_smoothing/2, 1-label_smoothing/2] throught the following transform:

                f(x) = x*(1-label_smoothing) + 0.5*label_smoothing
        :param aa_embedding_dim: Dimension of the linear amino acid embedding, ie number of 1x1 convolutional filters.
            This is set to the input dimension if aa_embedding_dim==0.
        :param training_set:
            training dataset
            - "10x" for 10x database.
            - "iedb" for IE database.
            - "vdjdb" for VDJ database.
        :param test_set:
            testing dataset
            - "subset": split test set from training set.
            - "10x" for 10x database.
            - "iedb" for IE database.
            - "vdjdb" for VDJ database.
        :param epochs: number of epochs for training
        :param blosum: whether to use blosum encoding for TCR sequences.
        :param use_iedb_targetted: whether to use certain peptides as training set.
        :return:
        """
        print("Make sure that the files vdj_v1_hs_aggregated_donor1_binarized_matrix.csv,"
              " vdj_v2_hs_aggregated_donor1_binarized_matrix.csv,"
              " vdj_v1_hs_aggregated_donor1_clonotypes.csv and "
              " vdj_v1_hs_aggregated_donor2_clonotypes.csv are copied into ./data/example_files "
              " to execute these tests. Refer to the publication for download instructions.")

        # This does not belong into a unit test but we keep it here for now to make sure
        # that tf versions are not an issue:
        print(tf.__version__)

        # Path of input directory.
        blosumfile = indir + "blosum/BLOSUM50.txt"
        fn_iedb = indir+"example_files/iedb.csv"
        fns_vdjdb = [indir + x for x in ["example_files/vdjdb.tsv"]]
        fns = ["example_files/vdj_v1_hs_aggregated_donor1_binarized_matrix.csv",
               "example_files/vdj_v1_hs_aggregated_donor2_binarized_matrix.csv"]
        fns_clonotype = ["example_files/vdj_v1_hs_aggregated_donor1_clonotypes.csv",
                         "example_files/vdj_v1_hs_aggregated_donor2_clonotypes.csv"]

        # List of column names in .csv dataset of lables to predicts.
        target_ids = [
            'A0201_ELAGIGILTV_MART-1_Cancer_binder',
            'A0201_GILGFVFTL_Flu-MP_Influenza_binder',
            'A0201_GLCTLVAML_BMLF1_EBV_binder',
            'A0301_KLGGALQAK_IE-1_CMV_binder',
            'A0301_RLRAEAQVK_EMNA-3A_EBV_binder',
            'A1101_IVTDFSVIK_EBNA-3B_EBV_binder',
            'A1101_AVFDRKSDAK_EBNA-3B_EBV_binder',
            'B0801_RAKFKQLL_BZLF1_EBV_binder'
        ]
        # List of antigens for training ffn with iedb.
        iedb_categ_ids = [
            "GILGFVFTL",
            "NLVPMVATV",
            "GLCTLVAML",
            "LLWNGPMAV",
            "VLFGLGFAI"
        ]
        np.random.seed(0)
        # Create model object.
        ffn = tm.models.EstimatorBinary()

        # Create training dataset.
        if training_set == "iedb":
            ffn.read_iedb(
                fns=fn_iedb,
                fn_blosum=blosumfile,
                antigen_ids=iedb_categ_ids if use_iedb_targetted else None,
                blosum_encoding=blosum,
                is_train=True,
                chains="trb"
            )
            ffn.assign_clonotype(flavor="manhatten")
            ffn.downsample_clonotype(max_obs=1)
            # Remove rare antigens.
            ffn.remove_antigens_byfreq(min_count=5, data="train")
            # Sample negative binding pairs for training.
            ffn.sample_negative_data(is_train=True)
        elif training_set.lower() == "vdjdb":
            ffn.read_vdjdb(
                fns=fns_vdjdb,
                fn_blosum=blosumfile,
                blosum_encoding=blosum,
                is_train=True,
                chains="trb"
            )
            ffn.assign_clonotype(flavor="manhatten")
            ffn.downsample_clonotype(max_obs=10)
            # Sample negative binding pairs for training.
            ffn.sample_negative_data(is_train=True)
        elif training_set == "10x":
            ffn.read_binarized_matrix(
                fns=[indir + x for x in fns],
                fns_clonotype=[indir + x for x in fns_clonotype],
                is_train=True,
                chains="trb",
                label_cols=target_ids,
                add_non_binder_for_softmax=False,
                fn_blosum=blosumfile,
                blosum_encoding=blosum
            )
            ffn.downsample_clonotype(max_obs=10)
            # Transform binary matrix into (TCR, antigen) binding pairs.
            ffn.divide_multilabel_observations_by_label(
                is_train=True,
                down_sample_negative=True
            )
        else:
            raise ValueError("invalid training set")

        # Padding zeros to tcr sequences in both training and testing set to make sure they have same size.
        ffn.pad_sequence(target_len=40, sequence="tcr")
        ffn.pad_sequence(target_len=25, sequence="antigen")
        # Create test dataset.
        # "subset": splitting testset from training set.
        if test_set.lower() == "subset":
            ffn.sample_test_set(test_split=0.1)
            ffn.remove_overlapping_tcrs(data="test")
            # Sample negative binding pairs for training.
            ffn.sample_negative_data(is_train=False)
        elif test_set == "iedb":
            ffn.read_iedb(
                fns=fn_iedb,
                fn_blosum=blosumfile,
                blosum_encoding=blosum,
                is_train=False,
                chains="trb"
            )
            ffn.remove_overlapping_tcrs(data="test")
            # Sample negative binding pairs for training.
            ffn.sample_negative_data(is_train=False)
        elif test_set.lower() == "vdjdb":
            ffn.read_vdjdb(
                fns=fns_vdjdb,
                fn_blosum=blosumfile,
                blosum_encoding=blosum,
                is_train=False,
                chains="trb"
            )
            ffn.remove_overlapping_antigens(data="test")
            ffn.assign_clonotype(flavor="manhatten", data="test")
            ffn.downsample_clonotype(max_obs=1, data="test")
            # Sample negative binding pairs for training.
            ffn.sample_negative_data(is_train=False)
        elif test_set == "10x":
            ffn.read_binarized_matrix(
                fns=[indir + x for x in fns],
                fns_clonotype=[indir + x for x in fns_clonotype],
                is_train=False,
                chains="trb",
                label_cols=target_ids,
                add_non_binder_for_softmax=False,
                fn_blosum=blosumfile,
                blosum_encoding=blosum
            )
            ffn.downsample_clonotype(max_obs=10)
            # Transform binary matrix into (TCR, antigen) binding pairs.
            ffn.divide_multilabel_observations_by_label(
                is_train=False,
                down_sample_negative=True
            )
            ffn.remove_overlapping_antigens(data="test")
        else:
            assert False

        # Downsample data to given number of observations.
        ffn.downsample_data(n=1000, data="train")
        ffn.downsample_data(n=200, data="test")

        # Padding zeros to tcr sequences in both training and testing set to make sure they have same size.
        ffn.pad_sequence(target_len=40, sequence="tcr")
        ffn.pad_sequence(target_len=25, sequence="antigen")

        optimizer = "adam"

        # Create different models in ffn
        if model == "bilstm":
            ffn.build_bilstm(
                residual_connection=True,
                aa_embedding_dim=aa_embedding_dim,
                topology=[20, 20],
                optimizer=optimizer,
                split=True,
                loss=cost,
                lr=0.001,
                label_smoothing=label_smoothing,
                optimize_for_gpu=False
            )
        elif model == "bigru":
            ffn.build_bigru(
                residual_connection=True,
                aa_embedding_dim=aa_embedding_dim,
                topology=[20, 20],
                optimizer=optimizer,
                loss=cost,
                lr=0.001,
                label_smoothing=label_smoothing,
                optimize_for_gpu=False
            )
        elif model == "selfattention":
            ffn.build_self_attention(
                residual_connection=True,
                aa_embedding_dim=aa_embedding_dim,
                attention_size=[5, 5],
                attention_heads=[4, 4],
                optimizer=optimizer,
                loss=cost,
                lr=0.001,
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
                optimizer=optimizer,
                loss=cost,
                lr=0.001,
                label_smoothing=label_smoothing
            )
        elif model == "linear":
            ffn.build_linear(
                aa_embedding_dim=aa_embedding_dim,
                optimizer=optimizer,
                loss=cost,
                lr=0.001,
                label_smoothing=label_smoothing
            )
        elif model == "nettcr":
            ffn.build_nettcr(
                n_filters=100,
                n_hid=10,
                lr=0.001,
                aa_embedding_dim=aa_embedding_dim,
                optimizer=optimizer,
                label_smoothing=label_smoothing
            )
        elif model == "inception":
            ffn.build_inception(
                aa_embedding_dim=aa_embedding_dim,
                split=True,
                n_filters_1x1=[8],
                n_filters_out=[8],
                depth_final_dense=2,
                n_hidden=8,
                final_pool="max",
                residual_connection=False,
                optimizer=optimizer,
                lr=0.001,
                loss=cost,
                label_smoothing=label_smoothing
            )
        else:
            assert False

        # Training!
        ffn.train(epochs=epochs, steps_per_epoch=1, batch_size=8)
        ffn.evaluate()
        ffn.predict()
        if model != "nettcr":
            print(ffn.model.summary())
        return ffn


class TestFfn_TrainSets(_TestFfnBinary_Base):

    def test_training_set(self):
        """ Tests whether linear feed forward networks work for different training set. """
        _ = self._test_ffn(model="linear", training_set="iedb", test_set="subset", use_iedb_targetted=False, blosum=True)
        _ = self._test_ffn(model="linear", training_set="iedb", test_set="subset", use_iedb_targetted=True)
        _ = self._test_ffn(model="linear", training_set="vdjdb", test_set="subset")
        _ = self._test_ffn(model="linear", training_set="10x", test_set="subset")


class TestFfn_TestSets(_TestFfnBinary_Base):

    def test_test_set_homogenous(self):
        """ Tests whether linear feed forward networks work training and testing set derived from same data base. """
        _ = self._test_ffn(model="linear", training_set="iedb", test_set="iedb")
        _ = self._test_ffn(model="linear", training_set="vdjdb", test_set="vdjdb")
        _ = self._test_ffn(model="linear", training_set="10x", test_set="10x")

    def _test_external_test_sets_iedb_trained(
            self,
            model="linear",
            blosum: bool = True,
            indir="~/Desktop/tcellmatch/data/"
    ):
        """ Test external testset on trained ffn models with iedb dataset. """
        fns_vdjdb = ["example_files/vdjdb.tsv"]
        blosumfile = indir + "blosum/BLOSUM50.txt"
        fns_10x = [
            "example_files/vdj_v1_hs_aggregated_donor1_binarized_matrix.csv",
            "example_files/vdj_v1_hs_aggregated_donor2_binarized_matrix.csv"
        ]
        fns_10x_clonotype = [
            "example_files/vdj_v1_hs_aggregated_donor1_clonotypes.csv",
            "example_files/vdj_v1_hs_aggregated_donor2_clonotypes.csv"
        ]
        target_ids = [
            'A0201_ELAGIGILTV_MART-1_Cancer_binder',
            'A0201_GILGFVFTL_Flu-MP_Influenza_binder',
            'A0201_GLCTLVAML_BMLF1_EBV_binder',
            'A0301_KLGGALQAK_IE-1_CMV_binder',
            'A0301_RLRAEAQVK_EMNA-3A_EBV_binder',
            'A1101_IVTDFSVIK_EBNA-3B_EBV_binder',
            'A1101_AVFDRKSDAK_EBNA-3B_EBV_binder',
            'B0801_RAKFKQLL_BZLF1_EBV_binder'
        ]
        # Train model on IEDB:
        ffn = self._test_ffn(model=model, training_set="iedb", test_set="subset",
                             aa_embedding_dim=None if blosum else 20)

        # Predict with model on VDJdb.
        ffn.clear_test_data()
        ffn.read_vdjdb(
            fns=[indir + x for x in fns_vdjdb],
            fn_blosum=blosumfile,
            blosum_encoding=blosum,
            chains="trb",
            is_train=False
        )
        ffn.assign_clonotype(flavor="manhatten", data="test")
        ffn.downsample_clonotype(max_obs=10, data="test")
        ffn.downsample_data(n=200, data="test")
        ffn.evaluate()
        ffn.predict()

        # Predict with model on 10x CD8
        ffn.clear_test_data()
        ffn.read_binarized_matrix(
            fns=[indir + x for x in fns_10x],
            fns_clonotype=[indir + x for x in fns_10x_clonotype],
            chains="trb",
            add_non_binder_for_softmax=False,
            label_cols=target_ids,
            fn_blosum=blosumfile,
            blosum_encoding=blosum,
            is_train=False
        )
        ffn.downsample_clonotype(max_obs=10, data="test")
        ffn.divide_multilabel_observations_by_label(is_train=False, down_sample_negative=True)
        ffn.downsample_data(n=200, data="test")
        ffn.evaluate()
        ffn.predict()

    def test_external_test_sets_iedb_trained(self):
        """ Test external testset on trained ffn models with iedb dataset """
        self._test_external_test_sets_iedb_trained(blosum=False)
        self._test_external_test_sets_iedb_trained(blosum=True)


class TestFfn_Saving(_TestFfnBinary_Base):

    def test_load_model_on_vdjdb(
            self,
            indir="~/Desktop/tcellmatch/data/",
            fn_tmp="~/Desktop/temp"
    ):
        """ Test loading trained model on vdjdb dataset """
        fn_vdjdb = indir + "example_files/vdjdb.tsv"
        blosumfile = indir + "blosum/BLOSUM50.txt"

        ffn = self._test_ffn(model="bilstm", training_set="iedb", test_set="random")
        ffn.save_model_full(fn_tmp)
        eval1 = np.array(ffn.results_test)
        pred1 = ffn.predictions

        # Try to reproduce evaluation in a new instance of model that receives same weights.
        ffn2 = tm.models.EstimatorFfn()
        ffn2.load_model_full(fn_tmp)
        # Recover prediction: validate model.
        ffn2.evaluate()
        ffn2.predict()
        eval2 = np.array(ffn2.results_test)
        pred2 = ffn2.predictions
        # Compare results and predictions of both models.
        mad_eval = np.mean(np.abs(eval1 - eval2))
        mad_pred = np.mean(np.abs(pred1 - pred2))
        assert mad_eval < 1e-12 and mad_pred < 1e-12, "mad_eval %f, mad_pred %f" % (mad_eval, mad_pred)
        # Run new data through model.
        ffn2.read_vdjdb(
            fns=fn_vdjdb,
            fn_blosum=blosumfile,
            blosum_encoding=ffn2.model_hyperparam["aa_embedding_dim"] is None,
            is_train=False
        )
        ffn2.evaluate()
        ffn2.evaluate_custom()
        ffn2.predict()
        eval_vdj = np.array(ffn2.results_test)
        pred_vdj = ffn2.predictions


class TestFfn_Models(_TestFfnBinary_Base):

    def test_losses(self):
        """ Tests whether probability and continuous positive output types work. """
        _ = self._test_ffn(model="bigru", aa_embedding_dim=0, cost="cce")
        _ = self._test_ffn(model="bigru", aa_embedding_dim=0, cost="mse")
        _ = self._test_ffn(model="bigru", aa_embedding_dim=0, cost="msle")

    def test_label_smoothing(self):
        """Test whether label smoothing works."""
        _ = self._test_ffn(model="bigru", label_smoothing=0.1)

    def test_nettcr(self):
        """Test whether nettcr works."""
        _ = self._test_ffn(model="nettcr", aa_embedding_dim=None)
        _ = self._test_ffn(model="nettcr", aa_embedding_dim=0)

    def test_bilstm(self):
        """Test whether BiLSTM works."""
        _ = self._test_ffn(model="bilstm", aa_embedding_dim=None)
        _ = self._test_ffn(model="bilstm", aa_embedding_dim=0)

    def test_bigru(self):
        """Test whether BiGRU works."""
        _ = self._test_ffn(model="bigru", aa_embedding_dim=None)
        _ = self._test_ffn(model="bigru", aa_embedding_dim=0)

    def test_selfattention(self):
        """ Tests whether self-attention network works. """
        _ = self._test_ffn(model="selfattention", aa_embedding_dim=None)
        _ = self._test_ffn(model="selfattention", aa_embedding_dim=0)

    def test_conv(self):
        """ Tests whether convolution network works. """
        _ = self._test_ffn(model="conv", aa_embedding_dim=None)
        _ = self._test_ffn(model="conv", aa_embedding_dim=0)

    def test_inception(self):
        """ Tests whether inception network works. """
        _ = self._test_ffn(model="inception", aa_embedding_dim=None, label_smoothing=0)


if __name__ == '__main__':
    unittest.main()
