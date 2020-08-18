import anndata
import pandas as pd
import random
import scipy.sparse
import tensorflow as tf
from typing import Union, Tuple, List
import numpy as np
import patsy
import os

from tcellmatch.utils.utils_aa_embedding import read_blosum, encode_as_blosum, encode_as_onehot


class EstimatorBase:
    """
    T-cell receptor (TCR) data estimator base class.

    Contains helper functions to run models on sequence data.

    Sequence format:
    TCR and antigens:
    The general sequence container has the dimensions:
    [observations, sequences of observation, positions in sequence, amino acid embedding dimensions]
    If multiple TCR chains are modelled, e.g. TRA and TRB, then these are currently always padded to the same maximal
    sequence length (dimension 3) and concatenated. Note that the TCR length always has to be a multiple of two
    if "concat" is used because padding will always operate independently on both chains.
    If antigen sequences are included, these are padded to an individual sequence length in dimension 3 based on
    self.pep_len. The entire antigen encoding is concatenated to the TCR sequences in dimension 3.
    Dimensions are set in the primary reading functions:

        - read_iedb
        - read_vdjdb
        - read_binarized_matrix
        - (read_consensus annotation)

    through calls to _format_tcr_chains and _format_antigen and are later adopted for further reading operations
    as soon self.tcr_len and self.pep_len respectively are set once.
    The length of TCR and antigen encoding can still be changed by the user through calling pad_sequences.

    Note: (developers) All shape updates and all new input must be channeled through _format_tcr_chains and
    _format_antigen to avoid shape bugs.
    Note: read_vdjdb_matched_to_categorical_mdoel can only be used as secondary reading operation and therefore adopts
    pres-set shapes.
    Note that clear_test_data clears all test data but will still be loaded according to these pre-set shapes.

    Amino acid embeddings:

        - one-hot
        - blosum
    """
    x_train: np.ndarray
    covariates_train: np.ndarray
    y_train: np.ndarray
    nc_train: np.ndarray
    clone_train: np.ndarray
    idx_train_val: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray

    x_test: np.ndarray
    covariates_test: np.ndarray
    y_test: np.ndarray
    nc_test: np.ndarray
    clone_test: np.ndarray
    idx_test: np.ndarray

    def __init__(self):
        # Assemble reference dictionary of amino acids.
        # Code explained here https://de.mathworks.com/help/bioinfo/ref/aminolookup.html
        # and here https://www.dnastar.com/MegAlign_Help/index.html#!Documents/iupaccodesforaminoacids.htm
        self.eos_char = "_"  # End-of-sequence char.
        self.aa_list = [
            'A', 'R', 'N', 'D', 'C',
            'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P',
            'S', 'T', 'W', 'Y', 'V',
            'B', 'J', 'Z', 'X', '*',
            self.eos_char
        ]
        idx_list = np.arange(0, len(self.aa_list))
        self.dict_aa = dict(zip(self.aa_list, idx_list))
        self.dict_bb = dict(zip(idx_list, self.aa_list))

        self.frac_positives = None
        self.frac_test_positives = None

        self.x_train = None
        self.covariates_train = None
        self.y_train = None
        self.nc_train = None
        self.clone_train = None
        self.idx_train_val = np.array([])
        self.idx_train = np.array([])
        self.idx_val = np.array([])

        self.x_test = None
        self.covariates_test = None
        self.y_test = None
        self.nc_test = None
        self.clone_test = None
        self.idx_test = np.array([])

        self.tcr_len = None  # Length of all CDR3 chains and padding together.
        self.pep_len = None
        self.chains = None
        self.label_cols = None
        self.label_ids = None
        self.add_non_binder_for_softmax = None

        # Full TCR and peptide sequences.
        self.peptide_seqs_train = None
        self.tra_seqs_train = None
        self.trb_seqs_train = None

        self.peptide_seqs_test = None
        self.tra_seqs_test = None
        self.trb_seqs_test = None

    @property
    def peptide_seqs_unique(self):
        peptide_seqs = []
        if self.peptide_seqs_test is not None:
            peptide_seqs = peptide_seqs + list(self.peptide_seqs_test)
        if self.peptide_seqs_train is not None:
            peptide_seqs = peptide_seqs + list(self.peptide_seqs_train)
        return np.sort(np.unique(np.asarray(peptide_seqs))).tolist()

    def read_iedb(
            self,
            fns: Union[list, str],
            fn_blosum: Union[str, None] = None,
            antigen_ids: Union[List[str], None] = None,
            chains: str = "trb",
            blosum_encoding: bool = False,
            is_train: bool = True,
            all_trb: bool = True
    ):
        """ Read IEDB download files.

        :param fns: File names of positive observations file, ie. the iedb download.
        :param fn_blosum: File with BLOSUM50.txt embedding matrix. Only needed if blosum_encoding==True.
        :param antigen_ids: Target antigen sequences: only load observations that match these.
        :param chains: {"tra",  "trb", "separate", "concat"} Which TCR CDR chain(s) to keep.

            - "tra": Only keep TRA chain and discard TRB chain. Not supported for IEDB.
            - "trb": Only keep TRB chain and discard TRA chain.
            - "separate": Keep both TRA and TRB chain as separate entries, ie. with a TCR chain dimension
                of length 2 in the input data tensor. This can be used to compute independent embeddings of
                both chains with RNN models.
            - "concat": Keep both TRA and TRB chain as a single entry, ie. with a TCR chain dimension
                of length 1 in the input data tensor. This concatenation allows the computation of a single
                embedding of both chains with a RNN model.
        :param blosum_encoding: Whether to use blosum_encoding.
        :param is_train: Whether to use this file as training data.
        :param all_trb: Whether all chains are TRB chains.
        :return:
        """
        assert chains.lower() != "tra", "Can only read TRB chains from IEDB."
        self.chains = chains

        # Read table from file.
        if isinstance(fns, str):
            fns = [fns]
        iedb_out = pd.concat([pd.read_csv(x).fillna(value="None") for x in fns])
        if all_trb:
            self.chains = "trb"
            iedb_out["temp_gene_id"] = "trb"
        else:
            assert False, "Mixed chains not yet supported for IEDB input."

        cdr3_seqs, pep_seqs = self._read_tcr_and_peptide_from_table(
            tab=iedb_out,
            cdr3_id="Chain 2 CDR3 Curated",
            peptide_id="Description",
            gene_id="temp_gene_id",
            obs_id=None,
            chains=self.chains,
            none_id="none"
        )
        if antigen_ids is not None:
            to_keep = [x[0] in antigen_ids for x in pep_seqs]
            print(
                "Found %i observations that match target antigen sequences out of %i." %
                (np.sum(to_keep), len(to_keep))
            )
            pep_seqs = [x for i, x in enumerate(pep_seqs) if to_keep[i]]
            cdr3_seqs = [x for i, x in enumerate(cdr3_seqs) if to_keep[i]]

        peps = self._format_data_aa(
            x=pep_seqs,
            fn_blosum=fn_blosum if blosum_encoding else None,
            tcr=False
        )
        cdr3s = self._format_data_aa(
            x=cdr3_seqs,
            fn_blosum=fn_blosum if blosum_encoding else None,
            tcr=True
        )

        # Assign class data container attributes.
        self.label_ids = ["bound"]
        self._assign_test_train(
            is_train=is_train,
            x=np.concatenate([cdr3s, peps], axis=2),
            covariates=np.empty([cdr3s.shape[0], 0]),
            y=np.ones([cdr3s.shape[0], 1]),
            pep_seqs=pep_seqs,
            cdr3_seqs=cdr3_seqs
        )
        print(
            "Found %i observations and assigned to " % len(peps) +
            ("train" if is_train else "test") + " data."
        )

    def read_vdjdb(
            self,
            fns: Union[list, str],
            fn_blosum: Union[str, None] = None,
            antigen_ids: Union[List[str], None] = None,
            chains: str = "trb",
            blosum_encoding: bool = False,
            is_train: bool = True
    ):
        """ Read VDJDB download files.

        :param fns: File names of positive observations file, ie. the iedb download.
        :param fn_blosum: File with BLOSUM50.txt embedding matrix. Only needed if blosum_encoding==True.
        :param antigen_ids: Target antigen sequences: only load observations that match these.
        :param chains: {"tra",  "trb", "separate", "concat"} Which TCR CDR chain(s) to keep.

            - "tra": Only keep TRA chain and discard TRB chain.
            - "trb": Only keep TRB chain and discard TRA chain.
            - "separate": Keep both TRA and TRB chain as separate entries, ie. with a TCR chain dimension
                of length 2 in the input data tensor. This can be used to compute independent embeddings of
                both chains with RNN models.
            - "concat": Keep both TRA and TRB chain as a single entry, ie. with a TCR chain dimension
                of length 1 in the input data tensor. This concatenation allows the computation of a single
                embedding of both chains with a RNN model.
        :param blosum_encoding: Whether to use blosum_encoding.
        :return:
        """
        self.chains = chains

        # Read table from file.
        if isinstance(fns, str):
            fns = [fns]
        vdjdb_out = pd.concat([pd.read_table(x).fillna(value="None") for x in fns])
        cdr3_seqs, pep_seqs = self._read_tcr_and_peptide_from_table(
            tab=vdjdb_out,
            cdr3_id="CDR3",
            peptide_id="Epitope",
            gene_id="Gene",
            obs_id="complex.id",
            chains=self.chains,
            tra_id="tra",
            trb_id="trb",
            none_id="none"
        )
        if antigen_ids is not None:
            to_keep = [x[0] in antigen_ids for x in pep_seqs]
            print(
                "Found %i observations that match target antigen sequences out of %i." %
                (np.sum(to_keep), len(to_keep))
            )
            pep_seqs = [x for i, x in enumerate(pep_seqs) if to_keep[i]]
            cdr3_seqs = [x for i, x in enumerate(cdr3_seqs) if to_keep[i]]

        # Transform amino acid encoding.
        peps = self._format_data_aa(
            x=pep_seqs,
            fn_blosum=fn_blosum if blosum_encoding else None,
            tcr=False
        )
        cdr3s = self._format_data_aa(
            x=cdr3_seqs,
            fn_blosum=fn_blosum if blosum_encoding else None,
            tcr=True
        )

        # Assign class data container attributes.
        self.label_ids = ["bound"]
        self._assign_test_train(
            is_train=is_train,
            x=np.concatenate([cdr3s, peps], axis=2),
            covariates=np.empty([cdr3s.shape[0], 0]),
            y=np.ones([cdr3s.shape[0], 1]),
            pep_seqs=pep_seqs,
            cdr3_seqs=cdr3_seqs
        )
        print(
            "Found %i observations and assigned to " % len(peps) +
            ("train" if is_train else "test") + " data."
        )

    def read_iedb_matched_to_categorical_model(
            self,
            fns: Union[list, str],
            fn_blosum: Union[str, None] = None,
            blosum_encoding: bool = False,
            is_train: bool = True,
            all_trb: bool = True,
            same_antigen: Union[bool, None] = None
    ):
        """ Read IEDB download files and extract observations with antigens that match categorical model.

        Adopts chains models from class attribute, ie assumes that another data set has been read and reads TCR
        chains with the same policies as this previous data set.

        Note: This cannot be used if covariates are used in the categorical models as these covariates are typically
        not available in VDJdb.
        Note: This function does not accommodate for multi-label models yet.

        :param fns: File names of positive observations file, ie. the iedb download.
        :param fn_blosum: File with BLOSUM50.txt embedding matrix. Only needed if blosum_encoding==True.
        :param blosum_encoding: Whether to use blosum_encoding.
        :param is_train: Whether to set data as train or test data.
        :param all_trb: Whether all chains are TRB chains.
        :param same_antigen: Whether to load TCRs of same antigen or of all other antigens. Set as None to load all.
        :return:
        """
        # Read table from file.
        if isinstance(fns, str):
            fns = [fns]
        iedb_out = pd.concat([pd.read_csv(x).fillna(value="None") for x in fns])
        if all_trb:
            iedb_out["temp_gene_id"] = "trb"
        else:
            assert False, "Mixed chains not yet supported for IEDB input."

        self._read_table_matched_to_categorical_model(
            tab=iedb_out,
            fn_blosum=fn_blosum,
            blosum_encoding=blosum_encoding,
            is_train=is_train,
            same_antigen=same_antigen,
            add_non_binder_for_softmax=self.add_non_binder_for_softmax,
            cdr3_id="Chain 2 CDR3 Curated",
            peptide_id="Description",
            gene_id="temp_gene_id",
            obs_id=None,
            tra_id="tra",
            trb_id="trb",
            none_id="none"
        )

    def read_vdjdb_matched_to_categorical_model(
            self,
            fns: Union[list, str],
            fn_blosum: Union[str, None] = None,
            blosum_encoding: bool = False,
            is_train: bool = True,
            same_antigen: Union[bool, None] = None
    ):
        """ Read VDJDB download files and extract observations with antigens that match categorical model.

        Adopts chains models from class attribute, ie assumes that another data set has been read and reads TCR
        chains with the same policies as this previous data set.

        Note: This cannot be used if covariates are used in the categorical models as these covariates are typically
        not available in VDJdb.
        Note: This function does not accommodate for multi-label models yet.

        :param fns: File names of positive observations file, ie. the iedb download.
        :param fn_blosum: File with BLOSUM50.txt embedding matrix. Only needed if blosum_encoding==True.
        :param blosum_encoding: Whether to use blosum_encoding.
        :param is_train: Whether to set data as train or test data.
        :param same_antigen: Whether to load TCRs of same antigen or of all other antigens. Set as None to load all.
        :return:
        """
        # Read table from file.
        if isinstance(fns, str):
            fns = [fns]
        vdjdb_out = pd.concat([pd.read_table(x).fillna(value="None") for x in fns])

        self._read_table_matched_to_categorical_model(
            tab=vdjdb_out,
            fn_blosum=fn_blosum,
            blosum_encoding=blosum_encoding,
            is_train=is_train,
            same_antigen=same_antigen,
            add_non_binder_for_softmax=self.add_non_binder_for_softmax,
            cdr3_id="CDR3",
            peptide_id="Epitope",
            gene_id="Gene",
            obs_id="complex.id",
            tra_id="tra",
            trb_id="trb",
            none_id="none"
        )

    def read_iedb_as_categorical_model(
            self,
            fns: Union[list, str],
            antigen_ids: List[str],
            fn_blosum: Union[str, None] = None,
            blosum_encoding: bool = False,
            is_train: bool = True,
            chains: str = "trb",
            all_trb: bool = True,
            add_non_binder_for_softmax: bool = True
    ):
        """ Read IEDB download files and extract observations with antigens that match categorical model.

        Adopts chains models from class attribute, ie assumes that another data set has been read and reads TCR
        chains with the same policies as this previous data set.

        Note: This cannot be used if covariates are used in the categorical models as these covariates are typically
        not available in VDJdb.
        Note: This function does not accommodate for multi-label models yet.

        :param fns: File names of positive observations file, ie. the iedb download.
        :param antigen_ids: Antigens to include as output labels.
        :param fn_blosum: File with BLOSUM50.txt embedding matrix. Only needed if blosum_encoding==True.
        :param blosum_encoding: Whether to use blosum_encoding.
        :param is_train: Whether to set data as train or test data.
        :param chains: {"tra",  "trb", "separate", "concat"} Which TCR CDR chain(s) to keep.

            - "tra": Only keep TRA chain and discard TRB chain. Not supported for IEDB.
            - "trb": Only keep TRB chain and discard TRA chain.
            - "separate": Keep both TRA and TRB chain as separate entries, ie. with a TCR chain dimension
                of length 2 in the input data tensor. This can be used to compute independent embeddings of
                both chains with RNN models.
            - "concat": Keep both TRA and TRB chain as a single entry, ie. with a TCR chain dimension
                of length 1 in the input data tensor. This concatenation allows the computation of a single
                embedding of both chains with a RNN model.
        :param all_trb: Whether all chains are TRB chains.
        :param add_non_binder_for_softmax: Whether to add an additional non-binder category
            for softmax activation function. This category is set to 1 if all other categories are 0.
        :return:
        """
        self.chains = chains
        self.add_non_binder_for_softmax = add_non_binder_for_softmax
        # Read table from file.
        if isinstance(fns, str):
            fns = [fns]
        iedb_out = pd.concat([pd.read_csv(x).fillna(value="None") for x in fns])
        if all_trb:
            iedb_out["temp_gene_id"] = "trb"
        else:
            assert False, "Mixed chains not yet supported for IEDB input."

        self.label_ids = antigen_ids
        self._read_table_as_categorical_model(
            tab=iedb_out,
            fn_blosum=fn_blosum,
            blosum_encoding=blosum_encoding,
            is_train=is_train,
            antigen_ids=antigen_ids,
            add_non_binder_for_softmax=add_non_binder_for_softmax,
            cdr3_id="Chain 2 CDR3 Curated",
            peptide_id="Description",
            gene_id="temp_gene_id",
            obs_id=None,
            tra_id="tra",
            trb_id="trb",
            none_id="none"
        )

    def read_vdjdb_as_categorical_model(
            self,
            fns: Union[list, str],
            antigen_ids: List[str],
            fn_blosum: Union[str, None] = None,
            blosum_encoding: bool = False,
            is_train: bool = True,
            chains: str = "trb",
            add_non_binder_for_softmax: bool = True
    ):
        """ Read VDJDB download files and extract observations with antigens that match categorical model.

        Adopts chains models from class attribute, ie assumes that another data set has been read and reads TCR
        chains with the same policies as this previous data set.

        Note: This cannot be used if covariates are used in the categorical models as these covariates are typically
        not available in VDJdb.
        Note: This function does not accommodate for multi-label models yet.

        :param fns: File names of positive observations file, ie. the iedb download.
        :param antigen_ids: Antigens to include as output labels.
        :param fn_blosum: File with BLOSUM50.txt embedding matrix. Only needed if blosum_encoding==True.
        :param blosum_encoding: Whether to use blosum_encoding.
        :param is_train: Whether to set data as train or test data.
        :param chains: {"tra",  "trb", "separate", "concat"} Which TCR CDR chain(s) to keep.

            - "tra": Only keep TRA chain and discard TRB chain. Not supported for IEDB.
            - "trb": Only keep TRB chain and discard TRA chain.
            - "separate": Keep both TRA and TRB chain as separate entries, ie. with a TCR chain dimension
                of length 2 in the input data tensor. This can be used to compute independent embeddings of
                both chains with RNN models.
            - "concat": Keep both TRA and TRB chain as a single entry, ie. with a TCR chain dimension
                of length 1 in the input data tensor. This concatenation allows the computation of a single
                embedding of both chains with a RNN model.
        :param add_non_binder_for_softmax: Whether to add an additional non-binder category
            for softmax activation function. This category is set to 1 if all other categories are 0.
        :return:
        """
        self.chains = chains
        self.add_non_binder_for_softmax = add_non_binder_for_softmax
        # Read table from file.
        if isinstance(fns, str):
            fns = [fns]
        vdjdb_out = pd.concat([pd.read_table(x).fillna(value="None") for x in fns])

        self.label_ids = antigen_ids
        self._read_table_as_categorical_model(
            tab=vdjdb_out,
            fn_blosum=fn_blosum,
            blosum_encoding=blosum_encoding,
            is_train=is_train,
            antigen_ids=antigen_ids,
            add_non_binder_for_softmax=add_non_binder_for_softmax,
            cdr3_id="CDR3",
            peptide_id="Epitope",
            gene_id="Gene",
            obs_id="complex.id",
            tra_id="tra",
            trb_id="trb",
            none_id="none"
        )

    def _read_table_matched_to_categorical_model(
            self,
            tab: pd.DataFrame,
            fn_blosum: Union[str, None],
            blosum_encoding: bool,
            is_train: bool,
            same_antigen: Union[bool, None],
            add_non_binder_for_softmax: bool,
            cdr3_id: str,
            peptide_id: str,
            gene_id: Union[None, str],
            obs_id: Union[None, str],
            tra_id: str = "tra",
            trb_id: str = "trb",
            none_id: str = "none"
    ):
        """ Read table download files and extract observations with antigens that match categorical model.

        Adpots chains models from class attribute, ie assumes that another data set has been read and reads TCR
        chains with the same policies as this previous data set.

        Note: This cannot be used if covariates are used in the categorical models as these covariates are typically
        not available in VDJdb.
        Note: This function does not accommodate for multi-label models yet.

        :param fns: File names of positive observations file, ie. the iedb download.
        :param fn_blosum: File with BLOSUM50.txt embedding matrix. Only needed if blosum_encoding==True.
        :param blosum_encoding: Whether to use blosum_encoding.
        :param is_train: Whether to set data as train or test data.
        :param same_antigen: Whether to load TCRs of same antigen or of all other antigens. Set as None to load all.
        :return:
        """
        cdr3_seqs, pep_seqs = self._read_tcr_and_peptide_from_table(
            tab=tab,
            cdr3_id=cdr3_id,
            peptide_id=peptide_id,
            gene_id=gene_id,
            obs_id=obs_id,
            chains=self.chains,
            tra_id=tra_id,
            trb_id=trb_id,
            none_id=none_id
        )

        if add_non_binder_for_softmax:
            y = np.zeros([len(pep_seqs), len(self.label_ids) + 1])
        else:
            y = np.zeros([len(pep_seqs), len(self.label_ids)])
        for i, x in enumerate(pep_seqs):
            if len(self.label_ids) > 1:
                if x[0] in self.label_ids:
                    y[i, self.label_ids.index(x[0])] = 1
                else:
                    if add_non_binder_for_softmax:
                        y[i, -1] = 1
            else:
                if x[0] in self.label_ids:
                    y[i, 0] = 1

        if same_antigen is None:
            # Keep all observations
            idx_to_keep = np.arange(0, len(pep_seqs))
        else:
            if same_antigen:
                # Only keep observations that match peptide sequences that were utilized in categorical model.
                idx_to_keep = [i for i, x in enumerate(pep_seqs) if x[0] in self.label_ids]
            else:
                # Only keep observations that do not match peptide sequences that were utilized in categorical model.
                idx_to_keep = [i for i, x in enumerate(pep_seqs) if x[0] not in self.label_ids]
        cdr3_seqs = [x for i, x in enumerate(cdr3_seqs) if i in idx_to_keep]
        y = y[idx_to_keep, :]
        if same_antigen is None:
            print(
                "Found " + str(len(set(self.label_ids).intersection(set([x[0] for x in pep_seqs])))) +
                " antigen observations that match" +
                " and " + str(len(set([x[0] for x in pep_seqs]) - set(self.label_ids))) +
                " that do not match antigens in categorical index (" +
                ", ".join(self.label_ids) + ") in query."
            )
        else:
            print(
                "Found " + str(len(cdr3_seqs)) + " antigen observations that " +
                ("match" if same_antigen else "do not match") +
                " antigens in categorical index (" + ", ".join(self.label_ids) +
                ") in query."
            )

        # Transform amino acid encoding.
        cdr3s = self._format_data_aa(
            x=cdr3_seqs,
            fn_blosum=fn_blosum if blosum_encoding else None,
            tcr=True
        )

        # Format covariates: all are set to zero.
        covariates = np.zeros([cdr3s.shape[0], self.covariates_train.shape[1]])

        # Assign class data container attributes.
        self._assign_test_train(
            is_train=is_train,
            x=cdr3s,
            covariates=covariates,
            y=y,
            pep_seqs=pep_seqs,
            cdr3_seqs=cdr3_seqs
        )

    def _read_table_as_categorical_model(
            self,
            tab: pd.DataFrame,
            fn_blosum: Union[str, None],
            blosum_encoding: bool,
            is_train: bool,
            antigen_ids: List[str],
            add_non_binder_for_softmax: bool,
            cdr3_id: str,
            peptide_id: str,
            gene_id: Union[None, str],
            obs_id: Union[None, str],
            tra_id: str = "tra",
            trb_id: str = "trb",
            none_id: str = "none"
    ):
        """ Read table download files as categorical model.

        Builds a multi-label classifier for the antigens listed in antigen_ids. The remaining observations
        are treated as negatives.

        :param fns: File names of positive observations file, ie. the iedb download.
        :param fn_blosum: File with BLOSUM50.txt embedding matrix. Only needed if blosum_encoding==True.
        :param blosum_encoding: Whether to use blosum_encoding.
        :param is_train: Whether to set data as train or test data.
        :param antigen_ids: Antigens to include as output labels.
        :return:
        """
        cdr3_seqs, pep_seqs = self._read_tcr_and_peptide_from_table(
            tab=tab,
            cdr3_id=cdr3_id,
            peptide_id=peptide_id,
            gene_id=gene_id,
            obs_id=obs_id,
            chains=self.chains,
            tra_id=tra_id,
            trb_id=trb_id,
            none_id=none_id
        )

        if add_non_binder_for_softmax:
            y = np.zeros([len(pep_seqs), len(antigen_ids) + 1])
        else:
            y = np.zeros([len(pep_seqs), len(antigen_ids)])
        for i, x in enumerate(pep_seqs):
            if x[0] in antigen_ids:
                y[i, antigen_ids.index(x[0])] = 1
            else:
                if add_non_binder_for_softmax:
                    y[i, -1] = 1

        # Transform amino acid encoding.
        cdr3s = self._format_data_aa(
            x=cdr3_seqs,
            fn_blosum=fn_blosum if blosum_encoding else None,
            tcr=True
        )

        # Format covariates: all are set to zero.
        if self.covariates_train:
            covariates = np.zeros([cdr3s.shape[0], self.covariates_train.shape[1]])
        else:
            covariates = np.empty([cdr3s.shape[0], 0])

        # Assign class data container attributes.
        self._assign_test_train(
            is_train=is_train,
            x=cdr3s,
            covariates=covariates,
            y=y,
            pep_seqs=pep_seqs,
            cdr3_seqs=cdr3_seqs
        )

    def _read_tcr_and_peptide_from_table(
            self,
            tab: pd.DataFrame,
            cdr3_id: str,
            peptide_id: str,
            chains: str,
            gene_id: Union[None, str] = None,
            obs_id: Union[None, str] = None,
            tra_id: str = "tra",
            trb_id: str = "trb",
            none_id: str = "none"
    ):
        # Subset table by rows that have an observed peptide and TCR chain.
        tab = tab.iloc[np.where([x.lower() != none_id for x in tab[peptide_id].values])[0], :]
        tab = tab.iloc[np.where([x.lower() != none_id for x in tab[cdr3_id].values])[0], :]
        # Subset to TRA or TRB only if selected.
        if chains.lower() == "tra":
            if gene_id is not None:
                tab = tab.iloc[np.where([x.lower() == tra_id for x in tab[gene_id].values])[0], :]
            else:
                gene_id = "temp_gene_col"
                tra_id = "tra"
                tab["temp_gene_col"] = tra_id
        if chains.lower() == "trb":
            if gene_id is not None:
                tab = tab.iloc[np.where([x.lower() == trb_id for x in tab[gene_id].values])[0], :]
            else:
                gene_id = "temp_gene_col"
                trb_id = "trb"
                tab["temp_gene_col"] = trb_id
        # Subset to rows containing antigen and TCR sequences only with known amino acids.
        cdr3_aa_known = np.array([
            np.all([aa in self.aa_list for aa in x])
            for x in tab[cdr3_id].values
        ])
        pep_aa_known = np.array([
            np.all([aa in self.aa_list for aa in x])
            for x in tab[peptide_id].values
        ])
        to_keep = np.logical_and(cdr3_aa_known, pep_aa_known)
        print("Found %i CDR3 observations with unkown amino acids out of %i." %
              (np.sum(np.logical_not(cdr3_aa_known)), len(cdr3_aa_known)))
        print("Found %i antigen observations with unkown amino acids out of %i." %
              (np.sum(np.logical_not(pep_aa_known)), len(pep_aa_known)))
        print("Found %i CDR3+antigen observations with unkown amino acids out of %i, leaving %i observations." %
              (np.sum(np.logical_not(to_keep)), len(to_keep), np.sum(to_keep)))
        tab = tab.iloc[np.where(to_keep)[0], :]

        if obs_id is None:
            obs_id = "temp_id_col"
            tab[obs_id] = np.arange(0, tab.shape[0])
        observation_set = np.unique(tab[obs_id].values).tolist()
        cdr3s = [[None, None] for i in observation_set]
        peps = [[None] for i in observation_set]
        for i in range(tab.shape[0]):
            if tab[gene_id].values[i].lower() == tra_id:
                cdr3s[observation_set.index(tab[obs_id].values[i])][0] = tab[cdr3_id].values[i]
            if tab[gene_id].values[i].lower() == trb_id:
                cdr3s[observation_set.index(tab[obs_id].values[i])][1] = tab[cdr3_id].values[i]
            peps[observation_set.index(tab[obs_id].values[i])][0] = tab[peptide_id].values[i]
        print("Assembled %i single-chain observations into %i multiple chain observations." %
              (tab.shape[0], len(observation_set)))

        return cdr3s, peps

    def _read_tcr_from_csv_table(
        self,
        cell_table: pd.DataFrame,
        fn_blosum: Union[str, None],
        blosum_encoding: bool,
        is_train: bool,
        obs_id: str,
        tra_id: str,
        trb_id: str,
        covariate_formula_categ:list = [],
        covariate_formula_numeric: list = [],
        rename_covariates_for_patsy: bool = True
    ):
        """ Read and extract tcr from dataframe tables.

        :param fns: File names of observations files
        :param fn_blosum: File with BLOSUM50.txt embedding matrix. Only needed if blosum_encoding==True.
        :param blosum_encoding: Whether to use blosum_encoding.
        :param is_train: Whether to set data as train or test data.
        :param obs_id: column name of observation id in file
        :param tra_id: column name of TCR chain alpha in file
        :param trb_id: column name of TCR chain beta in file
        :param covariate_formula_categ: Terms for patsy formula to build categorical covariate matrix from based on input table.
            Leave out "~0+" at the start of the formula, this is added automatically. Note that it does not make sense
            to include an intercept as this is already present in the network in the form of a bias.
            Numeric values in the table are automatically transformed to strings.
        :param covariate_formula_numeric: Terms for patsy formula to build numeric covariate matrix from based on input table.
            Leave out "~0+" at the start of the formula, this is added automatically. Note that it does not make sense
            to include an intercept as this is already present in the network in the form of a bias.
            Categorical values in the table are automatically transformed to numeric.
        :param rename_covariates_for_patsy: Whether to automatically change predictor naming to comply with patsy.
            In particular, column names that include a "~", "+" or a "-" are stripped of these characters.
        """
        # Format TCR sequences by cell:
        # Extract sequences from cell table.

        tras = cell_table[tra_id].values[..., np.newaxis].tolist()
        trbs = cell_table[trb_id].values[..., np.newaxis].tolist()
        # Replace empty lists by None:
        tras = [x if len(x) > 0 else [None] for x in tras]
        trbs = [x if len(x) > 0 else [None] for x in trbs]
        # Assemble two dimensional list.
        cdr3_seqs = [[tras[i][0], trbs[i][0]] for i, _ in enumerate(tras)]

        cdr3s = self._format_data_aa(
            x=cdr3_seqs,
            fn_blosum=fn_blosum if blosum_encoding else None,
            tcr=True
        )

        covariates_table = pd.DataFrame(index=cell_table[obs_id])
        for x in covariate_formula_categ:
            covariates_table[x] = cell_table[x].values
        covariates = self._format_covariates_from_table(
            formula_categ=covariate_formula_categ,
            formula_numeric=covariate_formula_numeric,
            rename_covariates_for_patsy=rename_covariates_for_patsy,
            table=covariates_table
        )
        self._assign_test_train(
            is_train=is_train,
            x=cdr3s,
            covariates=covariates,
            y=cdr3s,
            cdr3_seqs=cdr3_seqs
        )

    def read_tcr_from_csv_files(
        self,
        fns: Union[list, str],
        fn_blosum: Union[str, None],
        blosum_encoding: bool,
        is_train: bool,
        obs_id: Union[None, str] = None,
        tra_id: Union[None, str] = None,
        trb_id: Union[None, str] = None,
        covariates:dict = {},
        covariate_formula_categ: list = [],
        covariate_formula_numeric: list = [],
        rename_covariates_for_patsy: bool = True
    ):
        """ Read and extract tcr from csv files.

        :param fns: File names of observations files
        :param fn_blosum: File with BLOSUM50.txt embedding matrix. Only needed if blosum_encoding==True.
        :param blosum_encoding: Whether to use blosum_encoding.
        :param is_train: Whether to set data as train or test data.
        :param obs_id: column name of observation id in file
        :param tra_id: column name of TCR chain alpha in file
        :param trb_id: column name of TCR chain beta in file
        :param covariates Nested dictionary with each child dict corresponding to the covariates of a file name in parent dict
        :param covariate_formula_categ: Terms for patsy formula to build categorical covariate matrix from based on input table.
            Leave out "~0+" at the start of the formula, this is added automatically. Note that it does not make sense
            to include an intercept as this is already present in the network in the form of a bias.
            Numeric values in the table are automatically transformed to strings.
        :param covariate_formula_numeric: Terms for patsy formula to build numeric covariate matrix from based on input table.
            Leave out "~0+" at the start of the formula, this is added automatically. Note that it does not make sense
            to include an intercept as this is already present in the network in the form of a bias.
            Categorical values in the table are automatically transformed to numeric.
        :param rename_covariates_for_patsy: Whether to automatically change predictor naming to comply with patsy.
            In particular, column names that include a "~", "+" or a "-" are stripped of these characters.
        """
        self.chains = 'concat'  # is it necessary to make this an input variable?
        if isinstance(fns, str):  # if only one filename string is passed, make it a list
            fns = [fns]

        # checks the first file's name and sets default column names if it's a bladder tumor csv file
        if "GSM45066" in fns[0]:
            obs_id = "cell.barcode" if obs_id is None else obs_id
            tra_id = "alphaCDR3" if tra_id is None else tra_id
            trb_id = "betaCDR3" if trb_id is None else trb_id

        # check if all column names have been provided, send an error otherwise
        if tra_id is None:
            print("Please provide tra_id: column name of TCR chain alpha in the file")
            return

        if trb_id is None:
            print("Please provide trb_id: column name of TCR chain beta in file")
            return

        if obs_id is None:
            print("Please provide obs_id: column name of observation id in file")
            return

        # make sure all the covariates for all files have been passed, send an error otherwise
        for fn in fns:
            file_name = os.path.basename(fn)
            if file_name in covariates:
                if len(covariates[file_name].keys()) != len(covariate_formula_categ):
                    print("Please provide all covariate categories for "+file_name)
                    return
            else:
                print("Please provide covariates for " + file_name)
                return
        ###############################################################################################

        # for each file, each covariate-value pair from it's corresponding covariate object is added to it's dataframe
        # and then all dataframes are concatenated
        all_fns_table = pd.concat([pd.read_csv(fn).assign(**covariates[os.path.basename(fn)]) for fn in fns])

        self._read_tcr_from_csv_table(
            cell_table=all_fns_table,
            fn_blosum=fn_blosum,
            blosum_encoding= blosum_encoding,
            is_train=is_train,
            obs_id=obs_id,
            tra_id=tra_id,
            trb_id=trb_id,
            covariate_formula_categ=covariate_formula_categ,
            covariate_formula_numeric=covariate_formula_numeric,
            rename_covariates_for_patsy=rename_covariates_for_patsy
        )

    def sample_negative_data(
            self,
            is_train: bool = True
    ):
        """ Create one negative antigen-TCR pair for each TCR.

        :param is_train: Whether partition to sample negative samples from and for is train or test data.
        :return:
        """
        if is_train:
            tra_seq = self.tra_seqs_train
            trb_seq = self.trb_seqs_train
            antigen_seq = self.peptide_seqs_train
            clones = self.clone_train
            nc = self.nc_train
            x_embedded = self.x_train
        else:
            tra_seq = self.tra_seqs_test
            trb_seq = self.trb_seqs_test
            antigen_seq = self.peptide_seqs_test
            clones = self.clone_test
            nc = self.nc_test
            x_embedded = self.x_test

        assert np.all([x[0] is not None for x in antigen_seq]), \
            "entries of antigen_seq cannot be None here: %s" % antigen_seq

        # Build (trb,list(antigens)) table for sampling negative data, the antigens in this list could bind with trb.
        binding_table = {}
        for i, w in enumerate(trb_seq):
            if w not in binding_table:
                binding_table[w] = []
            binding_table[w].append(antigen_seq[i][0])

        # Generate a negative sample for each element in TCR list.
        antigen_seq_flat = [x[0] for x in antigen_seq]
        antigen_set = set(antigen_seq_flat)
        # Find first occurrence of each antigen in corresponding data to use indexing to carry over embedded sequences.
        antigen_first_idx = dict(zip(
            list(antigen_set),
            [next(i for i, xx in enumerate(antigen_seq_flat) if xx == x) for x in list(antigen_set)]
        ))
        trb_idx_neg = []
        antigen_idx_neg = []
        for i, w in enumerate(trb_seq):
            # Draw an antigen that was not observed to bind to the given TRB.
            antigens_not_bound = list(antigen_set - set(binding_table[w]))
            # Only append synthetic negative sample if TCR was not seen in combination to at least one antigen.
            if len(antigens_not_bound) > 0:
                antigen_idx_neg.append(antigen_first_idx[random.sample(population=antigens_not_bound, k=1)[0]])
                trb_idx_neg.append(i)

        # Append synthetic negative samples to real samples based on permutation defined above.
        self._assign_test_train(
            is_train=is_train,
            x=np.concatenate([
                x_embedded[trb_idx_neg, :, :self.tcr_len, :],
                x_embedded[antigen_idx_neg, :, self.tcr_len:, :]
            ], axis=2),
            covariates=np.empty([len(trb_idx_neg), 0]),
            clonotype=clones[trb_idx_neg] if clones is not None else None,
            nc=nc[trb_idx_neg] if nc is not None else None,
            y=np.zeros([len(trb_idx_neg), 1]),
            pep_seqs=[[antigen_seq[i]] for i in antigen_idx_neg],
            cdr3_seqs=[[tra_seq[i], trb_seq[i]] for i in trb_idx_neg]
        )
        print(
            "Generated %i negative samples in %s data, yielding %i total observations." %
            (len(trb_idx_neg),
             "train" if is_train else "test",
             self.x_train.shape[0] if is_train else self.x_test.shape[0])
        )

    def read_consensus_annotation(
            self,
            fn: str,
            y=None,
            is_train: bool = True,
            is_test: bool = False,
            version: str = "v3_0"
    ):
        """ TODO not working?

        :param fn:
        :param y:
        :param is_train:
        :param is_test:
        :param version:
        :return:
        """
        if version == "v3_0":
            x_aa = self._read_consensus_annotation_v3(fn=fn)
        else:
            raise ValueError('version %s not recognized' % version)

        x = self._format_data_aa(x=x_aa)
        covariates = np.zeros([self.x.shape[0], 1])
        y = y if y is not None else np.zeros([self.x.shape[0], 1])

        if is_train:
            self.x_train = x
            self.covariates_train = covariates
            self.y_train = y

        if is_test:
            self.x_test = x
            self.covariates_test = covariates
            self.y_test = y

    def _read_consensus_annotation_v3(
            self,
            fn: str
    ) -> List:
        """ TODO not working?

        Note that CDR3s do NOT have to have the same length.
        TODO this does not group chains by cell yet.

        :param fn: cell ranger "*_consensus_annotation.csv" output file.
        :return: List of strings with TCR CDR3 sequence by cell and chain [observations, chains]
        """
        cellranger_out = pd.read_csv(fn)
        cellranger_out = cellranger_out.iloc[np.where([x != "None" for x in cellranger_out["cdr3"].values])[0], :]
        cdr3s = [[x] for x in cellranger_out["cdr3"].values]
        # Extract functions for conditional input

        return cdr3s

    def read_binarized_matrix(
            self,
            fns: Union[List[str], str],
            label_cols: Union[List[str], str],
            fns_clonotype: Union[List[str], str, None] = None,
            fn_blosum: Union[str, None] = None,
            blosum_encoding: bool = False,
            nc_cols: Union[str, list, None] = None,
            add_nc_to_covariates: bool = False,
            chains: str = "trb",
            id_col: str = "barcode",
            covariate_formula_categ: list = [],
            covariate_formula_numeric: list = [],
            add_non_binder_for_softmax: bool = False,
            rename_covariates_for_patsy: bool = True,
            discard_doublets: bool = True,
            fns_covar: Union[List[str], str] = [],
            sparse: bool = False,
            id_cdr3: str = 'cell_clono_cdr3_aa',
            is_train: bool = True,
            sep: str = ",",
            version: str = "v3_0"
    ):
        """ Read cellranger *binarized_matrix.csv files.

        Only use covariate_formula on an entire dataset and subset into test and training afterwards,
        otherwise one-hot encoding of categorical predictors may differ between training and test data set.

        :param fns: cell ranger "*_binarized_matrix.csv" output files.
        :param label_cols: List of column names in fn of lables to predict.
        :param fns_clonotype: cell ranger "*_clonotypes.csv" output files. Important: clonotypes are processed
            based on the assumption that clonotypes are exclusive to a file, ie are not shared across donors if
            one file corersponds to a donor.
        :param fn_blosum: File with BLOSUM50.txt embedding matrix. Only needed if blosum_encoding==True.
        :param blosum_encoding: Whether to use blosum_encoding.
        :param nc_cols: List of column names in fn of negative controls.
        :param add_nc_to_covariates: Whether to add log transformed negative control counts as indicated in nc_cols
            to covariates of model.
        :param chains: {"tra",  "trb", "separate", "concat"} Which TCR CDR chain(s) to keep.

            - "tra": Only keep TRA chain and discard TRB chain.
            - "trb": Only keep TRB chain and discard TRA chain.
            - "separate": Keep both TRA and TRB chain as separate entries, ie. with a TCR chain dimension
                of length 2 in the input data tensor. This can be used to compute independent embeddings of
                both chains with RNN models.
            - "concat": Keep both TRA and TRB chain as a single entry, ie. with a TCR chain dimension
                of length 1 in the input data tensor. This concatenation allows the computation of a single
                embedding of both chains with a RNN model.
        :param id_col: Name of column which observation identifier.
        :param covariate_formula_categ: Terms for patsy formula to build categorical covariate matrix from based on input table.
            Leave out "~0+" at the start of the formula, this is added automatically. Note that it does not make sense
            to include an intercept as this is already present in the network in the form of a bias.
            Numeric values in the table are automatically transformed to strings.
        :param covariate_formula_numeric: Terms for patsy formula to build numeric covariate matrix from based on input table.
            Leave out "~0+" at the start of the formula, this is added automatically. Note that it does not make sense
            to include an intercept as this is already present in the network in the form of a bias.
            Categorical values in the table are automatically transformed to numeric.
        :param add_non_binder_for_softmax: Whether to add an additional non-binder category
            for softmax activation function. This category is set to 1 if all other categories are 0.
        :param rename_covariates_for_patsy: Whether to automatically change predictor naming to comply with patsy.
            In particular, column names that include a "~", "+" or a "-" are stripped of these characters.
        :param discard_doublets: Whether to discard cells with multiple observations for a single chain.
            Only keeps one chain at random if False.
        :param fns_covar: Tabular files that contain additional covariates, ie ones that are not included in fns.
            The list of files has to be matched in length and sequence to fns.
            The rows in each fns file have to occur in the correspond fns_covar file. To ensure this, fns_covar
            has to contain a column named id_col, which is compared against the corresponding column in fns.
            Has to have same separator as fns.
        :param sparse:
        :param id_cdr3:
        :param is_train: Whether to use this file as training data or for test data.
        :param sep:
        :param version:
        :return:
        """
        self.chains = chains
        self.add_non_binder_for_softmax = add_non_binder_for_softmax

        if isinstance(label_cols, str):
            label_cols = [label_cols]
        if isinstance(fns, str):
            fns = [fns]
        if isinstance(fns_clonotype, str):
            fns_clonotype = [fns_clonotype]
        if isinstance(fns_covar, str):
            fns_covar = [fns_covar]

        # Check that clonotypes file list matches length of cell observation files:
        if fns_clonotype is not None:
            if len(fns_clonotype) != len(fns):
                raise ValueError("length of fns_clonotype is not matched to length of fns, "
                                 "these must be corresponding")
        else:
            fns_clonotype = [None for x in fns]

        if version == "v3_0":
            antigen_idx = 1  # Index of antigen peptide sequence in column name split by "_".
            inputs = [self._read_binarized_matrix_v3_0(
                fn=fn,
                lable_cols=label_cols,
                fns_clonotype=fns_clonotype[i],
                nc_cols=nc_cols,
                id_col=id_col,
                discard_doublets=discard_doublets,
                sparse=sparse,
                id_cdr3=id_cdr3,
                sep=sep,
            ) for i, fn in enumerate(fns)]
        else:
            raise ValueError('version %s not recognized' % version)

        cdr3_seqs = []
        for x in inputs:
            cdr3_seqs = cdr3_seqs + x[0]
        cdr3s = self._format_data_aa(
            x=cdr3_seqs,
            fn_blosum=fn_blosum if blosum_encoding else None,
            tcr=True
        )
        covariates_tables = [x[1] for x in inputs]
        # Add covariates from additional files:
        if len(fns_covar) > 0:
            if len(fns_covar) != len(fns):
                raise ValueError(
                    "The arguments fns (%i) and fns_covar (%i) must have the same length." %
                    (len(fns), len(fns_covar))
                )
            for i, fi in enumerate(fns_covar):
                temp = pd.read_csv(fi, sep=sep, header=0)
                temp.set_index(id_col, inplace=True)
                covariates_to_add = [y for y in temp.columns if y != id_col]
                for y in covariates_to_add:
                    covariates_tables[i][y] = temp.loc[covariates_tables[i][id_col].values, :][y].values

        covariates_table = pd.concat(covariates_tables, axis=0)
        covariates = self._format_covariates_from_table(
            formula_categ=covariate_formula_categ,
            formula_numeric=covariate_formula_numeric,
            rename_covariates_for_patsy=rename_covariates_for_patsy,
            table=covariates_table
        )
        y = np.concatenate([x[2] for x in inputs], axis=0)
        if add_non_binder_for_softmax:
            non_binder_category = np.expand_dims(
                np.asarray(np.asarray(np.sum(y, axis=1) == 0, dtype=int), dtype=float),
                axis=1
            )
            y = np.concatenate([y, non_binder_category], axis=1)
            if not np.all(np.sum(y, axis=1) == 1):
                raise ValueError("add_non_binder_for_softmax was used but input labels did not add up to 0 or 1.")
        nc = np.concatenate([x[3] for x in inputs], axis=0)
        if add_nc_to_covariates:
            nc_as_covar = np.log(nc + 1)
            covariates = np.concatenate([covariates, nc_as_covar], axis=1)

        # Make clonotypes unique across donors:
        clonotype_assign = [x[4] for x in inputs]
        for i in range(len(clonotype_assign)):
            if i > 0:
                clonotype_assign[i] = clonotype_assign[i] + np.max(clonotype_assign[i-1])
        clonotype_assign = np.concatenate(clonotype_assign, axis=0)
        print(
            "Found %i clonotypes for %i observations and assigned to " %
            (len(np.unique(clonotype_assign)), len(clonotype_assign)) +
            ("train" if is_train else "test") + " data."
        )

        # Assign class data container attributes.
        self.label_cols = label_cols
        self.label_ids = [x.split("_")[antigen_idx] for x in label_cols]
        self._assign_test_train(
            is_train=is_train,
            x=cdr3s,
            covariates=covariates,
            y=y,
            nc=nc,
            clonotype=clonotype_assign,
            cdr3_seqs=cdr3_seqs
        )

    def _read_binarized_matrix_v3_0(
            self,
            fn: str,
            lable_cols: Union[str, list],
            fns_clonotype: Union[List[str], str, None] = None,
            nc_cols: Union[str, list, None] = None,
            id_col: str = "barcode",
            discard_doublets: bool = True,
            sparse: bool = False,
            id_cdr3: str = 'cell_clono_cdr3_aa',
            sep: str = ","
    ) -> Tuple:
        """

        Note that CDR3s do NOT have to have the same length.

        :param fn: cell ranger "*_binarized_matrix.csv" output file.
        :param lable_cols: List of column names in fn of lables to predict.
        :param fns_clonotype: cell ranger "*_clonotypes.csv" output files.
        :param nc_cols: List of column names in fn of negative controls.
        :param id_col: Name of column which observation identifier.
        :param discard_doublets: Whether to discard cells with multiple observations for a single chain.
            Only keeps one chain at random if False.
        :return: Tuple

            - covariates: pd.DataFrame with covariates including TCR CDR3 sequence by cell and chain [observations, covariates]
            - y: np.ndarray of numeric lables of cells [observations, lable dimension]
            - nc: np.ndarray negative control table
        """
        # Read files.
        cell_table = pd.read_csv(fn, sep=sep, header=0)
        covariates_table = pd.DataFrame(index=cell_table[id_col])
        for x in cell_table.columns:
            covariates_table[x] = cell_table[x].values
        if fns_clonotype is not None:
            clonotype_table = pd.read_csv(fn, sep=",", header=0)
        else:
            clonotype_table = None

        # Format TCR sequences by cell:
        # Extract sequences from cell table.
        tcr_cell_raw = cell_table[id_cdr3].values
        tcr_seqs = [[y.split(':') for y in x.split(';')] for x in tcr_cell_raw]
        tras = [[y[1] for y in x if y[0] == 'TRA'] for x in tcr_seqs]
        trbs = [[y[1] for y in x if y[0] == 'TRB'] for x in tcr_seqs]
        n_tras = np.array([len(x) for x in tras])
        n_trbs = np.array([len(x) for x in trbs])
        # Replace empty lists by None:
        tras = [x if len(x) > 0 else [None] for x in tras]
        trbs = [x if len(x) > 0 else [None] for x in trbs]
        # Assemble two dimensional list.
        cdr3s = [[tras[i][0], trbs[i][0]] for i, _ in enumerate(tras)]

        # Format and match clonotype TCR sequences:
        if clonotype_table is not None:
            tcr_clono_raw = clonotype_table[id_cdr3].values
            tcr_clono_raw_ls = tcr_clono_raw.tolist()
            cell_assign_clono = np.array([tcr_clono_raw_ls.index(x) for x in tcr_cell_raw])
            assert len(cell_assign_clono) == len(tcr_cell_raw)
        else:
            # Define one clonotype per cell.
            cell_assign_clono = np.arange(0, len(tcr_cell_raw))
        print("Found %i clonotypes for %i observations in single file." % (len(np.unique(cell_assign_clono)), len(cell_assign_clono)))

        # Extract labels:
        # Check that lable_cols is a list to ensure that y is 2D and
        # does not collapse to 1D if only one label is given:
        if isinstance(lable_cols, str):
            lable_cols = [lable_cols]
        # Turns boolean columns into numeric and keeps numeric as numeric.
        y = cell_table[lable_cols].values.astype(float)
        if sparse:
            y = scipy.sparse.csr_matrix(y)

        # Extract negative controls:
        if nc_cols is not None:
            if isinstance(nc_cols, str):
                nc_cols = [nc_cols]
            # Turns boolean columns into numeric and keeps numeric as numeric.
            nc = cell_table[nc_cols].values.astype(float)
        else:
            nc = np.zeros([y.shape[0], 0])

        # Discard doublets:
        if discard_doublets:
            idx_tokeep = np.where(np.logical_and(n_tras <= 1, n_trbs <= 1))[0]
            cdr3s = [cdr3s[i] for i in idx_tokeep]
            covariates_table = covariates_table.iloc[idx_tokeep, :]
            y = y[idx_tokeep, :]
            nc = nc[idx_tokeep, :]
            cell_assign_clono = cell_assign_clono[idx_tokeep]

        return cdr3s, covariates_table, y, nc, cell_assign_clono

    def _format_covariates_from_table(
            self,
            formula_categ: list,
            formula_numeric: list,
            rename_covariates_for_patsy: bool,
            table: pd.DataFrame
    ):
        """ Build design matrix based on formula and extracted covarate (sample annotation) table.

        :param formula_categ: Terms for patsy formula for categorical predictors of design matrix.
            Leave out "~0+" at the start of the formula, this is added automatically. Note that it does not make sense
            to include an intercept as this is already present in the network in the form of a bias.
        :param formula_numeric: Terms for patsy formula for numerical predictors of design matrix.
            Leave out "~0+" at the start of the formula, this is added automatically. Note that it does not make sense
            to include an intercept as this is already present in the network in the form of a bias.
        :param rename_covariates_for_patsy: Whether to automatically change predictor naming to comply with patsy.
            In particular, column names that include a "~", "+" or a "-" are stripped of these characters.
        :param table: Table of predictors to build design matrix from.
        :return:
        """
        # Process term names
        def strip_string(string_to_strip):
            x_new = string_to_strip
            if len(x_new.split("~")) > 0:
                x_new = "_".join(x_new.split("~"))
            if len(x_new.split("+")) > 0:
                x_new = "_".join(x_new.split("+"))
            if len(x_new.split("-")) > 0:
                x_new = "_".join(x_new.split("-"))
            if len(x_new.split("(")) > 0:
                x_new = "_".join(x_new.split("("))
            if len(x_new.split(")")) > 0:
                x_new = "_".join(x_new.split(")"))
            return x_new

        table_temp = table.copy()
        if rename_covariates_for_patsy:
            for x in table_temp.columns:
                new_id = strip_string(x)
                table_temp[new_id] = table_temp[x].values
            for i, x in enumerate(formula_categ):
                new_id = strip_string(x)
                formula_categ[i] = new_id
            for i, x in enumerate(formula_numeric):
                new_id = strip_string(x)
                formula_numeric[i] = new_id

        # Process table data types.
        if formula_categ != "":
            for term in formula_categ:
                term = term.split(" ")
                term = [x for x in term if len(x) > 0][0]
                table_temp[term] = [str(x) for x in table_temp[term].values]
        if formula_numeric != "":
            for term in formula_numeric:
                term = term.split(" ")
                term = [x for x in term if len(x) > 0][0]
                table_temp[term] = [float(x) for x in table_temp[term].values]

        # Assemble formula.
        formula_terms = formula_categ + formula_numeric
        if len(formula_terms) > 0:
            formula = "~0+" + "+".join(formula_terms)
        else:
            formula = "~0"

        # Build design matrix.
        dmat = patsy.dmatrix(
            formula_like=formula,
            data=table_temp,
            return_type="matrix"
        )
        return dmat

    def divide_multilabel_observations_by_label(
            self,
            is_train,
            down_sample_negative: bool = True
    ):
        """ Split multilabel observations into individual observations.

        Call after reading with one of the following:

            - read_binarzed_matrix()

        :param is_train: Whether to use split training data or test data.
        :param down_sample_negative: Whether to downsample negative observations of each label
            to number of positive observations.
        """
        # Extract all peptides from label_cols.
        if self.label_cols is None:
            raise ValueError("label_cols was not set.")
        pep_seq = self.label_ids
        pep_embedding = self._format_data_aa(x=[[x] for x in pep_seq], tcr=False)

        # Reformat data: separate into one observation by peptide.
        if is_train:
            n_raw_obs = self.y_train.shape[0]
            n_labels = self.y_train.shape[1]
            print("Summary data before label split:")
            print(pd.DataFrame({
                "bound": self.y_train.flatten(),
                "unbound": 1 - self.y_train.flatten(),
                "label": np.tile(pep_seq, n_raw_obs)
            }).groupby("label").sum())

            idx_tokeep = np.arange(0, n_raw_obs * n_labels)
            if down_sample_negative:
                neg_class_sampling_weight = np.tile(
                    np.expand_dims(1 / (1 - np.mean(self.y_train, axis=0)) - 1, axis=0),
                    [n_raw_obs, 1]
                )
                sample_weight = np.ones_like(self.y_train)
                is_negative = self.y_train < 0.5
                sample_weight[is_negative] = neg_class_sampling_weight[is_negative]
                idx_tokeep = idx_tokeep[np.asarray(np.random.binomial(n=1, p=sample_weight.flatten()), bool)]

            self.x_train = np.repeat(self.x_train, repeats=n_labels, axis=0)[idx_tokeep]
            self.covariates_train = np.repeat(self.covariates_train, repeats=n_labels, axis=0)[idx_tokeep]
            self.y_train = np.reshape(self.y_train, [-1, 1])[idx_tokeep]
            self.nc_train = np.repeat(self.nc_train, repeats=n_labels, axis=0)[idx_tokeep]
            self.clone_train = np.repeat(self.clone_train, repeats=n_labels, axis=0)[idx_tokeep]
            self.idx_train_val = np.arange(0, len(idx_tokeep))
            self.peptide_seqs_train = np.tile(pep_seq, n_raw_obs)[idx_tokeep]
            pep_embedding = np.tile(pep_embedding, [n_raw_obs, 1, 1, 1])[idx_tokeep]

            if down_sample_negative:
                counts_ds = pd.DataFrame({
                    "bound": self.y_train.flatten(),
                    "unbound": 1 - self.y_train.flatten(),
                    "label": np.tile(pep_seq, n_raw_obs)[idx_tokeep]
                }).groupby("label").sum()
                print("Summary data after down-sampling:")
                print(counts_ds)

            pep_embedding = self._format_antigens(x=pep_embedding)
            self.x_train = np.concatenate([self.x_train, pep_embedding], axis=2)
        else:
            n_raw_obs = self.y_test.shape[0]
            n_labels = self.y_test.shape[1]
            print("Summary data before label split:")
            print(pd.DataFrame({
                "bound": self.y_test.flatten(),
                "unbound": 1 - self.y_test.flatten(),
                "label": np.tile(pep_seq, n_raw_obs)
            }).groupby("label").sum())

            idx_tokeep = np.arange(0, n_raw_obs * n_labels)
            if down_sample_negative:
                neg_class_sampling_weight = np.tile(
                    np.expand_dims(1 / (1 - np.mean(self.y_test, axis=0)) - 1, axis=0),
                    [n_raw_obs, 1]
                )
                sample_weight = np.ones_like(self.y_test)
                is_negative = self.y_test < 0.5
                sample_weight[is_negative] = neg_class_sampling_weight[is_negative]
                idx_tokeep = idx_tokeep[np.asarray(np.random.binomial(n=1, p=sample_weight.flatten()), bool)]

            self.x_test = np.repeat(self.x_test, repeats=n_labels, axis=0)[idx_tokeep]
            self.covariates_test = np.repeat(self.covariates_test, repeats=n_labels, axis=0)[idx_tokeep]
            self.y_test = np.reshape(self.y_test, [-1, 1])[idx_tokeep]
            self.nc_test = np.repeat(self.nc_test, repeats=n_labels, axis=0)[idx_tokeep]
            self.clone_test = np.repeat(self.clone_test, repeats=n_labels, axis=0)[idx_tokeep]
            self.idx_test = np.arange(0, len(idx_tokeep))
            self.peptide_seqs_test = np.tile(pep_seq, n_raw_obs)[idx_tokeep]
            pep_embedding = np.tile(pep_embedding, [n_raw_obs, 1, 1, 1])[idx_tokeep]

            if down_sample_negative:
                counts_ds = pd.DataFrame({
                    "bound": self.y_test.flatten(),
                    "unbound": 1 - self.y_test.flatten(),
                    "label": np.tile(pep_seq, n_raw_obs)[idx_tokeep]
                }).groupby("label").sum()
                print("Summary data after down-sampling:")
                print(counts_ds)

            pep_embedding = self._format_antigens(x=pep_embedding)
            self.x_test = np.concatenate([self.x_test, pep_embedding], axis=2)

        self.label_ids = ["bound"]

    def read_tcr_from_adata(
            self,
            adata: anndata.AnnData,
            fn_blosum: Union[str, None] = None,
            blosum_encoding: bool = False,
            covariate_formula_categ: list = [],
            covariate_formula_numeric: list = [],
            rename_covariates_for_patsy: bool = True,
            ignore_doublets: bool = True,
            chains: str = 'trb',
            id_cdr3: str = 'cell_clono_cdr3_aa',
            is_train: bool = True
    ):
        """ Read TCR sequences and covariates from adata object into a predefined estimator.

        :return:
        """
        self.chains = chains.lower()
        cdr3_seqs = []
        tcr_temp = [[y.split(':') for y in x.split(';')] for x in adata.obs[id_cdr3]]
        tras = [[y[1] for y in x if y[0] == 'TRA'] for x in tcr_temp]
        trbs = [[y[1] for y in x if y[0] == 'TRB'] for x in tcr_temp]
        for i in range(len(tras)):
            if len(tras[i]) == 0:
                tra = None
            elif len(tras[i]) == 1:
                tra = tras[i][0]
            else:
                if ignore_doublets:
                    tra = None
                else:
                    tra = tras[i][0]
            if len(trbs[i]) == 0:
                trb = None
            elif len(trbs[i]) == 1:
                trb = trbs[i][0]
            else:
                if ignore_doublets:
                    trb = None
                else:
                    trb = trbs[i][0]
            if self.chains.lower() == "tra":
                cdr3_seqs.append([tra, None])
            elif self.chains.lower() == "trb":
                cdr3_seqs.append([None, trb])
            elif self.chains.lower() == "concat":
                cdr3_seqs.append([tra, trb])
            else:
                raise ValueError("chains %s not recognized" % self.chains)

        cdr3s = self._format_data_aa(
            x=cdr3_seqs,
            fn_blosum=fn_blosum if blosum_encoding else None,
            tcr=True
        )
        covariates = self._format_covariates_from_table(
            formula_categ=covariate_formula_categ,
            formula_numeric=covariate_formula_numeric,
            rename_covariates_for_patsy=rename_covariates_for_patsy,
            table=adata.obs
        )

        # Assign class data container attributes.
        self._assign_test_train(
            is_train=is_train,
            x=cdr3s,
            covariates=covariates,
            y=np.zeros([cdr3s.shape[0], self.y_train.shape[1]]) + np.nan,
            nc=None,
            clonotype=None,
            cdr3_seqs=cdr3_seqs
        )

    def _assign_test_train(
            self,
            x: np.ndarray,
            covariates: np.ndarray,
            y: np.ndarray,
            is_train: bool,
            nc=None,
            clonotype=None,
            pep_seqs: list = [],
            cdr3_seqs: list = []
    ):
        """ Unified interface to assign newly read data to test or train.

        :param x: Observations
        :param covariates: Covariates.
        :param y: Labels.
        :param nc: Negative controls.
        :param clone: Clone assignment vector of observations.
        :param is_train: Whether data is to be stored in training or test data.
        :param pep_seqs: List of char encoded peptide sequences.
            Does not have to be supplied.
        :param cdr3_seqs: List of lists char encoded TRA and TRB sequences by observation.
            Does not have to be supplied.
        :return:
        """
        if is_train:
            if self.x_train is None:
                self.x_train = x
                self.covariates_train = covariates
                self.y_train = y
                self.nc_train = nc
                self.clone_train = clonotype
                self.idx_train_val = np.arange(0, x.shape[0])

                self.peptide_seqs_train = [x[0] for x in pep_seqs] if len(pep_seqs) > 0 else None
                self.tra_seqs_train = [x[0] for x in cdr3_seqs]
                self.trb_seqs_train = [x[1] for x in cdr3_seqs]
            else:
                self.x_train = np.concatenate([self.x_train, x], axis=0)
                self.covariates_train = np.concatenate([self.covariates_train, covariates], axis=0)
                self.y_train = np.concatenate([self.y_train, y], axis=0)
                if nc is not None:
                    self.nc_train = np.concatenate([self.nc_train, nc], axis=0)
                if clonotype is not None:
                    self.clone_train = np.concatenate([self.clone_train, clonotype], axis=0)
                self.idx_train_val = np.concatenate([
                    self.idx_train_val,
                    np.arange(np.max(self.idx_train_val), np.max(self.idx_train_val) + x.shape[0])
                ], axis=0)

                if len(pep_seqs) > 0:
                    self.peptide_seqs_train.extend([x[0] for x in pep_seqs])
                self.tra_seqs_train.extend([x[0] for x in cdr3_seqs])
                self.trb_seqs_train.extend([x[1] for x in cdr3_seqs])
        else:
            if self.x_test is None:
                self.x_test = x
                self.covariates_test = covariates
                self.y_test = y
                self.nc_test = nc
                self.clone_test = clonotype
                self.idx_test = np.arange(0, x.shape[0])

                self.peptide_seqs_test = [x[0] for x in pep_seqs] if len(pep_seqs) > 0 else None
                self.tra_seqs_test = [x[0] for x in cdr3_seqs]
                self.trb_seqs_test = [x[1] for x in cdr3_seqs]
            else:
                self.x_test = np.concatenate([self.x_test, x], axis=0)
                self.covariates_test = np.concatenate([self.covariates_test, covariates], axis=0)
                self.y_test = np.concatenate([self.y_test, y], axis=0)
                if nc is not None:
                    self.nc_test = np.concatenate([self.nc_test, nc], axis=0)
                if clonotype is not None:
                    self.clone_test = np.concatenate([self.clone_test, clonotype], axis=0)
                self.idx_test = np.concatenate([
                    self.idx_test,
                    np.arange(np.max(self.idx_test), np.max(self.idx_test) + x.shape[0])
                ], axis=0)

                if len(pep_seqs) > 0:
                    self.peptide_seqs_test.extend([x[0] for x in pep_seqs])
                self.tra_seqs_test.extend([x[0] for x in cdr3_seqs])
                self.trb_seqs_test.extend([x[1] for x in cdr3_seqs])

    def _subset_data(self, idx_new: np.ndarray, data: str):
        """

        :param idx_new: Indices of new observation set.
        :param data: {"train", "test"} Whether to perform subsetting on train or test data.
        :return:
        """
        if len(idx_new) == 0:
            raise ValueError("Observation removal criteria were too stringent " +
                             "and left zero observations at this point.")
        if data.lower() == "train":
            self.x_train = self.x_train[idx_new, :]
            self.covariates_train = self.covariates_train[idx_new, :]
            self.y_train = self.y_train[idx_new, :]
            if self.nc_train is not None:
                self.nc_train = self.nc_train[idx_new, :]
            if self.clone_train is not None:
                self.clone_train = self.clone_train[idx_new]
            self.idx_train_val = self.idx_train_val[idx_new]
            self.frac_positives = np.sum(self.y_train > 0.5) / (self.y_train.shape[0] * self.y_train.shape[1])
            if self.peptide_seqs_train is not None:
                self.peptide_seqs_train = [self.peptide_seqs_train[i] for i in idx_new]
            if self.tra_seqs_train is not None:
                self.tra_seqs_train = [self.tra_seqs_train[i] for i in idx_new]
            if self.trb_seqs_train is not None:
                self.trb_seqs_train = [self.trb_seqs_train[i] for i in idx_new]
        elif data.lower() == "test":
            self.x_test = self.x_test[idx_new, :]
            self.covariates_test = self.covariates_test[idx_new, :]
            self.y_test = self.y_test[idx_new, :]
            if self.nc_test is not None:
                self.nc_test = self.nc_test[idx_new, :]
            if self.clone_test is not None:
                self.clone_test = self.clone_test[idx_new]
            self.idx_test = self.idx_test[idx_new]
            self.frac_test_positives = np.sum(self.y_test > 0.5) / (self.y_test.shape[0] * self.y_test.shape[1])
            if self.peptide_seqs_test is not None:
                self.peptide_seqs_test = [self.peptide_seqs_test[i] for i in idx_new]
            if self.tra_seqs_test is not None:
                self.tra_seqs_test = [self.tra_seqs_test[i] for i in idx_new]
            if self.trb_seqs_test is not None:
                self.trb_seqs_test = [self.trb_seqs_test[i] for i in idx_new]
        else:
            assert False

    def _format_data_aa(
            self,
            x: list,
            fn_blosum: Union[str, None] = None,
            tcr: bool = False
    ) -> np.ndarray:
        """
        Create numeric input data from amino acid sequence data.

        :param x: Input as list (length observations) of lists with strings of amino acid code of each chain.
        :param fn_blosum:
        :return: 4D tensor [observations, chains, amino acid position, amino acid embedding]
            One-hot encoded input.
        """
        if fn_blosum is not None:
            # Blosum encoding
            blosum_embedding = read_blosum(fn_blosum)
            x_encoded = encode_as_blosum(x=x, blosum_embedding=blosum_embedding)
        else:
            # One-hot encoding
            x_encoded = encode_as_onehot(x=x, dict_aa=self.dict_aa, eos_char=self.eos_char)

        if tcr:
            x_encoded = self._format_tcr_chains(x=x_encoded)
        else:
            x_encoded = self._format_antigens(x=x_encoded)
        return x_encoded

    def _format_tcr_chains(
            self,
            x
    ):
        """

        :param x:
        :return:
        """
        if self.chains == "tra":
            x = np.expand_dims(x[:, 0, :, :], axis=1)
        elif self.chains == "trb":
            x = np.expand_dims(x[:, 1, :, :], axis=1)
        elif self.chains == "separate":
            pass
        elif self.chains == "concat":
            x = np.concatenate([
                np.expand_dims(x[:, 0, :, :], axis=1),
                np.expand_dims(x[:, 1, :, :], axis=1)
            ], axis=2)
        else:
            raise ValueError("self.chains %s not recognized" % self.chains)

        if self.tcr_len is None:
            self.tcr_len = x.shape[2]
        x = self._pad_tcr(x=x)
        return x

    def _pad_tcr(
            self,
            x: np.ndarray
    ):
        """ Pad TCR to desired length.

        Takes care of chain concatenation: If self.chain is "concat", splits x equally in axis 2 into TRA and TRB
        and pads each to self.tcr_len / 2, then concatenates in axis 2 again. If self.chain is "tra" or "trb", pads
        x to self.tcr_len in axis 2.

        :param x: TCR sequence encoding.
        :return: Padded TCR encoding.
        """
        def pad_block(shape):
            pad_embedding = np.zeros([1, len(self.aa_list)])
            pad_embedding[0, -1] = 1
            return np.zeros(shape) + pad_embedding

        if self.chains.lower() == "concat":
            assert self.tcr_len % 2 == 0, \
                "self.tcr_len (%i) must be divisible by two if 'concat' mode is used for TCR." % self.tcr_len
            assert x.shape[2] % 2 == 0, \
                "dimension 3 of x (%i) must be divisible by two if 'concat' mode is used for TCR." % x.shape[2]
            assert x.shape[2] <= self.tcr_len, \
                "Required tcr length (%i) must at least as high as existing tcr length (%i)." % \
                (self.tcr_len, x.shape[2])
            xa = x[:, :, :int(x.shape[2] / 2), :]
            xb = x[:, :, int(x.shape[2] / 2):, :]
            xa = np.concatenate([
                xa, pad_block([x.shape[0], x.shape[1], int(self.tcr_len / 2) - int(x.shape[2] / 2), x.shape[3]])
            ], axis=2)
            xb = np.concatenate([
                xb, pad_block([x.shape[0], x.shape[1], int(self.tcr_len / 2) - int(x.shape[2] / 2), x.shape[3]])
            ], axis=2)
            x = np.concatenate([xa, xb], axis=2)
        elif self.chains.lower() in ["tra", "trb", "separate"]:
            x = np.concatenate([
                x, pad_block([x.shape[0], x.shape[1], self.tcr_len - x.shape[2], x.shape[3]])
            ], axis=2)
        else:
            raise ValueError("self.chains %s not recognized" % self.chains)
        return x

    def _format_antigens(
            self,
            x
    ):
        """

        :param x:
        :return:
        """
        def pad_block(shape):
            pad_embedding = np.zeros([1, len(self.aa_list)])
            pad_embedding[0, -1] = 1
            return np.zeros(shape) + pad_embedding

        x = np.expand_dims(x[:, 0, :, :], axis=1)
        if self.pep_len is None:
            self.pep_len = x.shape[2]
        else:
            assert self.pep_len >= x.shape[2], \
                "Pre-set antigen length (%i) is smaller than found antigen length %i." % (self.pep_len, x.shape[2])
            x = np.concatenate([
                x, pad_block([x.shape[0], x.shape[1], self.pep_len - x.shape[2], x.shape[3]])
            ], axis=2)
        return x

    def assign_clonotype(
            self,
            flavor: str = "manhatten",
            data: str = "train"
    ):
        """ Assign clonotypes to data stored in x_train.

        Use this before train-test split before training to ensure that train-test and train-validation splits
        take clonotype substructure of the data into account.

        :param flavor: Distance metric.

            - "manhatten": Manhatten distance on peptide sequence. One unit is one amino acid mismatch.
                Any amino acid mistmatch, including length mismatch is included.
                Expects one-hot encoded peptides in self.x_train.
        :param data: {"train", "test"} Whether to downsample in training or test partition.
        :return:
        """
        dist_mat = self._compute_distance(flavor=flavor, data=data)
        clonotypes = np.zeros([dist_mat.shape[0]], int)
        if flavor.lower() == "manhatten":
            clonotype_counter = 0
            for i in range(dist_mat.shape[0]):
                matches = np.where(dist_mat[i, :i] == 0)[0]
                if len(matches) > 0:
                    clonotypes[i] = clonotypes[matches[0]]
                else:
                    clonotypes[i] = clonotype_counter
                    clonotype_counter = clonotype_counter + 1
        else:
            raise ValueError("flavor %s not recognized" % type)
        print("Found %i clonotypes for %i observations." % (len(np.unique(clonotypes)), len(clonotypes)))

        if data.lower() == "train":
            self.clone_train = clonotypes
        elif data.lower() == "test":
            self.clone_test = clonotypes
        else:
            raise ValueError("data %s not recognized" % data)

    def remove_overlapping_tcrs(
            self,
            data: str = "test"
    ):
        """ Remove TCRs from test or train data that also occur in other partition.

        If data is "test", the intersection of TCR sequences between test and train will be removed from test.
        If data is "train", the intersection of TCR sequences between test and train will be removed from train.
        The intersection is defined based on both TRA and TRB. If at least one chain is None, this chain is treated
        as matched.

        :param data: Which partition to remove TCR sequences from.
        :return:
        """
        assert self.tra_seqs_train is not None, "tra_seqs_train was not set"
        assert self.trb_seqs_train is not None, "trb_seqs_train was not set"
        assert self.tra_seqs_test is not None, "tra_seqs_test was not set"
        assert self.trb_seqs_test is not None, "trb_seqs_test was not set"

        intersection_tra = set(self.tra_seqs_train).intersection(self.tra_seqs_test) - set([None])
        intersection_trb = set(self.trb_seqs_train).intersection(self.trb_seqs_test) - set([None])
        if data.lower() == "train":
            n_cells = len(self.tra_seqs_train)
            idx_to_keep = np.where(np.logical_or(
                np.array([x not in intersection_tra for x in self.tra_seqs_train]),
                np.array([x not in intersection_trb for x in self.trb_seqs_train])
            ))[0]
            self._subset_data(idx_new=idx_to_keep, data="train")
        elif data.lower() == "test":
            n_cells = len(self.tra_seqs_test)
            idx_to_keep = np.where(np.logical_or(
                np.array([x not in intersection_tra for x in self.tra_seqs_test]),
                np.array([x not in intersection_trb for x in self.trb_seqs_test])
            ))[0]
            self._subset_data(idx_new=idx_to_keep, data="test")
        else:
            raise ValueError("data %s not recognized." % data)

        print("Reduced " + str(n_cells) + " cells to " + str(len(idx_to_keep)) + " cells in " +
              data + " data because of TCR overlap.")

    def remove_overlapping_antigens(
            self,
            data: str = "test"
    ):
        """ Remove antigens from test or train data that also occur in other partition.

        If data is "test", the intersection of antigens sequences between test and train will be removed from test.
        If data is "train", the intersection of antigens sequences between test and train will be removed from train.

        :param data: Which partition to remove antigens sequences from.
        :return:
        """
        assert self.peptide_seqs_train is not None, "pep_seqs_train was not set"
        assert self.peptide_seqs_test is not None, "pep_seqs_test was not set"

        intersection_pep = set(self.peptide_seqs_train).intersection(set(self.peptide_seqs_test))
        if data.lower() == "train":
            n_cells = len(self.peptide_seqs_train)
            idx_to_keep = np.where([x not in intersection_pep for x in self.peptide_seqs_train])[0]
            self._subset_data(idx_new=idx_to_keep, data="train")
        elif data.lower() == "test":
            n_cells = len(self.peptide_seqs_test)
            idx_to_keep = np.where([x not in intersection_pep for x in self.peptide_seqs_test])[0]
            self._subset_data(idx_new=idx_to_keep, data="test")
        else:
            raise ValueError("data %s not recognized." % data)

        print("Reduced " + str(n_cells) + " cells to " + str(len(idx_to_keep)) + " cells in " +
              data + " data because of antigen overlap.")

    def remove_antigens_byfreq(
            self,
            min_count: Union[int, None] = None,
            max_count: Union[int, None] = None,
            data: str = "train"
    ):
        """ Remove antigens by frequency in partition of data set.

        Keeps observations that correspond to antigens that occur minimum min_count times and
        maximum max_count times in partition of data set.

        :param min_count: Minimum number of occurrences of antigen corresponding to observation
            in data set to keep observation.
        :param max_count: Maximum number of occurrences of antigen corresponding to observation
            in data set to keep observation.
        :param data: Which partition to remove antigens sequences from.
        :return:
        """
        if data.lower() == "train":
            assert self.peptide_seqs_train is not None, "pep_seqs_train was not set"
            n_cells = len(self.peptide_seqs_train)
            antigen_freq = pd.Series(self.peptide_seqs_train).value_counts()
            to_keep = np.repeat(True, antigen_freq.shape[0])
            if min_count is not None:
                to_keep = np.logical_and(to_keep, antigen_freq.values >= min_count)
            if max_count is not None:
                to_keep = np.logical_and(to_keep, antigen_freq.values < max_count)
            antigens_to_keep = antigen_freq.index[to_keep]
            idx_to_keep = np.where([x in antigens_to_keep for x in self.peptide_seqs_train])[0]
            self._subset_data(idx_new=idx_to_keep, data="train")
        elif data.lower() == "test":
            assert self.peptide_seqs_test is not None, "pep_seqs_test was not set"
            n_cells = len(self.peptide_seqs_test)
            antigen_freq = pd.Series(self.peptide_seqs_test).value_counts()
            to_keep = np.repeat(True, antigen_freq.shape[0])
            if min_count is not None:
                to_keep = np.logical_and(to_keep, antigen_freq.values >= min_count)
            if max_count is not None:
                to_keep = np.logical_and(to_keep, antigen_freq.values < max_count)
            antigens_to_keep = antigen_freq.index[to_keep]
            idx_to_keep = np.where([x in antigens_to_keep for x in self.peptide_seqs_test])[0]
            self._subset_data(idx_new=idx_to_keep, data="test")
        else:
            raise ValueError("data %s not recognized." % data)

        print("Reduced " + str(n_cells) + " cells to " + str(len(idx_to_keep)) + " cells in " +
              data + " data because they did not match antigen frequency thresholds.")

    def downsample_clonotype(
            self,
            max_obs: int = 10,
            data: str = "train"
    ):
        """ Downsample clonotypes to data stored in x_train.

        This avoids training, evaluation or test set to be too biased to a subset of TCRs.
        Use this before train-test split before training to ensure that train-test and train-validation splits
        take clonotype substructure of the data into account.

        :param max_obs: Maximum number of observations per clonotype.
        :param data: {"train", "test"} Whether to downsample in training or test partition.
        :return:
        """
        if data.lower() == "train":
            clonotypes = np.unique(self.clone_train)
            n_cells = self.x_train.shape[0]
            idx_to_keep = np.concatenate([np.random.choice(
                a=np.arange(0, n_cells)[self.clone_train == x],
                size=np.min([max_obs, np.sum(self.clone_train == x)]),
                replace=False
            ) for x in np.unique(self.clone_train)], axis=0)
            self._subset_data(idx_new=idx_to_keep, data="train")
        elif data.lower() == "test":
            clonotypes = np.unique(self.clone_test)
            n_cells = self.x_test.shape[0]
            idx_to_keep = np.concatenate([np.random.choice(
                a=np.arange(0, n_cells)[self.clone_test == x],
                size=np.min([max_obs, np.sum(self.clone_test == x)]),
                replace=False
            ) for x in np.unique(self.clone_test)], axis=0)
            self._subset_data(idx_new=idx_to_keep, data="test")
        else:
            raise ValueError("data %s not recognized." % data)

        print("Downsampled %i clonotypes from %i cells to %i cells." % (len(clonotypes), n_cells, len(idx_to_keep)))

    def downsample_data(
            self,
            n: int = 10,
            data: str = "train",
            use_min: bool = True
    ):
        """ Downsample data to given number of observations.

        Use this before train-test split before training.

        :param n: Number of observations to downsample to.
        :param data: {"train", "test"} Whether to downsample in training or test partition.
        :param use_min: Whether to simply not downsample if the number of cells is smaller than the required number.
        :return:
        """
        if data.lower() == "train":
            n_cells = self.x_train.shape[0]
            if not use_min:
                assert n <= self.x_test.shape[0], "Subsample is too big: there are only %i cells in data." % n_cells
            idx_to_keep = np.random.choice(
                a=np.arange(0, n_cells),
                size=np.min([n, n_cells]) if use_min else n,
                replace=False
            )
            self._subset_data(idx_new=idx_to_keep, data="train")
        elif data.lower() == "test":
            n_cells = self.x_test.shape[0]
            if not use_min:
                assert n <= self.x_test.shape[0], "Subsample is too big: there are only %i cells in data." % n_cells
            idx_to_keep = np.random.choice(
                a=np.arange(0, n_cells),
                size=np.min([n, n_cells]) if use_min else n,
                replace=False
            )
            self._subset_data(idx_new=idx_to_keep, data="test")
        else:
            raise ValueError("data %s not recognized." % data)

        print("Downsampled " + data + " data from " + str(n_cells) + " cells to " + str(len(idx_to_keep)) + " cells.")

    def upsample_labels(
            self,
            data: str = "train"
    ):
        """ Upsample all classes to maximum number of observations observed across all classes.

        Use this before train-test split before training.

        :param data: {"train", "test"} Whether to downsample in training or test partition.
        :return:
        """
        if data.lower() == "train":
            y_classes = np.unique(self.y_train, axis=0)
            n_obs = np.array([np.sum(np.all(self.y_train == x, axis=1)) for x in y_classes])
            idx_upsample = np.concatenate([
                np.arange(0, self.y_train.shape[0]),  # original observations
                np.concatenate([
                    np.random.choice(  # upsample
                        a=np.arange(0, self.y_train.shape[0])[np.all(self.y_train == x, axis=1)],
                        size=np.max(n_obs) - n_obs[i],
                        replace=True
                    ) for i, x in enumerate(y_classes)
                ])
            ])
            self._subset_data(idx_new=idx_upsample, data="train")
        elif data.lower() == "test":
            y_classes = np.unique(self.y_test, axis=0)
            n_obs = np.array([np.sum(np.all(self.y_test == x, axis=1)) for x in y_classes])
            idx_upsample = np.concatenate([
                np.arange(0, self.y_test.shape[0]),  # original observations
                np.concatenate([
                    np.random.choice(  # upsample
                        a=np.arange(0, self.y_test.shape[0])[np.all(self.y_test == x, axis=1)],
                        size=np.max(n_obs) - n_obs[i],
                        replace=True
                    ) for i, x in enumerate(y_classes)
                ])
            ])
            self._subset_data(idx_new=idx_upsample, data="test")
        else:
            raise ValueError("data %s not recognized." % data)

        print("Upsampled %i classes from %s cells to %i cells." %
              (y_classes.shape[0], ", ".join([str(x) for x in n_obs]), np.max(n_obs)))

    def downsample_labels(
            self,
            data: str = "train",
            multiple_of_min: int = 1
    ):
        """ Downsample all classes to multiple of minimum number of observations observed across all classes.

        Use this before train-test split before training.

        :param data: {"train", "test"} Whether to downsample in training or test partition.
        :param multiple_of_min: Target number of observations after downsampling as multiple of minimum
            number of observations per class. Note that this will perform upsampling of small classes if >1.
            Is replaced by maximum number of observations per class if it exceeds this maximum.
        :return:
        """
        if data.lower() == "train":
            y_classes = np.unique(self.y_train, axis=0)
            n_obs = np.array([np.sum(np.all(self.y_train == x, axis=1)) for x in y_classes])
            idx_downsample = np.concatenate([
                np.random.choice(
                    a=np.arange(0, self.y_train.shape[0])[np.all(self.y_train == x, axis=1)],
                    size=np.min([np.min(n_obs) * multiple_of_min, np.max(n_obs)]),
                    replace=True
                ) for i, x in enumerate(y_classes)
            ])
            self._subset_data(idx_new=idx_downsample, data="train")
        elif data.lower() == "test":
            y_classes = np.unique(self.y_test, axis=0)
            n_obs = np.array([np.sum(np.all(self.y_test == x, axis=1)) for x in y_classes])
            idx_downsample = np.concatenate([
                np.random.choice(
                    a=np.arange(0, self.y_test.shape[0])[np.all(self.y_test == x, axis=1)],
                    size=np.min([np.min(n_obs) * multiple_of_min, np.max(n_obs)]),
                    replace=True
                ) for i, x in enumerate(y_classes)
            ])
            self._subset_data(idx_new=idx_downsample, data="test")
        else:
            raise ValueError("data %s not recognized." % data)

        print("Downsampled %i classes from %s cells to %i cells." %
              (y_classes.shape[0], ", ".join([str(x) for x in n_obs]), np.min(n_obs)))

    def pad_sequence(
            self,
            target_len: int = 0,
            sequence: str = "tcr"
    ):
        """ Pad TCR or peptide sequences to desired length.

        Use this before train-test split and before training.

        :param target_len: Sequence length to pad to.
        :param sequence: {"tcr", "antigen"} Whether to pad TCR or antigen sequence.
        :return:
        """
        if sequence.lower() == "tcr":
            if target_len >= self.tcr_len:
                tcr_len_old = self.tcr_len
                self.tcr_len = target_len
                # Pad train data.
                if self.x_train is not None:
                    x = self._pad_tcr(x=self.x_train[:, :, :tcr_len_old, :])
                    self.x_train = np.concatenate([x, self.x_train[:, :, tcr_len_old:, :]], axis=2)
                # Pad test data.
                if self.x_test is not None:
                    x = self._pad_tcr(x=self.x_test[:, :, :tcr_len_old, :])
                    self.x_test = np.concatenate([x, self.x_test[:, :, tcr_len_old:, :]], axis=2)
            else:
                raise ValueError(
                    "Padding for TCR chains was too short (%i), longer sequences observed (%i)." %
                    (target_len, self.tcr_len)
                )
        elif sequence.lower() == "antigen":
            if target_len >= self.pep_len:
                self.pep_len = target_len
                # Pad train data.
                if self.x_train is not None:
                    x = self.x_train[:, :, self.tcr_len:, :]
                    x = self._format_antigens(x=x)
                    self.x_train = np.concatenate([self.x_train[:, :, :self.tcr_len, :], x], axis=2)
                # Pad test data.
                if self.x_test is not None:
                    x = self.x_test[:, :, self.tcr_len:, :]
                    x = self._format_antigens(x=x)
                    self.x_test = np.concatenate([self.x_test[:, :, :self.tcr_len, :], x], axis=2)
            else:
                raise ValueError(
                    "Padding for antigen was too short (%i), longer sequences observed (%i)." %
                    (target_len, self.pep_len)
                )
        else:
            raise ValueError("sequence %s not recognized." % sequence)

    def clear_test_data(self):
        """ Deletes all test data. """
        self.x_test = None
        self.covariates_test = None
        self.y_test = None
        self.nc_test = None
        self.clone_test = None
        self.idx_test = None
        self.frac_test_positives = None

    def _compute_distance(
            self,
            flavor: str = "manhatten",
            data: str = "train"
    ):
        """ Compute full TCR-TCR distance matrix.

        :param flavor: Distance metric.

            - "manhatten": Manhatten distance on peptide sequence. One unit is one amino acid mismatch.
                Any amino acid mistmatch, including length mismatch is included.
                Expects one-hot encoded peptides in self.x_train.
        :param data: {"train", "test"} Whether to downsample in training or test partition.
        :return:
        """
        if data.lower() == "train":
            tra_seqs = self.tra_seqs_train
            trb_seqs = self.trb_seqs_train
        elif data.lower() == "test":
            tra_seqs = self.tra_seqs_test
            trb_seqs = self.trb_seqs_test
        else:
            raise ValueError("data %s not recognized" % data)

        if flavor.lower() == "manhatten":
            # Compute manhatten distance matrix.
            # Compute one-hot encoding for manhatten distance.
            x = self._format_data_aa(
                x=[[x, y] for x, y in zip(tra_seqs, trb_seqs)],
                fn_blosum=None,
                tcr=True
            )
            x_dist = (x - 0.5) * 2  # -1, 1 encoding
            dist_mat = (np.prod(x_dist.shape[1:]) - np.matmul(
                np.reshape(x_dist, newshape=[x_dist.shape[0], -1]),
                np.reshape(x_dist, newshape=[x_dist.shape[0], -1]).T,
            )) / (2 * 2)
            # Divide result by (1) two because a mismatch in one-hot encoding causes manhatten distance of two
            # and (2) by two because one mismatch yields 2 distance units in -1,1 encoding.
        else:
            raise ValueError("flavor %s not recognized" % type)

        return dist_mat

    def plot_tcr_distance_distribution(self, flavor: str = "manhatten"):
        """ Plot distribution of all TCR-TCR distances.

        :param flavor: Distance metric.

            - "manhatten": Manhatten distance on peptide sequence. One unit is one amino acid mismatch.
                Any amino acid mistmatch, including length mismatch is included.
                Expects one-hot encoded peptides in self.x_train.
        :return: axs
        """
        import seaborn as sns
        dist_mat = self._compute_distance(flavor=flavor)
        ax = sns.distplot(
            dist_mat[np.triu_indices(dist_mat.shape[0])],
            kde=False
        )
        return ax

    def rebuild_protein_seq(self, samples):
        """
        bulid protein sequences from one-hot encoding

        :param samples: one-hot encoded protein sequence
        :return: protein sequences with amino acid
        """
        output = {}
        for i in range(int(samples.shape[0])):
            output[i] = ''
            for j in range(int(samples.shape[1])):
                mid = samples[i,j,:]
                output[i] += self.dict_bb[np.argmax(mid)]
        return output

    def log1p_labels(self):
        """ Log transform training and test labels. """
        if self.y_train is not None:
            self.y_train = np.log(self.y_train + 1)
        if self.y_test is not None:
            self.y_test = np.log(self.y_test + 1)

    def sample_test_set(
            self,
            test_split: float
    ):
        """ Define test set based on samples in training data.

        The selected samples are removed from the training data. Important: do this before training!

        :param test_split: Fraction of training data to set aside and keep as held-out test data.
            Note that this split is not exact if clonotype information is present: The split then
            refers to the number of clonotypes in each partition, the fraction of observations in the
            test set then depends on the relative size of the clonotypes sampled for each partition.
        :return:
        """
        # Make sure that there is no test set loaded already.
        if self.x_test is not None:
            raise ValueError("Test data is already loaded (x_test is not None), do not use sample_test_set().")

        # Split training data into training+eval and test.
        # Perform this splitting based on clonotypes.
        clones = np.unique(self.clone_train)
        clones_test = clones[np.random.choice(
            a=np.arange(0, clones.shape[0]),
            size=round(clones.shape[0] * test_split),
            replace=False
        )]
        clones_train_eval = np.array([x for x in clones if x not in clones_test])
        # Collect observations by clone partition:
        idx_test = np.where([x in clones_test for x in self.clone_train])[0]
        idx_train_eval = np.where([x in clones_train_eval for x in self.clone_train])[0]
        # Assert that split is exclusive and complete:
        assert len(set(clones_test).intersection(set(clones_train_eval))) == 0, \
            "ERROR: train-test assignment was not exclusive on level of clones"
        assert len(set(idx_test).intersection(set(idx_train_eval))) == 0, \
            "ERROR: train-test assignment was not exclusive on level of cells"
        assert len(clones_test) + len(clones_train_eval) == len(clones), \
            "ERROR: train-test split was not complete on the level of clones"
        assert len(idx_test) + len(idx_train_eval) == len(self.clone_train), \
            "ERROR: train-test split was not complete on the level of cells"

        print("Number of observations in test data: %i" % len(idx_test))
        print("Number of observations in training+evaluation data: %i" % len(idx_train_eval))

        self.x_test = self.x_train[idx_test].copy()
        self.covariates_test = self.covariates_train[idx_test].copy()
        self.y_test = self.y_train[idx_test].copy()
        if self.nc_train is not None:
            self.nc_test = self.nc_train[idx_test].copy()
        self.clone_test = self.clone_train[idx_test].copy()
        self.idx_test = self.idx_train_val[idx_test]
        self.frac_test_positives = np.sum(self.y_test > 0.5) / (self.y_test.shape[0] * self.y_test.shape[1])
        self.tra_seqs_test = [x for i, x in enumerate(self.tra_seqs_train) if i in idx_test]
        self.trb_seqs_test = [x for i, x in enumerate(self.trb_seqs_train) if i in idx_test]
        self.peptide_seqs_test = [x for i, x in enumerate(self.peptide_seqs_train) if i in idx_test] \
            if self.peptide_seqs_train is not None else None

        self._subset_data(idx_new=idx_train_eval, data="train")

    def save_idx(self, fn: str):
        """ Save indices of observations used in test and training.

        :param fn: Path and file name prefix to write data indices to.
        :return:
        """
        np.save(arr=self.idx_train_val, file=fn + "_idx_train_val.npy")
        if self.x_test is not None:
            np.save(arr=self.idx_test, file=fn + "_idx_test.npy")
        if self.idx_train is not None:
            np.save(arr=self.idx_train, file=fn + "_idx_train.npy")
        if self.idx_val is not None:
            np.save(arr=self.idx_val, file=fn + "_idx_val.npy")

    def load_idx(self, fn: str):
        """ Load indices of observations used in test and training.

        :param fn: Path and file name prefix to read data indices from.
        :return:
        """
        self.idx_train_val = np.load(file=fn + "_idx_train_val.npy")
        if os.path.isfile(fn + "_idx_test.npy"):
            self.idx_test = np.load(file=fn + "_idx_test.npy")
        if os.path.isfile(fn + "_idx_train.npy"):
            self.idx_train = np.load(file=fn + "_idx_train.npy")
        if os.path.isfile(fn + "_idx_eval.npy"):
            self.idx_val = np.load(file=fn + "_idx_val.npy")

    def subset_from_saved(self, fn: str, check_overwrite: bool = True):
        """ Subset data based on saved index vectors.

        Data down-sampling and test-train splits are recovered if this is run on an objects that

        :param fn: Path and file name prefix to read data indices from.
        :param check_overwrite: Whether to check that test set is not loaded yet.
        :return:
        """
        # Make sure that there is no test set loaded already.
        if self.x_test is not None and check_overwrite:
            raise ValueError("Test data is already loaded (x_test is not None), do not use subset_from_saved().")

        # Load index vectors.
        self.load_idx(fn=fn)
        print("Number of observations in test data: %i" % len(self.idx_test))
        print("Number of observations in training+evaluation data: %i" % len(self.idx_train_val))

        # Subset data.
        self.x_test = self.x_train[self.idx_test].copy()
        self.covariates_test = self.covariates_train[self.idx_test].copy()
        self.y_test = self.y_train[self.idx_test].copy()
        if self.nc_train is not None:
            self.nc_test = self.nc_train[self.idx_test].copy()
        self.clone_test = self.clone_train[self.idx_test].copy()
        self.frac_test_positives = np.sum(self.y_test > 0.5) / (self.y_test.shape[0] * self.y_test.shape[1])

        self._subset_data(idx_new=self.idx_train_val, data="train")

    def build_input_pipline(self, batch_size):
        """Build an iterator over training batches."""
        training_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.covariates_train))
        training_batches = training_dataset.shuffle(
            self.x_train.shape[0], reshuffle_each_iteration=True).repeat().batch(batch_size)
        training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)
        samples = training_iterator.get_next()
        return samples
