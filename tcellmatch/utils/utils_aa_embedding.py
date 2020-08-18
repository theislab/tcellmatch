import pandas
import numpy as np
from typing import List


def read_blosum(fn):
    """ Read .txt encoded blosum matrix and return as dictionary of lists.

    :param fn:
    :return: Blosum embedding: keys are amino acids to embed and values are embedding as list
        (similarity ot other amino acids).
    """
    blosum_tab = pandas.read_csv(
        filepath_or_buffer=fn,
        skiprows=2, header=0,
        delimiter=" ", skipinitialspace=True,
        index_col=0
    )
    return blosum_tab.to_dict(orient="list")


def encode_as_blosum(
        x: List[str],
        blosum_embedding: dict
) -> np.ndarray:
    """ Embed amino acid sequences in BLOSUM space.

    Embedding: one dimension for distance to

        - each amino acid
        - end-of-sequence char

    The entries captured are:

        - each amino acid in dictionary
        - unobserved amino acids labelled as "*"
        - end of sequences positions which receive no penalty (value of zero in each dimension)
        - any element in an unobserved peptide: no penalty (value of zero in each dimension)
    
    :param x: Peptide sequences to encode.
    :param blosum_embedding: Blosum embedding: keys are amino acids to embed and values are embedding as list
        (similarity ot other amino acids).
    :return: Peptide sequences in embedding (observations, peptides, sequence positions, embedding dimensions).
    """
    dim_obs = len(x)
    dim_chains = len(x[0])
    dim_pos = np.max([np.max([len(xij) if xij is not None else 0 for xij in xi]) for xi in x]) + 1  # 1 padding
    dim_aa = len(next(iter(blosum_embedding.values()))) + 1  # Add end-of-sequence dimension to embedding.

    x_encoded = np.zeros([dim_obs, dim_chains, dim_pos, dim_aa])
    for i, xi in enumerate(x):  # Loop over observations.
        for j, xij in enumerate(xi):  # Loop over peptides per observation.
            if xij is None:  # Fill with end-of-sequence chars if peptide was not found.
                x_encoded[i, j, :, -1] = 1
            else:
                for k, aa in enumerate(xij):  # Loop over observed sequence positions.
                    x_encoded[i, j, k, :-1] = blosum_embedding[aa]
                # Fill remaining positions as None.
                for k in np.arange(len(xij), dim_pos):  # Loop over padded remaining sequence positions.
                    x_encoded[i, j, k, -1] = 1
    return x_encoded


def encode_as_onehot(
        x: List[str],
        dict_aa: dict,
        eos_char: str
):
    """
    Embed amino acid sequences in one-hot-encodeds space.

    Embedding: one dimension for distance to

        - each amino acid
        - unobserved amino acids ("*" entry in BLOSUM matrix)

    The entries captured are:

        - each amino acid in dictionary
        - end of sequences positions
        - any element in an unobserved peptide labeled as "#"

    :param x: Peptide sequences to encode.
    :param dict_aa: Index of each encoded element in categorical embedding.
    :param eos_char: End-of-sequence char.
    :return: Peptide sequences in embedding (observations, peptides, sequence positions, embedding dimensions).
    """
    dim_obs = len(x)
    dim_chains = len(x[0])
    dim_pos = np.max([np.max([len(xij) if xij is not None else 0 for xij in xi]) for xi in x]) + 1  # 1 padding
    dim_aa = len(dict_aa)

    x_encoded = np.zeros([dim_obs, dim_chains, dim_pos, dim_aa])
    for i, xi in enumerate(x):  # Loop over observations.
        for j, xij in enumerate(xi):  # Loop over peptides per observation.
            if xij is None:  # Write missing string if peptide was not found.
                pass
            else:
                for k, aa in enumerate(xij):  # Loop over observed sequence positions.
                    x_encoded[i, j, k, dict_aa[aa]] = 1
                # Fill remaining positions as None.
                for k in np.arange(len(xij), dim_pos):  # Loop over padded remaining sequence positions.
                    x_encoded[i, j, k, dict_aa[eos_char]] = 1
    return x_encoded
