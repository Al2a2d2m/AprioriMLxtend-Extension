# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd


def generate_new_combinations_inverse(old_combinations):
    """
    Generator of all combinations based on the last state of Apriori algorithm
    Parameters
    -----------
    old_combinations: np.array
        All combinations with enough support in the last step
        Combinations are represented by a matrix.
        Number of columns is equal to the combination size
        of the previous step.
        Each row represents one combination
        and contains item type ids in the ascending order
        ```
               0        1
        0      15       20
        1      15       22
        2      17       19
        ```

    Returns
    -----------
    Generator of all combinations from the last step x items
    from the previous step. Every combination is a tuple
    of item type ids in the ascending order.
    No combination other than generated
    do not have a chance to get enough support

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/generate_new_combinations/

    """

    items_types_in_previous_step = np.unique(old_combinations.flatten())
    for old_combination in old_combinations:
        #print(old_combination, "->", max(old_combination))
        max_combination = max(old_combination)
        for item in items_types_in_previous_step:
            if item > max_combination:
                res = tuple(old_combination) + (item,)
                yield res


def apriori_inverse(df, max_support=0.5, use_colnames=False, max_len=None):
    """Get frequent itemsets from a one-hot DataFrame
    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame the encoded format. For example,

    ```
             Apple  Bananas  Beer  Chicken  Milk  Rice
        0      1        0     1        1     0     1
        1      1        0     1        0     0     1
        2      1        0     1        0     0     0
        3      1        1     0        0     0     0
        4      0        0     1        1     1     1
        5      0        0     1        0     1     1
        6      0        0     1        0     1     0
        7      1        1     0        0     0     0
    ```

    min_support : float (default: 0.5)
      A float between 0 and 1 for minumum support of the itemsets returned.
      The support is computed as the fraction
      transactions_where_item(s)_occur / total_transactions.

    use_colnames : bool (default: False)
      If true, uses the DataFrames' column names in the returned DataFrame
      instead of column indices.

    max_len : int (default: None)
      Maximum length of the itemsets generated. If `None` (default) all
      possible itemsets lengths (under the apriori condition) are evaluated.

    Returns
    -----------
    pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
      that are >= `min_support` and < than `max_len`
      (if `max_len` is not None).

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

    """

    X = df.values
    ary_col_idx = np.arange(X.shape[1])
    support = (np.sum(X, axis=0) / float(X.shape[0]))
    support_dict = {1: support[support <= max_support]}
    itemset_dict = {1: ary_col_idx[support <= max_support].reshape(-1, 1)}
    # print(support_dict)

    max_itemset = 1
    rows_count = float(X.shape[0])

    if max_len is None:
        max_len = float('inf')

    while max_itemset and max_itemset < max_len:
        #print("ok1")

        next_max_itemset = max_itemset + 1
        combin = generate_new_combinations_inverse(itemset_dict[max_itemset])
        rare_items = []
        rare_items_support = []
        # calcul des support de chaque combinaison
        for c in combin:
            #print("ok2")
            together = X[:, c].all(axis=1)
            support = together.sum() / rows_count
            #print("ok")
            #print(c, "=>", support)

            # si support faible, on ajoute aux rares
            if support <= max_support:
                rare_items.append(c)
                rare_items_support.append(support)

        if rare_items:
            itemset_dict[next_max_itemset] = np.array(rare_items)
            support_dict[next_max_itemset] = np.array(rare_items_support)
            max_itemset = next_max_itemset
        else:
            max_itemset = 0
    # generation du df contenant les resultats
    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([i for i in itemset_dict[k]])

        res = pd.concat((support, itemsets), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ['support', 'itemsets']
    # mapping si utilisation des noms de cols
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df['itemsets'] = res_df['itemsets'].apply(lambda x: [mapping[i]
                                                                 for i in x])
    res_df = res_df.reset_index(drop=True)
   


    return res_df
