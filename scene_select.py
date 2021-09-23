import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn import preprocessing
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Ready')
    parser.add_argument('--stats', dest='stats', required=True,
                        help='Path to csv with image stats')
    parser.add_argument('--stats_idx', dest='stats_idx', required=False, type=int,
                        help='Column number (start from 0) of filepaths in stats csv')
    parser.add_argument('--meta', dest='meta', required=False,
                        help='Path to csv containing image metadata')
    parser.add_argument('--meta_idx', dest='meta_idx', required=False, type=int,
                        help='Column number (start from 0) of filepaths in meta csv')
    parser.add_argument('--out_fn', dest='out_fn', required=True,
                        help='Full filepath for output csv')
    parser.add_argument('--pref', dest='pref', type=float, required=False,
                        help='AP preference value to be used instead of default')
    args = parser.parse_args()
    stats = args.stats
    stats_idx = args.stats_idx
    meta = args.meta
    meta_idx = args.meta_idx
    out_fn = args.out_fn
    pref = args.pref

    tindex = "filepath"

    if not stats_idx:
        stats_idx = 0
    if not meta_idx:
        meta_idx = 0

    if meta:

        stats_df = pd.read_csv(stats, index_col=stats_idx)
        meta_df = pd.read_csv(meta, index_col=meta_idx)

        sample_df = pd.merge(stats_df, meta_df, left_index=True, right_index=True)

    else:
        sample_df = pd.read_csv(stats, index_col=stats_idx)

    img_vals = sample_df.values
    img_fns = sample_df.index.values

    scaled_vals = preprocessing.scale(img_vals)

    if pref:
        ap = AffinityPropagation(max_iter=5000, convergence_iter=100, affinity="euclidean",
                                 preference=pref).fit(scaled_vals)
    else:
        ap = AffinityPropagation(max_iter=5000, convergence_iter=100, affinity="euclidean").fit(scaled_vals)

    print("There are " + str(len(ap.cluster_centers_indices_)) + " exemplar scenes")

    exemplars = [i in ap.cluster_centers_indices_ for i in range(0, len(img_fns))]

    exemplar_df = pd.DataFrame(data=exemplars, index=img_fns, columns=['exemplar'])
    exemplar_df['cluster'] = ap.labels_
    exemplar_df.to_csv(out_fn, header=True, index=True, index_label='filepath')
