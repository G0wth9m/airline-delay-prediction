
import argparse
import pandas as pd
import networkx as nx
from src.utils.io import save_parquet, read_csv

def engineer_graph_features(df: pd.DataFrame, window=7):
    # Assume Origin, Dest, ArrDelay columns exist (minutes)
    df = df.copy()
    df['is_delayed15'] = (df['ArrDelay'] > 15).astype(int)

    # Rolling delay rate per airport
    df = df.sort_values(['Origin'])
    origin_delay = df.groupby('Origin')['is_delayed15'].rolling(window, min_periods=1).mean().reset_index()
    origin_delay.columns = ['Origin', 'idx', 'origin_delay_rate']
    df['origin_delay_rate'] = origin_delay['origin_delay_rate'].values

    dest_delay = df.groupby('Dest')['is_delayed15'].rolling(window, min_periods=1).mean().reset_index()
    dest_delay.columns = ['Dest', 'idx', 'dest_delay_rate']
    df['dest_delay_rate'] = dest_delay['dest_delay_rate'].values

    # Build aggregated graph from counts (static for simplicity)
    edges = df.groupby(['Origin','Dest']).size().reset_index(name='w')
    G = nx.from_pandas_edgelist(edges, source='Origin', target='Dest', edge_attr='w', create_using=nx.DiGraph())

    pr = nx.pagerank(G, weight='w') if len(G) else {}
    bc = nx.betweenness_centrality(G, weight='w', normalized=True) if len(G) else {}

    df['origin_pr'] = df['Origin'].map(pr).fillna(0)
    df['dest_pr'] = df['Dest'].map(pr).fillna(0)
    df['origin_bc'] = df['Origin'].map(bc).fillna(0)
    df['dest_bc'] = df['Dest'].map(bc).fillna(0)
    return df[['origin_delay_rate','dest_delay_rate','origin_pr','dest_pr','origin_bc','dest_bc']]

def main(args):
    df = read_csv(args.csv)
    feats = engineer_graph_features(df)
    save_parquet(feats, args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to flights CSV')
    parser.add_argument('--out', default='data/processed/graph_features.parquet')
    main(parser.parse_args())
