
import argparse, pandas as pd
from sklearn.model_selection import train_test_split
from src.models.baselines import make_regression_pipeline, make_classifier_pipeline
from src.features.calendar_time import add_calendar_time
from src.utils.metrics import regression_report, classification_report

def main(args):
    df = pd.read_csv(args.csv)
    df = add_calendar_time(df)
    df['arrival_delay_minutes'] = df.get('ArrDelay', df.get('ArrivalDelay', 0)).astype(float)
    df['is_delayed15'] = (df['arrival_delay_minutes'] > 15).astype(int)

    # merge graph features
    g = None
    if args.graph:
        try:
            import pandas as pd
            g = pd.read_parquet(args.graph)
            g.index = df.index[:len(g)]  # simple alignment for demo purposes
            df = pd.concat([df, g], axis=1)
        except Exception as e:
            print('Warning: could not load graph features:', e)

    cat = [c for c in ['Airline','Origin','Dest'] if c in df.columns]
    num = [c for c in ['Distance','ScheduledElapsedTime','TaxiOut','TaxiIn','dow','dom','month',
                       'origin_delay_rate','dest_delay_rate','origin_pr','dest_pr','origin_bc','dest_bc'] if c in df.columns]
    X = df[cat + num]; yreg = df['arrival_delay_minutes']; yclf = df['is_delayed15']

    X_tr, X_te, yreg_tr, yreg_te = train_test_split(X, yreg, test_size=0.2, random_state=42)
    _, _, yclf_tr, yclf_te = train_test_split(X, yclf, test_size=0.2, random_state=42)

    reg = make_regression_pipeline(cat, num); reg.fit(X_tr, yreg_tr)
    clf = make_classifier_pipeline(cat, num); clf.fit(X_tr, yclf_tr)

    print('Regression:', regression_report(yreg_te, reg.predict(X_te)))
    print('Classification:', classification_report(yclf_te, clf.predict_proba(X_te)[:,1]))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--graph', default='data/processed/graph_features.parquet')
    main(p.parse_args())
