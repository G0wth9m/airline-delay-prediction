
import argparse, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.models.baselines import make_regression_pipeline, make_classifier_pipeline
from src.features.calendar_time import add_calendar_time
from src.utils.metrics import regression_report, classification_report

def prepare(df, target_col):
    # Basic cleaning + engineered calendar features
    df = df.copy()
    df = add_calendar_time(df)
    df['arrival_delay_minutes'] = df.get('ArrDelay', df.get('ArrivalDelay', 0)).astype(float)
    df['is_delayed15'] = (df['arrival_delay_minutes'] > 15).astype(int)
    y_reg = df[target_col] if target_col in df.columns else df['arrival_delay_minutes']
    y_clf = df['is_delayed15']

    # Select a compact feature set
    cat = [c for c in ['Airline','Origin','Dest'] if c in df.columns]
    num = [c for c in ['Distance','ScheduledElapsedTime','TaxiOut','TaxiIn','dow','dom','month'] if c in df.columns]

    X = df[cat + num]
    return X, y_reg, y_clf, cat, num

def main(args):
    df = pd.read_csv(args.csv)
    X, y_reg, y_clf, cat, num = prepare(df, args.target)

    X_train, X_test, yreg_tr, yreg_te = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, yclf_tr, yclf_te = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    reg_pipe = make_regression_pipeline(cat, num)
    cls_pipe = make_classifier_pipeline(cat, num)

    reg_pipe.fit(X_train, yreg_tr)
    cls_pipe.fit(X_train, yclf_tr)

    reg_pred = reg_pipe.predict(X_test)
    cls_proba = cls_pipe.predict_proba(X_test)[:,1]

    print('Regression:', regression_report(yreg_te, reg_pred))
    print('Classification:', classification_report(yclf_te, cls_proba))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to flights CSV')
    parser.add_argument('--target', default='arrival_delay_minutes')
    main(parser.parse_args())
