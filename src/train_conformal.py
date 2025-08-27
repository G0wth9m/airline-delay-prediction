
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.baselines import make_regression_pipeline, make_classifier_pipeline
from src.features.calendar_time import add_calendar_time

def conformal_interval(residuals, alpha=0.1):
    q = np.quantile(np.abs(residuals), 1 - alpha)
    return q

def main(args):
    df = pd.read_csv(args.csv)
    df = add_calendar_time(df)
    df['arrival_delay_minutes'] = df.get('ArrDelay', df.get('ArrivalDelay', 0)).astype(float)
    df['is_delayed15'] = (df['arrival_delay_minutes'] > 15).astype(int)

    cat = [c for c in ['Airline','Origin','Dest'] if c in df.columns]
    num = [c for c in ['Distance','ScheduledElapsedTime','TaxiOut','TaxiIn','dow','dom','month'] if c in df.columns]
    X = df[cat + num]; y = df['arrival_delay_minutes']

    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = make_regression_pipeline(cat, num)
    pipe.fit(X_train, y_train)
    cal_pred = pipe.predict(X_cal)
    q = conformal_interval(y_cal - cal_pred, alpha=args.alpha)

    # Example: predict & output prediction intervals on the calibration set
    lower = cal_pred - q
    upper = cal_pred + q
    out = pd.DataFrame({'y_true': y_cal, 'y_pred': cal_pred, 'pi_lower': lower, 'pi_upper': upper})
    out.to_csv('reports/conformal_calibration.csv', index=False)
    print(f'Conformal absolute interval half-width (alpha={args.alpha}): {q:.2f} minutes')
    print('Saved reports/conformal_calibration.csv')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--alpha', type=float, default=0.1)
    main(p.parse_args())
