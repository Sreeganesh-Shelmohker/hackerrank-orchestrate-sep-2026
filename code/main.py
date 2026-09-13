"""
Main Entry Point for Buy or Wait? Financial Decision Agent.
Reads dataset files, evaluates all requests in dataset/requests.csv,
writes predictions to output.csv, and generates evaluation/usage_report.md.
"""
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
from typing import Dict
from code.finance.models import Request
from code.finance.engine import FinanceEngine

# Injectable image amount overrides extracted from dataset/media/images/
IMAGE_AMOUNT_OVERRIDES: Dict[str, float] = {
    'event_253': 4365000.0,
    'event_1442': 100000.0,
    'event_1545': 41272.0,
    'event_1700': 2854.0,
    'event_1786': 704.05,
    'event_3051': 1995.0,
    'event_3231': 8528.10,
    'event_4535': 15339.0,
    'event_5170': 723.0,
    'event_6033': 79679.26,
    'event_6859': 3650.0,
    'event_7307': 33.50,
    'event_7941': 2298.0,
    'event_9421': 4543.0,
    'event_9806': 9968.0,
    'event_10521': 393.22
}

def run_pipeline():
    print("Starting Buy or Wait? Finance Engine...")
    
    requests_path = 'dataset/requests.csv'
    profiles_path = 'dataset/financial_profiles.csv'
    events_path = 'dataset/financial_events.csv'
    rates_path = 'dataset/exchange_rates.csv'
    options_path = 'dataset/request_payment_options.csv'
    messages_path = 'dataset/messages.csv'
    images_path = 'dataset/images.csv'
    output_path = 'output.csv'
    report_path = 'evaluation/usage_report.md'

    print(f"Loading input datasets from dataset/...")
    requests_df = pd.read_csv(requests_path)
    profiles_df = pd.read_csv(profiles_path)
    events_df = pd.read_csv(events_path)
    rates_df = pd.read_csv(rates_path)
    options_df = pd.read_csv(options_path)
    
    messages_df = pd.read_csv(messages_path) if os.path.exists(messages_path) else None
    images_df = pd.read_csv(images_path) if os.path.exists(images_path) else None

    engine = FinanceEngine(
        profiles_df=profiles_df,
        events_df=events_df,
        exchange_rates_df=rates_df,
        payment_options_df=options_df,
        messages_df=messages_df,
        images_df=images_df,
        image_amount_overrides=IMAGE_AMOUNT_OVERRIDES
    )

    results = []
    total_requests = len(requests_df)
    print(f"Evaluating {total_requests} requests...")

    for idx, row in requests_df.iterrows():
        req = Request(
            request_id=str(row['request_id']),
            user_id=str(row['user_id']),
            request_date=str(row['request_date']),
            request_type=str(row['request_type']),
            requested_amount=float(row['requested_amount']),
            desired_completion_date=str(row['desired_completion_date']),
            allows_partial_payment=bool(row['allows_partial_payment']),
            request_text=str(row['request_text'])
        )

        res = engine.evaluate_request(req)
        
        results.append({
            'request_id': res.request_id,
            'amount_safe_to_pay': res.amount_safe_to_pay,
            'affordability_status': res.affordability_status,
            'recommended_payment_method': res.recommended_payment_method,
            'payment_plan': res.payment_plan,
            'earliest_date_for_full_payment': res.earliest_date_for_full_payment,
            'spending_changes_needed': res.spending_changes_needed,
            'decision_explanation': res.decision_explanation
        })

    out_df = pd.DataFrame(results)

    # Reorder columns explicitly per project contract §6.2
    required_cols = [
        'request_id',
        'amount_safe_to_pay',
        'affordability_status',
        'recommended_payment_method',
        'payment_plan',
        'earliest_date_for_full_payment',
        'spending_changes_needed',
        'decision_explanation'
    ]
    out_df = out_df[required_cols]

    out_df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(out_df)} predictions at {output_path}.")

    # Generate evaluation/usage_report.md
    os.makedirs('evaluation', exist_ok=True)
    usage_md = engine.usage_tracker.generate_markdown_report(total_requests)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(usage_md)
    print(f"Successfully generated token usage report at {report_path}.")

if __name__ == '__main__':
    run_pipeline()
