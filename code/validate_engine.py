"""
Validation harness script to test Finance Engine against sample_requests.csv.
"""
import sys, os
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

def main():
    print("Loading datasets...")
    sample_df = pd.read_csv('dataset/sample_requests.csv')
    profiles_df = pd.read_csv('dataset/financial_profiles.csv')
    events_df = pd.read_csv('dataset/financial_events.csv')
    exchange_rates_df = pd.read_csv('dataset/exchange_rates.csv')
    payment_options_df = pd.read_csv('dataset/request_payment_options.csv')
    messages_df = pd.read_csv('dataset/messages.csv') if os.path.exists('dataset/messages.csv') else None
    images_df = pd.read_csv('dataset/images.csv') if os.path.exists('dataset/images.csv') else None

    engine = FinanceEngine(
        profiles_df=profiles_df,
        events_df=events_df,
        exchange_rates_df=exchange_rates_df,
        payment_options_df=payment_options_df,
        messages_df=messages_df,
        images_df=images_df,
        image_amount_overrides=IMAGE_AMOUNT_OVERRIDES
    )

    total_samples = len(sample_df)
    matches = {
        'safe_amount': 0,
        'affordability_status': 0,
        'recommended_payment_method': 0,
        'earliest_date': 0,
        'spending_changes': 0
    }

    mismatches = []

    for idx, row in sample_df.iterrows():
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

        # Compare outputs
        gt_safe = float(row['amount_safe_to_pay'])
        gt_aff = str(row['affordability_status'])
        gt_method = str(row['recommended_payment_method'])
        gt_date = str(row['earliest_date_for_full_payment']) if not pd.isna(row['earliest_date_for_full_payment']) else ""
        gt_changes = str(row['spending_changes_needed'])

        m_safe = abs(res.amount_safe_to_pay - gt_safe) < 1.0  # within 1 currency unit allowance for float rounding
        m_aff = (res.affordability_status == gt_aff)
        m_method = (res.recommended_payment_method == gt_method)
        m_date = (res.earliest_date_for_full_payment == gt_date)
        m_changes = (res.spending_changes_needed == gt_changes)

        if m_safe: matches['safe_amount'] += 1
        if m_aff: matches['affordability_status'] += 1
        if m_method: matches['recommended_payment_method'] += 1
        if m_date: matches['earliest_date'] += 1
        if m_changes: matches['spending_changes'] += 1

        if not (m_safe and m_aff and m_method and m_date and m_changes):
            mismatches.append({
                'request_id': req.request_id,
                'pred_safe': res.amount_safe_to_pay, 'gt_safe': gt_safe,
                'pred_aff': res.affordability_status, 'gt_aff': gt_aff,
                'pred_method': res.recommended_payment_method, 'gt_method': gt_method,
                'pred_plan': res.payment_plan, 'gt_plan': str(row['payment_plan']),
                'pred_date': res.earliest_date_for_full_payment, 'gt_date': gt_date,
                'pred_changes': res.spending_changes_needed, 'gt_changes': gt_changes
            })

    print(f"\n--- Validation Summary ({total_samples} samples) ---")
    print(f"amount_safe_to_pay Accuracy: {matches['safe_amount']}/{total_samples} ({matches['safe_amount']/total_samples*100:.1f}%)")
    print(f"affordability_status Accuracy: {matches['affordability_status']}/{total_samples} ({matches['affordability_status']/total_samples*100:.1f}%)")
    print(f"recommended_payment_method Accuracy: {matches['recommended_payment_method']}/{total_samples} ({matches['recommended_payment_method']/total_samples*100:.1f}%)")
    print(f"earliest_date Accuracy: {matches['earliest_date']}/{total_samples} ({matches['earliest_date']/total_samples*100:.1f}%)")
    print(f"spending_changes Accuracy: {matches['spending_changes']}/{total_samples} ({matches['spending_changes']/total_samples*100:.1f}%)")

    if mismatches:
        print(f"\nFound {len(mismatches)} mismatches:")
        for m in mismatches:
            print(f"Request: {m['request_id']}")
            print(f"  Pred Safe: {m['pred_safe']} | GT Safe: {m['gt_safe']}")
            print(f"  Pred Aff: {m['pred_aff']} | GT Aff: {m['gt_aff']}")
            print(f"  Pred Method: {m['pred_method']} | GT Method: {m['gt_method']}")
            print(f"  Pred Plan: {m['pred_plan']} | GT Plan: {m['gt_plan']}")
            print(f"  Pred Date: {m['pred_date']} | GT Date: {m['gt_date']}")
            print(f"  Pred Changes: {m['pred_changes']} | GT Changes: {m['gt_changes']}")
            print('-'*50)

if __name__ == '__main__':
    main()
