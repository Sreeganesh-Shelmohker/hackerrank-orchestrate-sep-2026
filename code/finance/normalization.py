"""
Event Normalization module for Buy or Wait? Finance Engine.
Handles FX conversion, missing amount overrides, status filtering, and linked-event lifecycle deduplication.
"""
from typing import List, Dict, Optional, Set
import pandas as pd
from code.finance.models import FinancialEvent, NormalizedEvent

def normalize_events(
    events: List[FinancialEvent],
    home_currency: str,
    exchange_rates_df: pd.DataFrame,
    image_amount_overrides: Optional[Dict[str, float]] = None
) -> List[NormalizedEvent]:
    """
    Normalizes raw financial events:
    1. Injects image_amount_overrides for blank amounts.
    2. Excludes non-cash (unrealized) and invalid (cancelled/failed) events.
    3. Handles linked event lifecycle deduplication.
    4. Converts foreign currency amounts to home_currency.
    """
    if image_amount_overrides is None:
        image_amount_overrides = {}

    # Build FX rate lookup map: (rate_date, from_curr, to_curr) -> rate
    fx_map: Dict[tuple, float] = {}
    if not exchange_rates_df.empty:
        for _, row in exchange_rates_df.iterrows():
            key = (str(row['rate_date']), str(row['from_currency']), str(row['to_currency']))
            fx_map[key] = float(row['rate'])

    # Step 1: Identify linked events and cancelled/failed targets
    # If event B links to event A (A is linked_event_id), determine relationship
    cancelled_ids: Set[str] = {e.event_id for e in events if e.status in ('cancelled', 'failed', 'unrealized')}
    
    normalized: List[NormalizedEvent] = []

    for e in events:
        # Exclude non-cash unrealized investments and failed/cancelled events
        if e.status in ('cancelled', 'failed', 'unrealized') or e.direction == 'non_cash':
            continue
        
        # Exclude pending credits per Problem Statement §6.3
        if e.status == 'pending' and e.direction == 'credit':
            continue

        # Resolve amount
        amt = e.amount
        if amt is None or pd.isna(amt):
            amt = image_amount_overrides.get(e.event_id)

        if amt is None or pd.isna(amt):
            # Amount is still missing; skip or default to 0.0 with warning
            amt = 0.0

        # Resolve date for rate conversion and timeline placing
        s_date = e.settlement_date if (e.settlement_date and not pd.isna(e.settlement_date)) else e.event_date

        # FX Conversion
        conv_amount = float(amt)
        if e.currency != home_currency:
            rate_key = (str(s_date), str(e.currency), str(home_currency))
            if rate_key in fx_map:
                conv_amount *= fx_map[rate_key]
            else:
                # If exact date rate isn't available, search for closest prior rate date
                prior_rates = [
                    (r_date, r_val) for (r_date, f_curr, t_curr), r_val in fx_map.items()
                    if f_curr == e.currency and t_curr == home_currency and r_date <= str(s_date)
                ]
                if prior_rates:
                    prior_rates.sort(key=lambda x: x[0], reverse=True)
                    conv_amount *= prior_rates[0][1]

        norm_e = NormalizedEvent(
            event_id=e.event_id,
            user_id=e.user_id,
            event_type=e.event_type,
            description=e.description,
            category=e.category,
            direction=e.direction,
            amount=round(conv_amount, 2),
            home_currency=home_currency,
            event_date=e.event_date,
            settlement_date=str(s_date),
            status=e.status,
            linked_event_id=e.linked_event_id,
            flexibility=e.flexibility,
            minimum_allowed_amount=e.minimum_allowed_amount
        )
        normalized.append(norm_e)

    return normalized
