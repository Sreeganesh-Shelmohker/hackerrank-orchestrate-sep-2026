"""
State Reconstruction module for Buy or Wait? Finance Engine.
Reconstructs user financial state on request_date from profile balance, pending debits,
active recurring commitment streams, and scheduled future events.
"""
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import datetime
from code.finance.models import (
    FinancialProfile, NormalizedEvent, RecurringStream, FinancialState
)

# Descriptions of income events that represent one-off, final, or non-recurring payouts
NON_RECURRING_INCOME_KEYWORDS: Set[str] = {
    'final employer payroll',
    'previous employer payroll',
    'prorated first salary',
    'promotion arrears payment',
    'quarterly performance bonus',
    'prize proceeds',
    'august 2019 net salary',
    'temporary assignment pay',
    'peak-season wages',
    'website project payment',
    'freelance milestone payment',
    'consulting invoice payment',
    'independent work payment',
    'application project payment'
}

# Categories that represent recurring standing financial commitments
RECURRING_COMMITMENT_CATEGORIES: Set[str] = {
    'rent', 'utilities', 'housing', 'education', 'loan', 'debt_repayment',
    'healthcare', 'family_support', 'subscription', 'cloud_storage',
    'streaming', 'insurance', 'mortgage', 'delivery_membership', 'music_subscription'
}

def build_financial_state(
    profile: FinancialProfile,
    normalized_events: List[NormalizedEvent],
    request_date: str
) -> FinancialState:
    """
    Reconstructs FinancialState as of request_date:
    1. Reserves pending debits against profile.current_available_balance.
    2. Identifies active recurring commitment streams (rent, utilities, debt payments, subscriptions, healthcare, family support, flexible items).
    3. Identifies future scheduled events (e.g. next confirmed salary).
    """
    dt_req = datetime.datetime.strptime(request_date, "%Y-%m-%d")
    dt_cutoff = dt_req - datetime.timedelta(days=45)
    cutoff_date_str = dt_cutoff.strftime("%Y-%m-%d")

    # 1. Pending debits on or before request_date
    pending_debits = [
        e for e in normalized_events
        if e.status == 'pending' and e.direction == 'debit' and e.event_date <= request_date
    ]
    pending_debits_total = sum(e.amount for e in pending_debits)
    starting_available_balance = profile.current_available_balance - pending_debits_total

    # 2. Historical settled events on or before request_date
    historical_events = [
        e for e in normalized_events
        if e.status == 'settled' and e.settlement_date <= request_date
    ]

    # 3. Future scheduled events after request_date
    scheduled_future_events = [
        e for e in normalized_events
        if e.status == 'scheduled' and e.settlement_date > request_date
    ]

    # 4. Extract active recurring streams from historical events
    recurring_expense_streams: List[RecurringStream] = []
    recurring_income_streams: List[RecurringStream] = []

    grouped_income: Dict[Tuple[str, str], List[NormalizedEvent]] = defaultdict(list)
    grouped_expense: Dict[str, List[NormalizedEvent]] = defaultdict(list)

    for e in historical_events:
        if e.direction == 'credit' or e.event_type == 'income':
            grouped_income[(e.category, e.description)].append(e)
        else:
            grouped_expense[e.category].append(e)

    for (cat, desc), ev_list in grouped_income.items():
        ev_list.sort(key=lambda x: x.settlement_date)
        latest_ev = ev_list[-1]
        if latest_ev.settlement_date < cutoff_date_str:
            continue

        desc_lower = desc.strip().lower()
        flex = latest_ev.flexibility if (latest_ev.flexibility and latest_ev.flexibility.strip()) else 'fixed'

        is_non_recurring = any(kw in desc_lower for kw in NON_RECURRING_INCOME_KEYWORDS)
        if not is_non_recurring:
            try:
                dt = datetime.datetime.strptime(latest_ev.settlement_date, "%Y-%m-%d")
                day_of_month = dt.day
            except ValueError:
                day_of_month = 15

            stream = RecurringStream(
                event_id=latest_ev.event_id,
                category=cat,
                description=desc,
                amount=latest_ev.amount,
                day_of_month=day_of_month,
                flexibility=flex,
                minimum_allowed_amount=latest_ev.minimum_allowed_amount,
                is_income=True,
                frequency_days=30,
                last_settlement_date=latest_ev.settlement_date
            )
            recurring_income_streams.append(stream)

    for cat, ev_list in grouped_expense.items():
        ev_list.sort(key=lambda x: x.settlement_date)
        latest_ev = ev_list[-1]

        if latest_ev.settlement_date < cutoff_date_str:
            continue

        flex = latest_ev.flexibility if (latest_ev.flexibility and latest_ev.flexibility.strip()) else 'fixed'

        is_commitment = (
            latest_ev.event_type in ('subscription', 'debt_payment') or
            cat in RECURRING_COMMITMENT_CATEGORIES
        )

        is_recurring_expense = is_commitment or (len(ev_list) >= 2)

        if is_recurring_expense:
            freq_days = 30
            if len(ev_list) >= 2:
                dts = []
                for e in ev_list:
                    try:
                        dts.append(datetime.datetime.strptime(e.settlement_date, "%Y-%m-%d"))
                    except ValueError:
                        pass
                intervals = [(dts[i+1] - dts[i]).days for i in range(len(dts)-1) if (dts[i+1] - dts[i]).days > 0]
                if intervals:
                    recent = intervals[-3:]
                    avg_int = sum(recent) / len(recent)
                    if avg_int <= 8.5:
                        freq_days = 7
                    elif avg_int <= 12.0:
                        freq_days = 10
                    elif avg_int <= 17.5:
                        freq_days = 14
                    elif avg_int <= 25.0:
                        freq_days = 21
                    else:
                        freq_days = 30

            try:
                dt = datetime.datetime.strptime(latest_ev.settlement_date, "%Y-%m-%d")
                day_of_month = dt.day
            except ValueError:
                day_of_month = 15

            stream = RecurringStream(
                event_id=latest_ev.event_id,
                category=cat,
                description=latest_ev.description,
                amount=latest_ev.amount,
                day_of_month=day_of_month,
                flexibility=flex,
                minimum_allowed_amount=latest_ev.minimum_allowed_amount,
                is_income=False,
                frequency_days=freq_days,
                last_settlement_date=latest_ev.settlement_date
            )
            recurring_expense_streams.append(stream)

    return FinancialState(
        user_id=profile.user_id,
        home_currency=profile.home_currency,
        request_date=request_date,
        starting_available_balance=starting_available_balance,
        minimum_balance_to_keep=profile.minimum_balance_to_keep,
        pending_debits_total=pending_debits_total,
        recurring_expense_streams=recurring_expense_streams,
        recurring_income_streams=recurring_income_streams,
        scheduled_future_events=scheduled_future_events
    )
