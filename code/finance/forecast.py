"""
90-Day Balance Forecast Simulator module for Buy or Wait? Finance Engine.
Simulates daily balance progression and checks minimum balance safety over the 90-day window.
"""
from typing import List, Dict, Tuple, Optional
import datetime
from code.finance.models import FinancialState, ForecastEntry, RecurringStream, CandidatePlan

def simulate_90_day_forecast(
    state: FinancialState,
    proposed_plan_payments: Optional[Dict[str, float]] = None,
    spending_overrides: Optional[Dict[str, float]] = None
) -> Tuple[List[ForecastEntry], bool, float]:
    """
    Simulates daily balance progression for request_date to request_date + 90 days.
    
    Arguments:
    - state: FinancialState constructed as of request_date
    - proposed_plan_payments: Dict mapping date string 'YYYY-MM-DD' -> payment amount
    - spending_overrides: Dict mapping event_id -> new amount (0.0 for stop, or reduced amount)

    Returns:
    - timeline: List[ForecastEntry]
    - is_safe: bool (True if balance >= minimum_balance_to_keep on all 90 days)
    - min_surplus: float (lowest balance - minimum_balance_to_keep observed)
    """
    if proposed_plan_payments is None:
        proposed_plan_payments = {}
    if spending_overrides is None:
        spending_overrides = {}

    req_dt = datetime.datetime.strptime(state.request_date, "%Y-%m-%d")
    end_dt = req_dt + datetime.timedelta(days=90)

    # Build scheduled future cash flows map: date -> income, expense
    scheduled_income: Dict[str, float] = {}
    scheduled_expense: Dict[str, float] = {}

    for ev in state.scheduled_future_events:
        d_str = ev.settlement_date
        if ev.direction == 'credit' or ev.event_type == 'income':
            scheduled_income[d_str] = scheduled_income.get(d_str, 0.0) + ev.amount
        else:
            scheduled_expense[d_str] = scheduled_expense.get(d_str, 0.0) + ev.amount

    # Build next trigger date maps for recurring streams
    next_income_dates: Dict[str, datetime.datetime] = {}
    for s in state.recurring_income_streams:
        if s.frequency_days != 30 and s.last_settlement_date:
            last_dt = datetime.datetime.strptime(s.last_settlement_date, "%Y-%m-%d")
            next_dt = last_dt + datetime.timedelta(days=s.frequency_days)
            while next_dt < req_dt:
                next_dt += datetime.timedelta(days=s.frequency_days)
            next_income_dates[s.event_id] = next_dt

    next_expense_dates: Dict[str, datetime.datetime] = {}
    for s in state.recurring_expense_streams:
        if s.frequency_days != 30 and s.last_settlement_date:
            last_dt = datetime.datetime.strptime(s.last_settlement_date, "%Y-%m-%d")
            next_dt = last_dt + datetime.timedelta(days=s.frequency_days)
            while next_dt < req_dt:
                next_dt += datetime.timedelta(days=s.frequency_days)
            next_expense_dates[s.event_id] = next_dt

    timeline: List[ForecastEntry] = []
    curr_balance = state.starting_available_balance
    is_safe = True
    min_surplus = float('inf')

    curr_dt = req_dt
    while curr_dt <= end_dt:
        d_str = curr_dt.strftime("%Y-%m-%d")
        day_num = curr_dt.day

        # 1. Calculate Recurring Income for today
        rec_income_today = 0.0
        for s in state.recurring_income_streams:
            if s.frequency_days == 30:
                if s.day_of_month == day_num:
                    rec_income_today += s.amount
            else:
                if s.event_id in next_income_dates and next_income_dates[s.event_id] == curr_dt:
                    rec_income_today += s.amount
                    next_income_dates[s.event_id] += datetime.timedelta(days=s.frequency_days)

        sch_income_today = scheduled_income.get(d_str, 0.0)
        total_income_today = rec_income_today + sch_income_today

        # 2. Calculate Recurring Expenses for today
        rec_expense_today = 0.0
        for s in state.recurring_expense_streams:
            is_due = False
            if s.frequency_days == 30 and s.day_of_month == day_num:
                is_due = True
            elif s.frequency_days != 30 and s.event_id in next_expense_dates and next_expense_dates[s.event_id] == curr_dt:
                is_due = True
                next_expense_dates[s.event_id] += datetime.timedelta(days=s.frequency_days)

            if is_due:
                if s.event_id in spending_overrides:
                    rec_expense_today += spending_overrides[s.event_id]
                else:
                    rec_expense_today += s.amount

        sch_expense_today = scheduled_expense.get(d_str, 0.0)
        total_expense_today = rec_expense_today + sch_expense_today

        # 3. Plan payments for today
        plan_pmt_today = proposed_plan_payments.get(d_str, 0.0)

        # 4. Compute EOD balance
        start_of_day_bal = curr_balance
        end_of_day_bal = start_of_day_bal + total_income_today - total_expense_today - plan_pmt_today
        curr_balance = end_of_day_bal

        surplus = end_of_day_bal - state.minimum_balance_to_keep
        if surplus < min_surplus:
            min_surplus = surplus

        day_is_safe = (end_of_day_bal >= state.minimum_balance_to_keep - 1e-4)
        if not day_is_safe:
            is_safe = False

        entry = ForecastEntry(
            date=d_str,
            starting_balance=round(start_of_day_bal, 2),
            income=round(total_income_today, 2),
            expenses=round(total_expense_today, 2),
            plan_payments=round(plan_pmt_today, 2),
            ending_balance=round(end_of_day_bal, 2),
            is_safe=day_is_safe
        )
        timeline.append(entry)

        curr_dt += datetime.timedelta(days=1)

    return timeline, is_safe, round(min_surplus, 2)
