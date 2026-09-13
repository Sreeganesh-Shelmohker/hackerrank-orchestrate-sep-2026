"""
Affordability & Timeline Search module for Buy or Wait? Finance Engine.
Calculates amount_safe_to_pay and searches for earliest_date_for_full_payment.
"""
from typing import Optional, Dict, List, Tuple
import datetime
from code.finance.models import FinancialState
from code.finance.forecast import simulate_90_day_forecast

def calculate_amount_safe_to_pay(
    state: FinancialState,
    requested_amount: float,
    spending_overrides: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculates the largest single amount safe to pay on request_date before optional spending changes,
    enforcing that future unreceived salary credits cannot fund purchases made today.
    """
    if spending_overrides is None:
        spending_overrides = {}

    req_dt = datetime.datetime.strptime(state.request_date, "%Y-%m-%d")

    # 1. Starting Available Headroom
    headroom = state.starting_available_balance - state.minimum_balance_to_keep
    if headroom <= 0:
        return 0.0

    # 2. Next confirmed salary credit date S > req_dt
    income_dts: List[datetime.datetime] = []
    for ev in state.scheduled_future_events:
        if (ev.direction == 'credit' or ev.event_type == 'income') and ev.settlement_date > state.request_date:
            try:
                income_dts.append(datetime.datetime.strptime(ev.settlement_date, "%Y-%m-%d"))
            except ValueError:
                pass

    for s in state.recurring_income_streams:
        if s.frequency_days == 30:
            d = req_dt.day
            if s.day_of_month > d:
                try:
                    income_dts.append(req_dt.replace(day=s.day_of_month))
                except ValueError:
                    pass
            else:
                m = req_dt.month % 12 + 1
                y = req_dt.year + (1 if m == 1 else 0)
                try:
                    income_dts.append(datetime.datetime(y, m, min(s.day_of_month, 28)))
                except ValueError:
                    pass
        elif s.frequency_days != 30 and s.last_settlement_date:
            try:
                last_dt = datetime.datetime.strptime(s.last_settlement_date, "%Y-%m-%d")
                next_dt = last_dt + datetime.timedelta(days=s.frequency_days)
                while next_dt <= req_dt:
                    next_dt += datetime.timedelta(days=s.frequency_days)
                income_dts.append(next_dt)
            except ValueError:
                pass

    next_inc = min(income_dts) if income_dts else req_dt + datetime.timedelta(days=90)
    next_inc_str = next_inc.strftime("%Y-%m-%d")

    # 3. Mandatory debits occurring after T but before or on next confirmed salary credit S
    sch_debits = sum(
        ev.amount for ev in state.scheduled_future_events
        if not (ev.direction == 'credit' or ev.event_type == 'income') and state.request_date < ev.settlement_date <= next_inc_str
    )

    rec_debits = 0.0
    for s in state.recurring_expense_streams:
        amt = spending_overrides.get(s.event_id, s.amount)
        curr = req_dt + datetime.timedelta(days=1)
        while curr <= next_inc:
            occurs_today = False
            if s.frequency_days == 30 and curr.day == s.day_of_month:
                occurs_today = True
            elif s.frequency_days != 30 and s.last_settlement_date:
                try:
                    last_dt = datetime.datetime.strptime(s.last_settlement_date, "%Y-%m-%d")
                    diff_days = (curr - last_dt).days
                    if diff_days > 0 and diff_days % s.frequency_days == 0:
                        occurs_today = True
                except ValueError:
                    pass

            if occurs_today:
                rec_debits += amt

            curr += datetime.timedelta(days=1)

    pre_salary_debits = sch_debits + rec_debits
    safe_today = max(0.0, headroom - pre_salary_debits)

    final_safe = max(0.0, min(requested_amount, safe_today))
    return round(final_safe, 2)

def find_earliest_date_for_full_payment(
    state: FinancialState,
    requested_amount: float,
    spending_overrides: Optional[Dict[str, float]] = None
) -> Optional[str]:
    """
    Searches for the earliest date t in [request_date, request_date + 90 days]
    where paying requested_amount in full on date t satisfies minimum_balance_to_keep on all 90 days.
    If safe_today < requested_amount, candidate search evaluates starting from the next confirmed salary credit date.
    """
    if spending_overrides is None:
        spending_overrides = {}

    safe_today = calculate_amount_safe_to_pay(state, requested_amount, spending_overrides)
    if abs(safe_today - requested_amount) < 1e-2:
        return state.request_date

    req_dt = datetime.datetime.strptime(state.request_date, "%Y-%m-%d")
    end_dt = req_dt + datetime.timedelta(days=90)

    # Find next confirmed salary credit date S > req_dt
    income_dts: List[datetime.datetime] = []
    for ev in state.scheduled_future_events:
        if (ev.direction == 'credit' or ev.event_type == 'income') and ev.settlement_date > state.request_date:
            try:
                income_dts.append(datetime.datetime.strptime(ev.settlement_date, "%Y-%m-%d"))
            except ValueError:
                pass

    for s in state.recurring_income_streams:
        if s.frequency_days == 30:
            d = req_dt.day
            if s.day_of_month > d:
                try:
                    income_dts.append(req_dt.replace(day=s.day_of_month))
                except ValueError:
                    pass
            else:
                m = req_dt.month % 12 + 1
                y = req_dt.year + (1 if m == 1 else 0)
                try:
                    income_dts.append(datetime.datetime(y, m, min(s.day_of_month, 28)))
                except ValueError:
                    pass

    start_search_dt = min(income_dts) if income_dts else req_dt + datetime.timedelta(days=1)

    curr_dt = start_search_dt
    while curr_dt <= end_dt:
        d_str = curr_dt.strftime("%Y-%m-%d")
        test_pmt = {d_str: requested_amount}
        
        _, is_safe, _ = simulate_90_day_forecast(
            state=state,
            proposed_plan_payments=test_pmt,
            spending_overrides=spending_overrides
        )
        if is_safe:
            return d_str

        curr_dt += datetime.timedelta(days=1)

    return None
