"""
Spending Changes Optimization module for Buy or Wait? Finance Engine.
Generates and optimizes combinations of up to 3 spending change actions
(stop:<event_id> or reduce_to:<event_id>:<new_amount>).
"""
from typing import List, Dict, Tuple, Optional, Set
import itertools
from code.finance.models import (
    FinancialState, FinancialProfile, SpendingChangeAction, RecurringStream
)
from code.finance.forecast import simulate_90_day_forecast

def get_candidate_spending_actions(
    state: FinancialState,
    profile: FinancialProfile
) -> List[Tuple[SpendingChangeAction, float, float]]:
    """
    Returns candidate spending change actions for the user's active recurring streams:
    List of (SpendingChangeAction, monthly_savings, original_amount)
    """
    candidates: List[Tuple[SpendingChangeAction, float, float]] = []

    stop_cats = set(profile.expense_categories_user_is_willing_to_stop) if profile.expense_categories_user_is_willing_to_stop else set()
    reduce_cats = set(profile.expense_categories_user_is_willing_to_reduce) if profile.expense_categories_user_is_willing_to_reduce else set()

    for stream in state.recurring_expense_streams:
        if stream.is_income:
            continue

        flex = stream.flexibility
        cat = stream.category

        # 1. Candidate 'stop' action
        if flex in ('stoppable', 'reducible_or_stoppable') and cat in stop_cats:
            action = SpendingChangeAction(action_type='stop', event_id=stream.event_id)
            monthly_savings = stream.amount
            candidates.append((action, monthly_savings, stream.amount))

        # 2. Candidate 'reduce_to' action
        if flex in ('reducible', 'reducible_or_stoppable') and cat in reduce_cats:
            min_amt = stream.minimum_allowed_amount
            if min_amt is not None and min_amt < stream.amount:
                action = SpendingChangeAction(
                    action_type='reduce_to',
                    event_id=stream.event_id,
                    new_amount=min_amt
                )
                monthly_savings = stream.amount - min_amt
                candidates.append((action, monthly_savings, stream.amount))

    return candidates

def find_optimal_spending_changes(
    state: FinancialState,
    profile: FinancialProfile,
    proposed_plan_payments: Dict[str, float]
) -> Tuple[List[SpendingChangeAction], Dict[str, float], bool]:
    """
    Finds the optimal set of up to 3 spending change actions that makes proposed_plan_payments safe.
    
    Returns:
    - best_actions: List[SpendingChangeAction]
    - spending_overrides: Dict[event_id -> new_amount]
    - is_possible: bool
    """
    # 0. Test if plan is safe with 0 spending changes
    _, is_safe, _ = simulate_90_day_forecast(state, proposed_plan_payments, spending_overrides={})
    if is_safe:
        return [], {}, True

    candidates = get_candidate_spending_actions(state, profile)
    if not candidates:
        return [], {}, False

    # Evaluate combinations of size 1, 2, 3
    for k in (1, 2, 3):
        valid_combos = []
        for combo in itertools.combinations(candidates, k):
            # Check mutual exclusivity: one action per event_id
            event_ids = [act.event_id for act, _, _ in combo]
            if len(set(event_ids)) < len(event_ids):
                continue

            # Build spending overrides map
            overrides: Dict[str, float] = {}
            total_savings = 0.0
            for act, savings, _ in combo:
                total_savings += savings
                if act.action_type == 'stop':
                    overrides[act.event_id] = 0.0
                elif act.action_type == 'reduce_to':
                    overrides[act.event_id] = act.new_amount if act.new_amount is not None else 0.0

            # Test safety with this override combo
            _, safe, _ = simulate_90_day_forecast(state, proposed_plan_payments, overrides)
            if safe:
                valid_combos.append((combo, overrides, total_savings))

        if valid_combos:
            # Sort valid combos by implementation heuristic: minimize total monthly savings impact
            valid_combos.sort(key=lambda x: x[2])
            best_combo, best_overrides, _ = valid_combos[0]
            best_actions = [act for act, _, _ in best_combo]
            return best_actions, best_overrides, True

    return [], {}, False
