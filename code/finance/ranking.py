"""
6-Tier Plan Ranking & Recommendation Selection module for Buy or Wait? Finance Engine.
Evaluates candidate plans against user preferences and competition ranking rules.
"""
from typing import List, Dict, Optional, Tuple
import datetime
from code.finance.models import (
    Request, FinancialProfile, PaymentOption, CandidatePlan, FinancialState, SpendingChangeAction
)
from code.finance.affordability import calculate_amount_safe_to_pay, find_earliest_date_for_full_payment
from code.finance.forecast import simulate_90_day_forecast
from code.finance.spending_changes import find_optimal_spending_changes

def evaluate_and_rank_plans(
    request: Request,
    profile: FinancialProfile,
    state: FinancialState,
    payment_options: List[PaymentOption]
) -> CandidatePlan:
    """
    Evaluates all eligible candidate plans and selects the top-ranked recommendation
    per Competition §6.3 / §189 rules.
    """
    candidates: List[CandidatePlan] = []
    
    considered_methods = set(profile.payment_methods_user_will_consider)

    # 1. Baseline Safe Amount & Earliest Full Payment Date
    safe_today = calculate_amount_safe_to_pay(state, request.requested_amount, spending_overrides={})
    earliest_full_date = find_earliest_date_for_full_payment(state, request.requested_amount, spending_overrides={})

    # Helper date check
    def is_on_or_before_deadline(dt_str: Optional[str]) -> bool:
        if not dt_str:
            return False
        return dt_str <= request.desired_completion_date

    # -------------------------------------------------------------
    # CANDIDATE METHOD 1: full_payment today
    # -------------------------------------------------------------
    if 'full_payment' in considered_methods:
        # Test baseline full payment today
        pmt_today = {request.request_date: request.requested_amount}
        spending_actions, overrides, is_possible = find_optimal_spending_changes(state, profile, pmt_today)
        
        if is_possible and is_on_or_before_deadline(request.request_date):
            num_changes = len(spending_actions)
            changes_str = "|".join([a.to_string() for a in spending_actions]) if spending_actions else "none"
            aff_status = "affordable_now" if num_changes == 0 else "affordable_with_plan"
            
            amt_str = f"{request.requested_amount:.2f}".rstrip('0').rstrip('.')
            plan_str = f"{request.request_date}:{amt_str}"
            
            expl = f"Pay {profile.home_currency} {request.requested_amount:,.2f} today."
            if num_changes > 0:
                expl = f"Apply spending changes ({changes_str}), then pay {profile.home_currency} {request.requested_amount:,.2f} today."
            
            c_plan = CandidatePlan(
                method='full_payment',
                affordability_status=aff_status,
                payment_plan_str=plan_str,
                earliest_date_for_full_payment=request.request_date if aff_status == "affordable_now" else earliest_full_date,
                spending_changes_str=changes_str,
                spending_changes=spending_actions,
                total_paid=request.requested_amount,
                start_date=request.request_date,
                num_payments=1,
                payment_option_id=None,
                explanation=expl
            )
            candidates.append(c_plan)

    # -------------------------------------------------------------
    # CANDIDATE METHOD 2: partial_payment
    # -------------------------------------------------------------
    if 'partial_payment' in considered_methods and request.allows_partial_payment:
        if 0 < safe_today < request.requested_amount and earliest_full_date and is_on_or_before_deadline(earliest_full_date):
            rem_amt = request.requested_amount - safe_today
            pmt_schedule = {
                request.request_date: safe_today,
                earliest_full_date: rem_amt
            }
            spending_actions, overrides, is_possible = find_optimal_spending_changes(state, profile, pmt_schedule)
            
            if is_possible:
                changes_str = "|".join([a.to_string() for a in spending_actions]) if spending_actions else "none"
                safe_str = f"{safe_today:.2f}".rstrip('0').rstrip('.')
                rem_str = f"{rem_amt:.2f}".rstrip('0').rstrip('.')
                plan_str = f"{request.request_date}:{safe_str}|{earliest_full_date}:{rem_str}"
                
                expl = f"Pay {profile.home_currency} {safe_today:,.2f} today and remaining {profile.home_currency} {rem_amt:,.2f} on {earliest_full_date}."
                
                c_plan = CandidatePlan(
                    method='partial_payment',
                    affordability_status='affordable_with_plan',
                    payment_plan_str=plan_str,
                    earliest_date_for_full_payment=earliest_full_date,
                    spending_changes_str=changes_str,
                    spending_changes=spending_actions,
                    total_paid=request.requested_amount,
                    start_date=request.request_date,
                    num_payments=2,
                    payment_option_id=None,
                    explanation=expl
                )
                candidates.append(c_plan)

    # -------------------------------------------------------------
    # CANDIDATE METHOD 3: installments
    # -------------------------------------------------------------
    if 'installments' in considered_methods and profile.max_installment_months is not None:
        max_months = profile.max_installment_months
        
        for opt in payment_options:
            if opt.payment_method != 'installments':
                continue
            if opt.number_of_payments > max_months * 2:  # Safe check on payment count vs max months
                continue
                
            # Build payment schedule from option
            pmt_schedule: Dict[str, float] = {}
            first_dt = datetime.datetime.strptime(opt.first_payment_date, "%Y-%m-%d")
            freq_days = int(opt.payment_frequency_days) if opt.payment_frequency_days else 30
            
            curr_pmt_dt = first_dt
            for n in range(opt.number_of_payments):
                d_s = curr_pmt_dt.strftime("%Y-%m-%d")
                pmt_schedule[d_s] = pmt_schedule.get(d_s, 0.0) + opt.payment_amount
                curr_pmt_dt += datetime.timedelta(days=freq_days)

            completion_dt_str = (curr_pmt_dt - datetime.timedelta(days=freq_days)).strftime("%Y-%m-%d")

            # Check if completes by desired completion date
            if not is_on_or_before_deadline(completion_dt_str):
                continue

            spending_actions, overrides, is_possible = find_optimal_spending_changes(state, profile, pmt_schedule)
            if is_possible:
                changes_str = "|".join([a.to_string() for a in spending_actions]) if spending_actions else "none"
                
                # Format plan string
                plan_parts = []
                for p_dt, p_amt in sorted(pmt_schedule.items()):
                    p_amt_str = f"{p_amt:.2f}".rstrip('0').rstrip('.')
                    plan_parts.append(f"{p_dt}:{p_amt_str}")
                plan_str = "|".join(plan_parts)

                expl = f"Use {opt.number_of_payments} installments of {profile.home_currency} {opt.payment_amount:,.2f}, starting {opt.first_payment_date}."

                c_plan = CandidatePlan(
                    method='installments',
                    affordability_status='affordable_with_plan',
                    payment_plan_str=plan_str,
                    earliest_date_for_full_payment=earliest_full_date,
                    spending_changes_str=changes_str,
                    spending_changes=spending_actions,
                    total_paid=opt.total_payable_amount,
                    start_date=opt.first_payment_date,
                    num_payments=opt.number_of_payments,
                    payment_option_id=opt.payment_option_id,
                    explanation=expl
                )
                candidates.append(c_plan)

    # -------------------------------------------------------------
    # CANDIDATE METHOD 4: wait
    # -------------------------------------------------------------
    if 'full_payment' in considered_methods and earliest_full_date and earliest_full_date > request.request_date:
        if is_on_or_before_deadline(earliest_full_date):
            pmt_wait = {earliest_full_date: request.requested_amount}
            spending_actions, overrides, is_possible = find_optimal_spending_changes(state, profile, pmt_wait)
            
            if is_possible:
                changes_str = "|".join([a.to_string() for a in spending_actions]) if spending_actions else "none"
                amt_str = f"{request.requested_amount:.2f}".rstrip('0').rstrip('.')
                plan_str = f"{earliest_full_date}:{amt_str}"
                
                expl = f"Pay {profile.home_currency} {request.requested_amount:,.2f} in full on {earliest_full_date}."
                
                c_plan = CandidatePlan(
                    method='wait',
                    affordability_status='affordable_later',
                    payment_plan_str=plan_str,
                    earliest_date_for_full_payment=earliest_full_date,
                    spending_changes_str=changes_str,
                    spending_changes=spending_actions,
                    total_paid=request.requested_amount,
                    start_date=earliest_full_date,
                    num_payments=1,
                    payment_option_id=None,
                    explanation=expl
                )
                candidates.append(c_plan)

    # -------------------------------------------------------------
    # 6-TIER RANKING OF CANDIDATES (§189)
    # -------------------------------------------------------------
    if not candidates:
        # Fallback: not_recommended
        expl = f"Do not proceed with the {profile.home_currency} {request.requested_amount:,.2f} request. None of the available options keeps the required minimum balance protected."
        return CandidatePlan(
            method='not_recommended',
            affordability_status='not_affordable',
            payment_plan_str='none',
            earliest_date_for_full_payment="",
            spending_changes_str='none',
            spending_changes=[],
            total_paid=0.0,
            start_date='9999-12-31',
            num_payments=0,
            payment_option_id=None,
            explanation=expl
        )

    def ranking_key(plan: CandidatePlan) -> Tuple:
        # Tier 1: Complete full request by desired_completion_date (All candidates in list satisfy this)
        # Tier 2: Require no spending changes (0 changes first)
        num_changes = len(plan.spending_changes)
        
        # Tier 3: Minimize total amount paid
        tot_paid = plan.total_paid
        
        # Tier 4: Start payment earlier
        st_date = plan.start_date
        
        # Tier 5: Use fewer payments
        n_pmts = plan.num_payments
        
        # Tier 6: Lowest payment_option_id as tie-breaker
        opt_id = plan.payment_option_id if plan.payment_option_id else "zzzzzz"

        return (num_changes, tot_paid, st_date, n_pmts, opt_id)

    candidates.sort(key=ranking_key)
    return candidates[0]
