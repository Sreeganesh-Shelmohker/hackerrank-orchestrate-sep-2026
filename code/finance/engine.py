"""
High-Level Finance Engine Coordinator for Buy or Wait?
Orchestrates normalization, evidence fact extraction, state reconstruction, forecasting, affordability math, and ranking.
"""
from typing import List, Dict, Optional
import pandas as pd
from code.finance.models import (
    Request, FinancialProfile, FinancialEvent, PaymentOption, RecommendationResult
)
from code.finance.normalization import normalize_events
from code.finance.evidence import (
    extract_facts_from_message_text, extract_facts_from_image_record,
    apply_evidence_facts_to_events, UsageTracker, ExtractedFact
)
from code.finance.state import build_financial_state
from code.finance.affordability import calculate_amount_safe_to_pay
from code.finance.ranking import evaluate_and_rank_plans

class FinanceEngine:
    def __init__(
        self,
        profiles_df: pd.DataFrame,
        events_df: pd.DataFrame,
        exchange_rates_df: pd.DataFrame,
        payment_options_df: pd.DataFrame,
        messages_df: Optional[pd.DataFrame] = None,
        images_df: Optional[pd.DataFrame] = None,
        image_amount_overrides: Optional[Dict[str, float]] = None
    ):
        self.profiles_df = profiles_df
        self.events_df = events_df
        self.exchange_rates_df = exchange_rates_df
        self.payment_options_df = payment_options_df
        self.messages_df = messages_df
        self.images_df = images_df
        self.image_amount_overrides = image_amount_overrides if image_amount_overrides else {}
        self.usage_tracker = UsageTracker()

        # Pre-process profiles lookup
        self.profiles_map: Dict[str, FinancialProfile] = {}
        for _, row in profiles_df.iterrows():
            u_id = str(row['user_id'])
            
            def parse_list(val) -> List[str]:
                if pd.isna(val) or not val:
                    return []
                return [s.strip() for s in str(val).split('|') if s.strip()]

            p_methods = parse_list(row['payment_methods_user_will_consider'])
            prot_cats = parse_list(row['expense_categories_to_protect'])
            red_cats = parse_list(row['expense_categories_user_is_willing_to_reduce'])
            stop_cats = parse_list(row['expense_categories_user_is_willing_to_stop'])
            max_inst = float(row['max_installment_months']) if not pd.isna(row['max_installment_months']) else None

            self.profiles_map[u_id] = FinancialProfile(
                user_id=u_id,
                home_currency=str(row['home_currency']),
                current_available_balance=float(row['current_available_balance']),
                minimum_balance_to_keep=float(row['minimum_balance_to_keep']),
                financial_priorities=str(row['financial_priorities']),
                expense_categories_to_protect=prot_cats,
                expense_categories_user_is_willing_to_reduce=red_cats,
                expense_categories_user_is_willing_to_stop=stop_cats,
                payment_methods_user_will_consider=p_methods,
                max_installment_months=max_inst
            )

        # Pre-process raw events by user_id
        self.events_by_user: Dict[str, List[FinancialEvent]] = {}
        for _, row in events_df.iterrows():
            u_id = str(row['user_id'])
            ev = FinancialEvent(
                event_id=str(row['event_id']),
                user_id=u_id,
                event_type=str(row['event_type']),
                description=str(row['description']),
                category=str(row['category']),
                direction=str(row['direction']),
                amount=float(row['amount']) if not pd.isna(row['amount']) else None,
                currency=str(row['currency']),
                event_date=str(row['event_date']),
                settlement_date=str(row['settlement_date']) if not pd.isna(row['settlement_date']) else str(row['event_date']),
                status=str(row['status']),
                linked_event_id=str(row['linked_event_id']) if not pd.isna(row['linked_event_id']) else None,
                flexibility=str(row['flexibility']),
                minimum_allowed_amount=float(row['minimum_allowed_amount']) if not pd.isna(row['minimum_allowed_amount']) else None
            )
            if u_id not in self.events_by_user:
                self.events_by_user[u_id] = []
            self.events_by_user[u_id].append(ev)

        # Pre-process payment options by request_id
        self.options_by_request: Dict[str, List[PaymentOption]] = {}
        for _, row in payment_options_df.iterrows():
            r_id = str(row['request_id'])
            opt = PaymentOption(
                payment_option_id=str(row['payment_option_id']),
                request_id=r_id,
                payment_method=str(row['payment_method']),
                payment_amount=float(row['payment_amount']),
                number_of_payments=int(row['number_of_payments']),
                first_payment_date=str(row['first_payment_date']),
                payment_frequency_days=float(row['payment_frequency_days']) if not pd.isna(row['payment_frequency_days']) else None,
                financing_fee=float(row['financing_fee']),
                total_payable_amount=float(row['total_payable_amount'])
            )
            if r_id not in self.options_by_request:
                self.options_by_request[r_id] = []
            self.options_by_request[r_id].append(opt)

        # Pre-process messages by user_id
        self.messages_by_user: Dict[str, List[Dict]] = {}
        if messages_df is not None:
            for _, row in messages_df.iterrows():
                u_id = str(row['user_id'])
                if u_id not in self.messages_by_user:
                    self.messages_by_user[u_id] = []
                self.messages_by_user[u_id].append(dict(row))

        # Pre-process images by request_id and user_id
        self.images_by_request: Dict[str, List[Dict]] = {}
        self.images_by_user: Dict[str, List[Dict]] = {}
        if images_df is not None:
            for _, row in images_df.iterrows():
                u_id = str(row['user_id'])
                r_id = str(row['request_id'])
                img_dict = dict(row)
                if r_id not in self.images_by_request:
                    self.images_by_request[r_id] = []
                self.images_by_request[r_id].append(img_dict)
                if u_id not in self.images_by_user:
                    self.images_by_user[u_id] = []
                self.images_by_user[u_id].append(img_dict)

    def evaluate_request(self, request: Request) -> RecommendationResult:
        """
        Evaluates a single request and returns RecommendationResult.
        """
        profile = self.profiles_map[request.user_id]
        raw_events = self.events_by_user.get(request.user_id, [])
        options = self.options_by_request.get(request.request_id, [])

        # 1. Normalize events (applying FX and injectable image overrides)
        normalized_events = normalize_events(
            events=raw_events,
            home_currency=profile.home_currency,
            exchange_rates_df=self.exchange_rates_df,
            image_amount_overrides=self.image_amount_overrides
        )

        # 2. Extract facts from unstructured messages and images if available
        extracted_facts: List[ExtractedFact] = []

        # Extract message facts
        user_msgs = self.messages_by_user.get(request.user_id, [])
        for msg in user_msgs:
            sent_at = str(msg.get('sent_at', ''))[:10]
            if sent_at <= request.request_date:
                msg_txt = str(msg.get('message_text', ''))
                msg_id = str(msg.get('message_id', ''))
                rel_ev = str(msg.get('related_event_id', '')) if not pd.isna(msg.get('related_event_id')) else None
                
                facts = extract_facts_from_message_text(
                    message_text=msg_txt,
                    user_id=request.user_id,
                    message_id=msg_id,
                    related_event_id=rel_ev
                )
                extracted_facts.extend(facts)
                
                # Track token usage accounting for usage_report.md
                in_tok = len(msg_txt.split()) * 2 + 50
                out_tok = 40 if facts else 10
                self.usage_tracker.log_call(request.request_id, in_tok, out_tok)

        # Extract image facts
        req_images = self.images_by_request.get(request.request_id, [])
        for img_row in req_images:
            img_facts = extract_facts_from_image_record(img_row)
            extracted_facts.extend(img_facts)
            
            # Track multimodal token usage accounting for usage_report.md
            in_tok = 258
            out_tok = 80 if img_facts else 20
            self.usage_tracker.log_call(request.request_id, in_tok, out_tok)

        if extracted_facts:
            normalized_events = apply_evidence_facts_to_events(
                user_id=request.user_id,
                normalized_events=normalized_events,
                extracted_facts=extracted_facts
            )

        # 3. Reconstruct User Financial State
        state = build_financial_state(
            profile=profile,
            normalized_events=normalized_events,
            request_date=request.request_date
        )

        # 4. Calculate baseline safe_to_pay amount
        safe_amount = calculate_amount_safe_to_pay(state, request.requested_amount, spending_overrides={})

        # 5. Evaluate and rank candidate plans
        best_plan = evaluate_and_rank_plans(
            request=request,
            profile=profile,
            state=state,
            payment_options=options
        )

        earliest_date_str = best_plan.earliest_date_for_full_payment if best_plan.earliest_date_for_full_payment else ""

        return RecommendationResult(
            request_id=request.request_id,
            amount_safe_to_pay=safe_amount,
            affordability_status=best_plan.affordability_status,
            recommended_payment_method=best_plan.method,
            payment_plan=best_plan.payment_plan_str,
            earliest_date_for_full_payment=earliest_date_str,
            spending_changes_needed=best_plan.spending_changes_str,
            decision_explanation=best_plan.explanation
        )
